/* deepseek.c — DeepSeek V4 Flash 推理引擎
 *
 * 基于官方 inference/model.py 翻译成 C。
 *
 * V4 Flash 一层的前向传播:
 *   x → attn_norm → MLA Attention → hc_residual → ffn_norm → MoE → hc_residual
 *
 * MLA Attention:
 *   x → wq_a[1024,4096] → q_norm → wq_b[32768,1024] → Q[64×512]
 *   Q per-head RMSNorm → RoPE(last 64 dims)
 *   x → wkv[512,4096] → kv_norm → kv[512] → RoPE(last 64 dims)
 *   attention: Q · kv (512维向量内积, 不是标准 head-wise)
 *   output: o → reshape[8 groups] → wo_a → wo_b → x_out
 *
 * MoE:
 *   x → gate[256,4096] → sqrtsoftplus → top-6 → 选专家
 *   out = Σ weight_i × expert_i(x) + shared_expert(x)
 *   expert = silu(x@w1) * (x@w3) @ w2  (SwiGLU, clamp ±10)
 *
 * 注意: 这是简化版, 跳过了:
 *   - sparse_attn (用标准 causal attention 替代)
 *   - Compressor/Indexer (滑动窗口 + 压缩)
 *   - Hyper-Connections (用标准残差替代)
 *   - FP8 量化 (用 fp32 替代)
 *   先跑通, 后优化。
 */

#include "deepseek.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================
 * RMSNorm (和 Qwen 的一样, 但用于 V4 的维度)
 * ============================================================ */
static void ds_rmsnorm(float *out, const float *x, const float *weight,
                       int size, float eps) {
    double ss = 0.0;
    for (int i = 0; i < size; i++) ss += (double)x[i] * x[i];
    ss = ss / size + eps;
    float inv = 1.0f / sqrtf((float)ss);
    for (int i = 0; i < size; i++)
        out[i] = (x[i] * inv) * weight[i];
}

/* ============================================================
 * RoPE — 只对最后 rope_dim 维应用旋转
 *
 * V4 的 RoPE 和 Qwen 不同:
 *   Qwen: 对整个 head_dim 用 half 模式
 *   V4:   head_dim=512, 只对最后 64 维 (qk_rope_head_dim) 旋转
 *         前 448 维 (nope_head_dim) 不旋转
 * ============================================================ */
static void ds_apply_rope(float *vec, int head_dim, int rope_dim,
                          int pos, float theta) {
    /* 只旋转最后 rope_dim 维, 用 half 模式 */
    int start = head_dim - rope_dim;
    int half = rope_dim / 2;
    for (int i = 0; i < half; i++) {
        float freq = powf(theta, (float)(-2*i) / rope_dim);
        float angle = pos * freq;
        float c = cosf(angle), s = sinf(angle);
        /* half 模式: vec[start+i] 和 vec[start+i+half] 配对 */
        float a = vec[start + i];
        float b = vec[start + i + half];
        vec[start + i]         = a * c - b * s;
        vec[start + i + half]  = a * s + b * c;
    }
}

/* ============================================================
 * 矩阵向量乘 (和 net.c 的 matmul 一样)
 * ============================================================ */
static void ds_matmul(float *out, const float *W, const float *x,
                      int n, int d) {
    /* out[d] = W[d,n] @ x[n] */
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        const float *wrow = W + (size_t)i * n;
        for (int j = 0; j < n; j++)
            val += wrow[j] * x[j];
        out[i] = val;
    }
}

/* ============================================================
 * MLA Attention 前向传播 (一层的 attention 部分)
 *
 * 输入: x[hidden_size], pos
 * 输出: out[hidden_size]
 * 使用: MLAWeights *w (这层的 attention 权重)
 * 临时: float *buf (工作内存, 至少 32768 个 float)
 * ============================================================ */
void ds_mla_forward(float *out,                    /* 输出 [hidden_size] */
                    const float *x,                 /* 输入 [hidden_size] */
                    const MLAWeights *w,            /* 这层 attention 权重 */
                    const DSConfig *cfg,            /* 模型配置 */
                    int pos,                         /* 当前位置 */
                    /* KV Cache: 滑动窗口区 */
                    float *kv_win,                  /* [window_size, head_dim] */
                    int *win_pos)                   /* 窗口写入位置 */
{
    int hidden = cfg->hidden_size;       /* 4096 */
    int q_lora = cfg->q_lora_rank;       /* 1024 */
    int n_heads = cfg->n_heads;          /* 64 */
    int hd = cfg->head_dim;             /* 512 */
    int rope_dim = cfg->qk_rope_head_dim; /* 64 */
    int win = cfg->sliding_window;       /* 128 */
    int o_groups = cfg->o_groups;        /* 8 */
    int o_lora = cfg->o_lora_rank;       /* 1024 */
    float eps = cfg->rms_norm_eps;

    /* ---- ① Query 路径 ---- */
    /* c_q = wq_a @ x → [1024] */
    float *c_q = (float *)calloc(q_lora, sizeof(float));
    ds_matmul(c_q, w->wq_a, x, hidden, q_lora);

    /* q_norm: RMSNorm on c_q */
    ds_rmsnorm(c_q, c_q, w->q_norm, q_lora, eps);

    /* Q = wq_b @ c_q → [32768] = [64 heads × 512 dim] */
    float *Q = (float *)calloc(n_heads * hd, sizeof(float));
    ds_matmul(Q, w->wq_b, c_q, q_lora, n_heads * hd);

    /* Per-head RMSNorm: 每个 512 维 head 独立归一化 */
    for (int h = 0; h < n_heads; h++) {
        float *qh = Q + h * hd;
        double ss = 0.0;
        for (int i = 0; i < hd; i++) ss += (double)qh[i] * qh[i];
        float inv = 1.0f / sqrtf((float)(ss / hd + eps));
        for (int i = 0; i < hd; i++) qh[i] *= inv;
    }

    /* RoPE: 对每个 head 的最后 64 维旋转 */
    for (int h = 0; h < n_heads; h++) {
        ds_apply_rope(Q + h * hd, hd, rope_dim, pos, cfg->rope_theta);
    }

    /* ---- ② KV 路径 ---- */
    /* kv = wkv @ x → [512] */
    float kv[512];  /* V4 的 MLA 核心: kv[512] 直接参与 attention */
    ds_matmul(kv, w->wkv, x, hidden, hd);

    /* kv_norm */
    ds_rmsnorm(kv, kv, w->kv_norm, hd, eps);

    /* RoPE: 对 kv 的最后 64 维旋转 */
    ds_apply_rope(kv, hd, rope_dim, pos, cfg->rope_theta);

    /* ---- ③ KV Cache: 写入滑动窗口 ---- */
    int wp = *win_pos;
    memcpy(kv_win + wp * hd, kv, hd * sizeof(float));
    wp = (wp + 1) % win;
    *win_pos = wp;

    /* ---- ④ Attention 计算 ---- */
    /* 简化版: 标准因果 attention (不是 sparse_attn)
     * 每个 head: score = Q[h] · kv[t], t = 0..window_used
     * V4 的特点: Q[h] 和 kv 做整个 512 维的点积 (不是 head-wise K/V)
     *
     * 实际 V4 用 sparse_attn: 只选 top-k 个位置做 attention
     * 这里先用标准版, 后续替换为 sparse */
    int n_kv = (pos < win) ? (pos + 1) : win;
    float scale = 1.0f / sqrtf((float)hd);

    float *attn_out = (float *)calloc(n_heads * hd, sizeof(float));

    for (int h = 0; h < n_heads; h++) {
        float *qh = Q + h * hd;

        /* 算分数 */
        float *scores = (float *)calloc(n_kv, sizeof(float));
        float max_score = -1e30f;
        for (int t = 0; t < n_kv; t++) {
            float *kt = kv_win + t * hd;
            float s = 0.0f;
            for (int i = 0; i < hd; i++) s += qh[i] * kt[i];
            s = s * scale + w->attn_sink[h];  /* attn_sink 偏置 */
            scores[t] = s;
            if (s > max_score) max_score = s;
        }

        /* softmax */
        float sum = 0.0f;
        for (int t = 0; t < n_kv; t++) {
            scores[t] = expf(scores[t] - max_score);
            sum += scores[t];
        }
        float inv_sum = 1.0f / sum;

        /* 加权求和 → attn_out[h] */
        float *oh = attn_out + h * hd;
        for (int t = 0; t < n_kv; t++) {
            float *kt = kv_win + t * hd;
            float a = scores[t] * inv_sum;
            for (int i = 0; i < hd; i++) oh[i] += a * kt[i];
        }
        free(scores);
    }

    /* 对 attn_out 的最后 64 维逆旋转 RoPE (官方代码有 inverse RoPE) */
    for (int h = 0; h < n_heads; h++) {
        float *oh = attn_out + h * hd;
        /* 逆旋转: 简化处理 (实际 V4 有 inverse rotary) */
        /* TODO: implement inverse RoPE */
    }

    /* ---- ⑤ 输出投影 (分组 LoRA) ---- */
    /* o = attn_out reshape 为 [o_groups, n_heads*hd/o_groups]
     * wo_a: [o_groups * o_lora_rank, group_dim]
     * wo_b: [hidden, o_groups * o_lora_rank] */
    int group_dim = n_heads * hd / o_groups;  /* 32768/8 = 4096 */

    /* o_mid = wo_a @ o_reshaped → [o_groups * o_lora_rank] = [8192] */
    float *o_mid = (float *)calloc(o_groups * o_lora, sizeof(float));
    for (int g = 0; g < o_groups; g++) {
        /* 每组独立做 matmul */
        float *group_in = attn_out + g * group_dim;
        float *group_out = o_mid + g * o_lora;
        /* wo_a 这组的权重: [o_lora, group_dim] */
        float *wa_g = w->wo_a + g * o_lora * group_dim;
        ds_matmul(group_out, wa_g, group_in, group_dim, o_lora);
    }

    /* out = wo_b @ o_mid → [hidden] */
    ds_matmul(out, w->wo_b, o_mid, o_groups * o_lora, hidden);

    /* 清理 */
    free(c_q);
    free(Q);
    free(attn_out);
    free(o_mid);
}

/* ============================================================
 * MoE 路由 (Gate)
 *
 * 输入: x[hidden_size], layer_id (决定用 hash 还是 score routing)
 * 输出: selected[6] (选中的专家 id), weights[6] (路由权重)
 * ============================================================ */
void ds_moe_route(int *selected, float *weights,
                  const float *x, const MoEWeights *moe,
                  const DSConfig *cfg, int layer_id)
{
    int n_experts = cfg->n_routed_experts;  /* 256 */
    int top_k = cfg->num_experts_per_tok;   /* 6 */
    int hidden = cfg->hidden_size;
    float scale = cfg->routed_scaling_factor;

    /* ① 算分数: scores = x @ gate_weight^T */
    float *scores = (float *)malloc(n_experts * sizeof(float));
    for (int e = 0; e < n_experts; e++) {
        scores[e] = 0.0f;
        float *wrow = moe->gate_weight + (size_t)e * hidden;
        for (int j = 0; j < hidden; j++)
            scores[e] += wrow[j] * x[j];
    }

    /* ② sqrtsoftplus 激活 (V4 独有, 不是 softmax) */
    for (int e = 0; e < n_experts; e++) {
        /* sqrtsoftplus: sqrt(softplus(x)) = sqrt(ln(1+e^x)) */
        scores[e] = sqrtf(logf(1.0f + expf(scores[e])));
    }

    /* ③ 保存原始分数 (用于权重计算) */
    float orig_scores[256];
    for (int e = 0; e < n_experts; e++) orig_scores[e] = scores[e];

    /* ④ 加 bias (影响 top-k 选择, 不影响权重) */
    if (moe->gate_bias && layer_id >= cfg->num_hash_layers) {
        for (int e = 0; e < n_experts; e++)
            scores[e] += moe->gate_bias[e];
    }

    /* ⑤ 选 top-k */
    char *used = (char *)calloc(n_experts, 1);
    for (int k = 0; k < top_k; k++) {
        int best = -1;
        float bestv = -1e30f;
        for (int e = 0; e < n_experts; e++) {
            if (!used[e] && scores[e] > bestv) {
                bestv = scores[e];
                best = e;
            }
        }
        selected[k] = best;
        used[best] = 1;
    }
    free(used);

    /* ⑥ 用原始分数算权重, 归一化, 乘 scale */
    float wsum = 0.0f;
    for (int k = 0; k < top_k; k++) {
        weights[k] = orig_scores[selected[k]];
        wsum += weights[k];
    }
    for (int k = 0; k < top_k; k++) {
        weights[k] = (weights[k] / wsum) * scale;
    }

    free(scores);
}

/* ============================================================
 * 单个专家前向 (SwiGLU MLP)
 *
 * out = silu(clamp(x @ w1, max=limit)) * clamp(x @ w3, ±limit) @ w2
 * ============================================================ */
static void ds_expert_forward(float *out, const float *x,
                               const Expert *expert,
                               int dim, int inter_dim, float swiglu_limit)
{
    float *gate = (float *)malloc(inter_dim * sizeof(float));
    float *up   = (float *)malloc(inter_dim * sizeof(float));

    /* gate = x @ w1 */
    ds_matmul(gate, expert->w1, x, inter_dim, inter_dim);
    /* up = x @ w3 */
    ds_matmul(up, expert->w3, x, inter_dim, inter_dim);

    /* SwiGLU with clamp */
    for (int i = 0; i < inter_dim; i++) {
        float g = gate[i];
        float u = up[i];
        if (swiglu_limit > 0) {
            if (g > swiglu_limit) g = swiglu_limit;
            if (u > swiglu_limit) u = swiglu_limit;
            if (u < -swiglu_limit) u = -swiglu_limit;
        }
        gate[i] = (g / (1.0f + expf(-g))) * u;  /* silu(g) * u */
    }

    /* out = gate @ w2 */
    ds_matmul(out, expert->w2, gate, inter_dim, dim);

    free(gate);
    free(up);
}

/* ============================================================
 * MoE 前向 (一层)
 *
 * out = Σ weight_k × expert_k(x) + shared_expert(x)
 * ============================================================ */
void ds_moe_forward(float *out, const float *x,
                    const MoEWeights *moe,
                    const DSConfig *cfg, int layer_id)
{
    int hidden = cfg->hidden_size;          /* 4096 */
    int inter = cfg->moe_intermediate_size; /* 2048 */
    int top_k = cfg->num_experts_per_tok;   /* 6 */
    float limit = cfg->swiglu_limit;        /* 10.0 */

    /* ① 路由: 选 top-6 专家 */
    int selected[6];
    float weights[6];
    ds_moe_route(selected, weights, x, moe, cfg, layer_id);

    /* ② 累加 6 个选中专家的输出 */
    float *expert_out = (float *)calloc(hidden, sizeof(float));
    float *tmp = (float *)malloc(hidden * sizeof(float));

    for (int k = 0; k < top_k; k++) {
        int eid = selected[k];
        ds_expert_forward(tmp, x, &moe->experts[eid], hidden, inter, limit);
        for (int i = 0; i < hidden; i++)
            expert_out[i] += weights[k] * tmp[i];
    }

    /* ③ 加上共享专家 */
    ds_expert_forward(tmp, x, &moe->shared_expert, hidden, inter, limit);
    for (int i = 0; i < hidden; i++)
        expert_out[i] += tmp[i];

    memcpy(out, expert_out, hidden * sizeof(float));

    free(expert_out);
    free(tmp);
}

/* ============================================================
 * 完整一层的前向传播 (简化版, 用标准残差替代 Hyper-Connection)
 *
 * 输入: x[hidden_size]
 * 输出: x[hidden_size] (原地更新)
 * ============================================================ */
void ds_layer_forward(float *x,                      /* 输入/输出 [hidden_size] */
                      const MLAWeights *mla,         /* 这层 attention 权重 */
                      const MoEWeights *moe,         /* 这层 MoE 权重 */
                      const DSConfig *cfg,
                      int layer_id, int pos,
                      /* KV Cache (每层独立的滑动窗口) */
                      float *kv_win, int *win_pos)
{
    int hidden = cfg->hidden_size;
    float eps = cfg->rms_norm_eps;
    float limit = cfg->swiglu_limit;

    /* ---- Attention ---- */
    float *attn_in = (float *)malloc(hidden * sizeof(float));
    ds_rmsnorm(attn_in, x, mla->attn_norm, hidden, eps);

    float *attn_out = (float *)malloc(hidden * sizeof(float));
    ds_mla_forward(attn_out, attn_in, mla, cfg, pos, kv_win, win_pos);

    /* 残差 (简化版, 跳过 Hyper-Connection) */
    for (int i = 0; i < hidden; i++) x[i] += attn_out[i];

    /* ---- MoE ---- */
    float *ffn_in = (float *)malloc(hidden * sizeof(float));
    ds_rmsnorm(ffn_in, x, moe->ffn_norm, hidden, eps);

    float *ffn_out = (float *)malloc(hidden * sizeof(float));
    ds_moe_forward(ffn_out, ffn_in, moe, cfg, layer_id);

    /* 残差 */
    for (int i = 0; i < hidden; i++) x[i] += ffn_out[i];

    free(attn_in); free(attn_out);
    free(ffn_in); free(ffn_out);
}

/* ============================================================
 * KV Cache 内存分析
 * ============================================================ */
void deepseek_kv_cache_analysis(DSConfig *c) {
    int win = c->sliding_window;
    int hd = c->head_dim;
    int L  = c->num_hidden_layers;
    long win_kv = (long)L * win * hd * 4;
    long standard = (long)L * win * 2 * c->n_heads * hd * 4;

    printf("=== V4 Flash KV Cache 分析 ===\n\n");
    printf("滑动窗口区 (%d token):\n", win);
    printf("  每位置: kv[%d] = %d 字节\n", hd, hd*4);
    printf("  %d 层 = %ld MB\n\n", L, win_kv / (1024*1024));
    printf("标准 attention:\n");
    printf("  %ld MB\n\n", standard/(1024*1024));
    printf("MLA 压缩比: %.0fx\n", (double)standard / win_kv);
    printf("滑动窗口仅占: %ld MB RAM\n", win_kv / (1024*1024));
}
