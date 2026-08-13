#ifndef DEEPSEEK_H
#define DEEPSEEK_H

/*
 * deepseek.h — DeepSeek V4 Flash 架构定义 (基于真实 config.json + model.py)
 *
 * V4 Flash 的 MLA 和 V2/V3 完全不同:
 *
 *   Query 路径:
 *     x[4096] → wq_a[1024, 4096] → c_q[1024]
 *             → q_norm (RMSNorm 1024)
 *             → wq_b[32768, 1024] → Q[64 heads × 512 dim]
 *             → per-head RMSNorm
 *             → RoPE 对最后 64 维旋转 (qk_rope_head_dim=64)
 *
 *   KV 路径 (MLA 核心):
 *     x[4096] → wkv[512, 4096] → kv[512]   ← 只存这个! KV Cache
 *             → kv_norm (RMSNorm 512)
 *             → RoPE 对最后 64 维旋转
 *     注意: V4 不显式解压 K/V! kv[512] 直接参与 attention
 *     Q[512] · kv[512] 做 attention (512 维向量内积)
 *
 *   Attention:
 *     sparse_attn: 不是全因果! 用 topk_idxs 选择要 attend 的位置
 *     滑动窗口: 最近 128 个 token 全精度
 *     压缩: 交替层把旧 KV 压缩 (ratio 4 或 128)
 *     attn_sink: 每头一个可学习偏置 (稳定性)
 *
 *   输出路径:
 *     o[64 heads × 512] → reshape 为 [8 groups, group_dim]
 *     → wo_a[8192, group_dim] (分组 LoRA)
 *     → wo_b[4096, 8192]
 *     → 输出[4096]
 *
 *   MoE MLP:
 *     x[4096] → router[256, 4096] → 选 top-6 专家
 *     每个专家: w1[2048,2048] w2[4096,1024] w3[2048,2048] (SwiGLU)
 *     + 1 个共享专家
 *
 *   量化:
 *     attention 权重: F8_E4M3 (权重) + F8_E8M0 (scale), block [128,128]
 *     expert 权重: I8 (权重) + F8_E8M0 (scale)
 */

#include <stdint.h>
#include <stdio.h>

/* ============ DeepSeek V4 Flash 配置 (来自 config.json) ============ */
typedef struct {
    /* 基础 */
    int   hidden_size;            /* 4096 */
    int   num_hidden_layers;      /* 43 */
    int   vocab_size;             /* 129280 */
    int   max_position;           /* 1048576 (1M) */
    float rope_theta;             /* 10000 */
    float rms_norm_eps;           /* 1e-6 */
    int   tie_word_embeddings;    /* 0 (false, 有独立 lm_head) */
    int   bos_token_id;           /* 0 */
    int   eos_token_id;           /* 1 */

    /* MLA Attention */
    int   n_heads;                /* 64 */
    int   head_dim;               /* 512 */
    int   q_lora_rank;            /* 1024 (query 压缩维度) */
    int   qk_rope_head_dim;       /* 64 (RoPE 维度, head_dim 的最后 64 维) */
    int   nope_head_dim;          /* 448 (= head_dim - rope_head_dim) */
    int   o_lora_rank;            /* 1024 */
    int   o_groups;               /* 8 (输出分组数) */

    /* 滑动窗口 + 压缩 */
    int   sliding_window;         /* 128 */
    int   compress_rope_theta;    /* 160000 */

    /* MoE */
    int   n_routed_experts;       /* 256 */
    int   n_shared_experts;       /* 1 */
    int   num_experts_per_tok;    /* 6 (top-6) */
    int   moe_intermediate_size;  /* 2048 */
    int   num_hash_layers;        /* 3 (前3层用 hash routing) */
    float routed_scaling_factor;  /* 1.5 */
    float swiglu_limit;           /* 10.0 */

    /* DSpark */
    int   dspark_block_size;      /* 5 */
    int   dspark_noise_token_id;  /* 128799 */

    /* Index attention (稀疏注意力) */
    int   index_head_dim;         /* 128 */
    int   index_n_heads;          /* 64 */
    int   index_topk;             /* 512 */
} DSConfig;

/* ============ MLA Attention 权重 (一层的) ============ */
/* 张量名对应 safetensors 里的 layers.{l}.attn.xxx */
typedef struct {
    /* Query 压缩 */
    float *wq_a;          /* [1024, 4096] = [q_lora_rank, hidden] */
    float *wq_a_scale;    /* FP8 scale [8, 32] */
    float *wq_b;          /* [32768, 1024] = [n_heads*head_dim, q_lora_rank] */
    float *wq_b_scale;    /* FP8 scale [256, 8] */
    float *q_norm;        /* [1024] RMSNorm weight */

    /* KV 压缩 (MLA 核心 — 只存这个 latent!) */
    float *wkv;           /* [512, 4096] = [head_dim, hidden] */
    float *wkv_scale;     /* FP8 scale [4, 32] */
    float *kv_norm;       /* [512] RMSNorm weight */

    /* 输出投影 (分组 LoRA) */
    float *wo_a;          /* [8192, 4096] = [o_groups*o_lora_rank, hidden/o_groups] */
    float *wo_a_scale;    /* FP8 scale [64, 32] */
    float *wo_b;          /* [4096, 8192] = [hidden, o_groups*o_lora_rank] */
    float *wo_b_scale;    /* FP8 scale [32, 64] */

    /* Attention sink (稳定性) */
    float *attn_sink;     /* [64] = [n_heads] 每头一个偏置 */

    /* 输入 RMSNorm */
    float *attn_norm;     /* [4096] */
} MLAWeights;

/* ============ MoE 专家 ============ */
/* 单个专家 = SwiGLU MLP (张量名: layers.{l}.ffn.experts.{e}.w{1,2,3}) */
typedef struct {
    float *w1;            /* [2048, 2048] gate weight */
    float *w1_scale;      /* FP8 scale [2048, 128] */
    float *w2;            /* [4096, 1024] down weight */
    float *w2_scale;      /* FP8 scale [4096, 64] */
    float *w3;            /* [2048, 2048] up weight */
    float *w3_scale;      /* FP8 scale [2048, 128] */
} Expert;

/* 一层的 MoE 权重 */
typedef struct {
    /* Router */
    float *gate_weight;           /* [256, 4096] = [n_experts, hidden] */
    float *gate_bias;             /* [256] = [n_experts] (hash 层不用) */

    /* 256 个路由专家 (磁盘上, 按需加载) */
    Expert *experts;              /* [256] */

    /* 1 个共享专家 (常驻内存) */
    Expert shared_expert;

    /* Hash routing (前3层) */
    int *tid2eid;                 /* [vocab_size, num_experts_per_tok] hash 表 */

    /* FFN RMSNorm */
    float *ffn_norm;             /* [4096] */
} MoEWeights;

/* ============ KV Cache (V4 压缩版) ============ */
/*
 * V4 的 KV Cache 设计:
 *   滑动窗口区: 最近 128 个 token 的 kv[512] (全精度)
 *   压缩区: 压缩后的旧 kv (compress_ratio 决定压缩程度)
 *
 * 每位置只存 kv[512] (不是 K+V!)
 * 比标准 attention 省: (2 × n_heads × head_dim) / head_dim = 2 × 64 = 128×
 */
typedef struct {
    /* 滑动窗口区 (常驻 RAM, 全精度) */
    float *win_kv;               /* [n_layers, window_size, head_dim] = [43, 128, 512] */
    float *win_kr;               /* [n_layers, window_size, rope_dim] = [43, 128, 64] */

    /* 压缩区 (按层, 有的层 ratio=0 不压缩, 有的 ratio=4 或 128) */
    float *compress_kv;          /* 压缩后的旧 KV (mmap 到 SSD) */

    int current_win_pos;         /* 滑动窗口当前写入位置 */
} DSKVCache;

/* ============ 完整模型 ============ */
typedef struct {
    DSConfig cfg;

    /* Embedding + LM Head (不 tie) */
    float *token_embedding;       /* [129280, 4096] */
    float *lm_head;               /* [129280, 4096] */

    /* 43 层 */
    MLAWeights *mla;              /* [43] */
    MoEWeights *moe;              /* [43] */

    /* 最终 norm */
    float *final_norm;           /* [4096] */

    /* 运行状态 */
    DSKVCache kv_cache;
} DeepSeekModel;

/* ============ 初始化 ============ */

static void deepseek_v4_config(DSConfig *c) {
    c->hidden_size            = 4096;
    c->num_hidden_layers      = 43;
    c->vocab_size             = 129280;
    c->max_position           = 1048576;
    c->rope_theta             = 10000.0f;
    c->rms_norm_eps           = 1e-6f;
    c->tie_word_embeddings    = 0;
    c->bos_token_id           = 0;
    c->eos_token_id           = 1;

    c->n_heads                = 64;
    c->head_dim               = 512;
    c->q_lora_rank            = 1024;
    c->qk_rope_head_dim       = 64;
    c->nope_head_dim          = c->head_dim - c->qk_rope_head_dim; /* 448 */
    c->o_lora_rank            = 1024;
    c->o_groups               = 8;

    c->sliding_window         = 128;
    c->compress_rope_theta    = 160000;

    c->n_routed_experts       = 256;
    c->n_shared_experts       = 1;
    c->num_experts_per_tok    = 6;
    c->moe_intermediate_size  = 2048;
    c->num_hash_layers        = 3;
    c->routed_scaling_factor  = 1.5f;
    c->swiglu_limit           = 10.0f;

    c->dspark_block_size      = 5;
    c->dspark_noise_token_id  = 128799;

    c->index_head_dim         = 128;
    c->index_n_heads          = 64;
    c->index_topk             = 512;
}

/* KV Cache 内存分析 */
void deepseek_kv_cache_analysis(DSConfig *c);

/* ============ 前向传播函数 (在 deepseek.c 里实现) ============ */
void ds_mla_forward(float *out, const float *x, const MLAWeights *w,
                    const DSConfig *cfg, int pos,
                    float *kv_win, int *win_pos);
void ds_moe_route(int *selected, float *weights,
                  const float *x, const MoEWeights *moe,
                  const DSConfig *cfg, int layer_id);
void ds_moe_forward(float *out, const float *x,
                    const MoEWeights *moe,
                    const DSConfig *cfg, int layer_id);
void ds_layer_forward(float *x, const MLAWeights *mla, const MoEWeights *moe,
                      const DSConfig *cfg, int layer_id, int pos,
                      float *kv_win, int *win_pos);

/* ============ KV Cache 分析 (在 deepseek.c 实现) ============ */

#endif // DEEPSEEK_H
