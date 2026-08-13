#ifndef DEEPSEEK_H
#define DEEPSEEK_H

/*
 * deepseek.h — DeepSeek V4 Flash 架构定义
 *
 * 来自 deepseek_config.json (DeepSeek-V4-Flash-0731) 的精确参数。
 *
 * 和 Qwen2.5 的核心差异:
 *   1. MLA Attention (低秩压缩 KV, 解耦 RoPE)
 *   2. MoE MLP (256 路由专家 + 1 共享专家, top-6)
 *   3. 原生 FP4 专家权重 + FP8 量化
 *   4. DSpark 投机解码 (block_size=5)
 *   5. YaRN RoPE scaling (支持 1M 上下文)
 *   6. 层级压缩 (alternating compress ratios)
 *   7. Hash routing (前 3 层) + sqrtsoftplus scoring
 *   8. Index attention (稀疏注意力, 1M 上下文)
 */

#include "net.h"  /* 复用 Config 基础类型 */
#include <stdint.h>
#include <stdio.h>

/* ============ DeepSeek V4 Flash 配置 ============ */
typedef struct {
    /* 基础参数 */
    int   hidden_size;            /* 4096 */
    int   num_hidden_layers;      /* 43 */
    int   vocab_size;             /* 129280 */
    int   max_position;           /* 1048576 (1M) */
    float rope_theta;             /* 10000 */
    float rms_norm_eps;           /* 1e-6 */
    int   tie_word_embeddings;    /* false → 有独立 lm_head */
    int   bos_token_id;           /* 0 */
    int   eos_token_id;           /* 1 */

    /* MLA Attention 参数 */
    int   num_attention_heads;    /* 64 */
    int   head_dim;               /* 512 */
    int   num_key_value_heads;    /* 1 (MLA 只用 1 个 KV 头) */
    int   q_lora_rank;            /* 1024 (query 压缩维度) */
    int   qk_rope_head_dim;       /* 64 (解耦 RoPE 维度) */
    int   o_lora_rank;            /* 1024 (输出压缩) */
    int   o_groups;               /* 8 */

    /* Index Attention (稀疏注意力, 1M 上下文) */
    int   index_head_dim;         /* 128 */
    int   index_n_heads;          /* 64 */
    int   index_topk;             /* 512 */
    int   sliding_window;         /* 128 */

    /* MoE 参数 */
    int   n_routed_experts;       /* 256 */
    int   n_shared_experts;       /* 1 */
    int   num_experts_per_tok;    /* 6 (top-6) */
    int   moe_intermediate_size;  /* 2048 (每个专家的 MLP 宽度) */
    int   num_hash_layers;        /* 3 (前 3 层用 hash routing) */
    float routed_scaling_factor;  /* 1.5 */
    float swiglu_limit;           /* 10.0 */

    /* DSpark 投机解码 */
    int   num_nextn_predict_layers; /* 1 */
    int   dspark_block_size;        /* 5 (一次猜 5 个 token) */
    int   dspark_noise_token_id;    /* 128799 */

    /* 量化 */
    /* 原生 FP8 (e4m3) + 专家 FP4, block_size [128,128] */
} DeepSeekConfig;

/* ============ MLA Attention 权重 ============ */
/* 一层的 MLA attention 权重 */
typedef struct {
    /* Query 压缩 */
    float *W_DQ;          /* (q_lora_rank, hidden_size) = (1024, 4096) */
    float *W_UQ;          /* (n_heads * (head_dim - rope_dim), q_lora_rank) */

    /* KV 压缩 (MLA 核心) */
    float *W_DKV;         /* (kv_lora_rank, hidden_size) — 实际用 index_head_dim? */
    float *W_UK;          /* (n_heads * (head_dim - rope_dim), kv_compress) */
    float *W_UV;          /* (n_heads * head_dim, kv_compress) */

    /* 解耦 RoPE */
    float *W_KR;          /* (qk_rope_head_dim, hidden_size) = (64, 4096) */

    /* 输出投影 (带 LoRA 压缩) */
    float *W_O;           /* (hidden_size, n_heads * head_dim) — 或通过 o_lora_rank */

    /* RMSNorm */
    float *rms_weight;    /* (hidden_size,) */
} MLAWeights;

/* ============ MoE MLP 权重 ============ */
/* 单个专家 (就是一个小的 SwiGLU MLP) */
typedef struct {
    float *w_gate;        /* (moe_intermediate_size, hidden_size) = (2048, 4096) */
    float *w_up;          /* (2048, 4096) */
    float *w_down;        /* (4096, 2048) */
} ExpertWeights;

/* 一层的完整 MoE 权重 */
typedef struct {
    float *W_router;               /* (n_routed_experts, hidden_size) = (256, 4096) */
    ExpertWeights *experts;         /* [n_routed_experts] = [256] 个专家 */
    ExpertWeights shared_expert;    /* 1 个共享专家 (常驻内存) */
    float *rms_weight;             /* (hidden_size,) RMSNorm */
} MoEWeights;

/* ============ KV Cache (MLA 压缩版) ============ */
/* MLA 只存压缩 latent, 不存完整 K/V */
typedef struct {
    /* 热区: 最近 N 个 token, fp16, 在 RAM */
    float *hot_latent;     /* (n_layers, hot_size, kv_compress_dim) */
    float *hot_kr;         /* (n_layers, hot_size, qk_rope_head_dim) */
    int    hot_size;       /* 热区大小 (如 8192) */

    /* 温区: int4 量化, 在 RAM */
    uint8_t *warm_latent;  /* int4 打包 */
    int      warm_size;

    /* 冷区: mmap 到 SSD */
    float *cold_latent;    /* mmap, OS 自动分页 */
    int    cold_size;

    int current_pos;       /* 当前写到哪了 */
} DeepSeekKVCache;

/* ============ DSpark 草稿模块 ============ */
typedef struct {
    float *embed;          /* 和主模型共享 embedding */
    /* 几层轻量 transformer (1 层) */
    MLAWeights attn;
    MoEWeights moe;
    float *rms_weight;
    float *lm_head;        /* 输出投影 */
} DraftModule;

/* ============ 完整模型 ============ */
typedef struct {
    DeepSeekConfig cfg;

    /* Embedding */
    float *token_embedding;   /* (vocab_size, hidden_size) = (129280, 4096) */
    float *lm_head;           /* (vocab_size, hidden_size) — 不 tie! */

    /* 43 层: 每层有 MLA + MoE */
    MLAWeights *mla_layers;   /* [43] */
    MoEWeights *moe_layers;   /* [43] */

    /* 最终 norm */
    float *final_rms_weight;

    /* DSpark 草稿模块 */
    DraftModule draft;
} DeepSeekModel;

/* ============ 初始化 ============ */

/* 用 config.json 的值填充 DeepSeekConfig */
static void deepseek_v4_flash_config(DeepSeekConfig *c) {
    c->hidden_size            = 4096;
    c->num_hidden_layers      = 43;
    c->vocab_size             = 129280;
    c->max_position           = 1048576;
    c->rope_theta             = 10000.0f;
    c->rms_norm_eps           = 1e-6f;
    c->tie_word_embeddings    = 0;
    c->bos_token_id           = 0;
    c->eos_token_id           = 1;

    c->num_attention_heads    = 64;
    c->head_dim               = 512;
    c->num_key_value_heads    = 1;
    c->q_lora_rank            = 1024;
    c->qk_rope_head_dim       = 64;
    c->o_lora_rank            = 1024;
    c->o_groups               = 8;

    c->index_head_dim         = 128;
    c->index_n_heads          = 64;
    c->index_topk             = 512;
    c->sliding_window         = 128;

    c->n_routed_experts       = 256;
    c->n_shared_experts       = 1;
    c->num_experts_per_tok    = 6;
    c->moe_intermediate_size  = 2048;
    c->num_hash_layers        = 3;
    c->routed_scaling_factor  = 1.5f;
    c->swiglu_limit           = 10.0f;

    c->num_nextn_predict_layers = 1;
    c->dspark_block_size       = 5;
    c->dspark_noise_token_id   = 128799;
}

/* 参数量计算 */
static void deepseek_print_params(DeepSeekConfig *c) {
    long embed = (long)c->vocab_size * c->hidden_size;
    long lm_head = embed; /* 不 tie */

    /* MLA 每层 */
    long mla_per_layer =
        (long)c->q_lora_rank * c->hidden_size +           /* W_DQ */
        (long)c->num_attention_heads * (c->head_dim - c->qk_rope_head_dim) * c->q_lora_rank + /* W_UQ */
        /* W_DKV, W_UK, W_UV — 需要确认精确维度 */
        (long)c->num_attention_heads * c->head_dim * c->q_lora_rank +  /* W_UK + W_UV 估计 */
        (long)c->qk_rope_head_dim * c->hidden_size +     /* W_KR */
        (long)c->hidden_size * c->num_attention_heads * c->head_dim;   /* W_O */

    /* MoE 每层 */
    long expert_size = (long)c->moe_intermediate_size * c->hidden_size * 3; /* gate + up + down */
    long moe_per_layer =
        (long)c->n_routed_experts * c->hidden_size +     /* W_router */
        (long)c->n_routed_experts * expert_size +         /* 256 个专家 */
        expert_size;                                       /* 1 个共享专家 */

    long total_layers = (mla_per_layer + moe_per_layer) * c->num_hidden_layers;
    long total = embed + lm_head + total_layers;

    printf("DeepSeek V4 Flash 参数量:\n");
    printf("  Embedding:     %ld (%.1fB)\n", embed, embed/1e9);
    printf("  LM Head:       %ld (%.1fB)\n", lm_head, lm_head/1e9);
    printf("  MLA/层:        %ld (%.1fB)\n", mla_per_layer, mla_per_layer/1e9);
    printf("  MoE/层:        %ld (%.1fB)\n", moe_per_layer, moe_per_layer/1e9);
    printf("  43层合计:      %ld (%.1fB)\n", total_layers, total_layers/1e9);
    printf("  总参数:        %ld (%.1fB)\n", total, total/1e9);
    printf("\n");
    printf("  每token激活参数:\n");
    long active = mla_per_layer + expert_size * c->num_experts_per_tok + expert_size;
    printf("    MLA + 6专家 + 1共享 = %ld (%.1fB)\n", active, active/1e9);
    long active_total = active * c->num_hidden_layers;
    printf("    43层合计: %ld (%.1fB)\n", active_total, active_total/1e9);
}

#endif // DEEPSEEK_H
