# 第 11 章 · DeepSeek V4 Flash 支持方案

> 第 11 章 | deepseek-support 分支 | [上一章：第 10 章 生态对比](./10-ecosystem) | [目录](./README)
>
> 目标：在 18GB M3 Pro 上运行 284B MoE 大模型，目标 50 tok/s。

---

## 11.1 为什么选 DeepSeek V4 Flash

| | Qwen2.5-0.5B (当前) | DeepSeek V4 Flash |
|---|---|---|
| 总参数 | 0.49B | 284B |
| 激活参数 | 0.49B | 13B (MoE) |
| Attention | GQA (14Q, 2KV) | MLA (低秩压缩 KV) |
| MLP | SwiGLU (单一) | MoE (256 专家 + 1 共享) |
| 层数 | 24 | 43 (全 MoE) |
| 量化 | 无 | 原生 FP4/FP8/INT4 (QAT) |
| 上下文 | 32K | 1M |
| License | Apache 2.0 | MIT |

DeepSeek V4 Flash 的核心优势：**284B 参数的大模型能力，但每次推理只用 13B**（MoE），加上原生量化训练（QAT），在有限硬件上运行成为可能。

---

## 11.2 架构差异：Qwen vs DeepSeek

```
Qwen2.5 一层:
  x → RMSNorm → [QKV投影(GQA)] → RoPE → Attention → Wo → +残差
    → RMSNorm → [SwiGLU MLP] → +残差

DeepSeek V4 Flash 一层:
  x → RMSNorm → [MLA压缩] → [解耦RoPE] → MLA Attention → +残差
    → RMSNorm → [Router选6个专家] → 6×Expert + 1×Shared → +残差
```

三处核心差异：MLA Attention、MoE MLP、解耦 RoPE。

---

## 11.3 MLA (Multi-head Latent Attention)

### 和 GQA 的区别

标准 attention (我们当前的 GQA)：
```
每个 token 存: K[128维] + V[128维] = 256 个 float
43 层 × 1M 位置 = 43 × 1M × 256 × 4字节 = 44 GB (KV Cache)
```

MLA：
```
每个 token 只存压缩 latent: c_kv[512维] + k_r[64维] = 576 个 float
43 层 × 1M 位置 = 43 × 1M × 576 × 4字节 = 99 GB

看起来差不多？但 MLA 解压后是 128 头 × 128 维 = 16384 维
标准 attention 存 128 头 × 128 维 × 2(K和V) = 32768 维
MLA 压缩比: 32768 / 576 = 57×
```

### MLA 前向传播步骤

```
① 压缩: c_kv = x @ W_DKV          (d_model → d_c, 比如 7168 → 512)

② KV Cache 只存 c_kv (不存完整 K/V!)

③ 需要时解压:
   K = c_kv @ W_UK                  (d_c → n_heads × d_head)
   V = c_kv @ W_UV                  (d_c → n_heads × d_head)

④ 解耦 RoPE (和标准 RoPE 不同):
   k_r = x @ W_KR                   (d_model → d_r, 所有头共享)
   q_r = c_q @ W_QR                 (d_c → d_r)
   对 q_r 和 k_r 应用 RoPE 旋转

⑤ Attention 分数 (两部分相加):
   score = (K · Q + k_r · q_r) / sqrt(d_head + d_r)

⑥ softmax → 加权 V → 输出投影
```

### 关键矩阵

| 矩阵 | 形状 | 作用 |
|------|------|------|
| W_DKV | (d_c, d_model) | 压缩 KV |
| W_UK | (n_heads × d_head, d_c) | 解压 K |
| W_UV | (n_heads × d_head, d_c) | 解压 V |
| W_KR | (d_r, d_model) | RoPE key (所有头共享) |
| W_QR | (d_r, d_c) | RoPE query |
| W_O | (d_model, n_heads × d_head) | 输出投影 |

### 为什么 MLA 省 KV Cache 内存

```
标准: cache 存 K + V = 2 × n_heads × d_head × seq_len × n_layers
MLA:  cache 存 c_kv + k_r = (d_c + d_r) × seq_len × n_layers

DeepSeek V4 Flash:
  标准: 2 × 128 × 128 = 32768 个 float/位置
  MLA:  512 + 64 = 576 个 float/位置
  压缩: 57×
```

---

## 11.4 MoE (Mixture of Experts)

### 和标准 MLP 的区别

```
标准 MLP (我们的 SwiGLU):
  out = silu(x @ W_gate) * (x @ W_up) @ W_down
  → 每层一套权重, 每个 token 都走全部参数

MoE:
  ① router_logits = x @ W_router    → 256 个分数
  ② 选 top-6: 分数最高的 6 个专家
  ③ out = shared_expert(x) + Σ weight_i × expert_i(x)
  → 每层 256 个专家, 但每个 token 只激活 6 个
```

### 每个专家就是一个小的 MLP

```c
// 单个专家 (就是一个 SwiGLU):
float *expert_forward(float *x, ExpertWeights *ew) {
    // 和我们的 SwiGLU 完全一样的结构, 只是更小
    matmul(gate, ew->w_gate, x);
    matmul(up, ew->w_up, x);
    for (int i = 0; i < expert_dim; i++)
        gate[i] = silu(gate[i]) * up[i];
    matmul(out, ew->w_down, gate);
    return out;
}
```

### Router 实现

```c
// 路由器: 决定当前 token 用哪 6 个专家
void moe_route(float *router_logits, int n_experts, int top_k,
               int *selected, float *weights) {
    // router_logits: 256 个分数
    // 选 top_k=6 个最高的
    // 用 softmax 归一化权重
    for (int i = 0; i < top_k; i++) {
        // 找最大值, 记录 index
        // softmax 得到 weight
    }
}
```

### MoE 的内存优势 (关键!)

```
284B 总参数:
  共享层 (attention + embedding):  ~28B
  256 个专家 × 每个 ~1B:           ~256B
  总计:                            ~284B

每个 token 只激活:
  共享层 28B + 6 个专家 × 1B = 34B (实际约 13B, 因为专家更小)
  → 只需读 13B 参数, 不是 284B!
  → int4: 只需 6.5 GB 内存带宽/token
```

---

## 11.5 mmap 磁盘卸载方案

### 核心思路

```
模型文件 142GB (int4) → mmap 映射到虚拟内存
macOS 自动分页: 热门专家留 RAM, 冷门换 SSD

不需要手写专家加载逻辑!
  - mmap 142GB 文件, 只占虚拟地址空间
  - CPU 访问某页时, OS 自动从 SSD 加载到 RAM
  - 不访问的页, OS 自动换出到 SSD (LRU)
  - 对程序来说: 访问 142GB 空间和访问 1GB 没区别
```

### 为什么在 M3 Pro 上特别有效

```
M3 Pro 统一内存架构:
  1. CPU 和 GPU 共享同一块 18GB → 不需要拷贝
  2. NVMe SSD 7GB/s → 换页速度极快
  3. macOS page cache → 自动 LRU 管理
  4. Metal GPU 可以直接访问 mmap 的数据
```

### mmap + madvise 优化

```c
// 基础 mmap (已有):
void *map = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);

// 优化: 告诉 OS 接下来要访问哪些页
// 在计算第 N 层时, 预取第 N+1 层的专家
posix_madvise(next_layer_experts, size, POSIX_MADV_WILLNEED);

// 优化: 告诉 OS 哪些页不再需要 (主动释放)
posix_madvise(old_experts, size, POSIX_MADV_DONTNEED);
```

---

## 11.6 KV Cache 五级压缩方案

### 问题有多大

```
1M 上下文, 标准 KV Cache (fp16):
  2 × 43层 × 1M × 128头 × 128维 × 2字节 = 281 GB → 不可能

MLA 压缩后 (fp16):
  43层 × 1M × (512+64) × 2字节 = 49 GB → 还是太大

需要更多压缩 ↓
```

### 五级方案

**L1: MLA 压缩 (架构级, 57×)**
```
不存完整 K/V, 只存 latent c_kv[d_c] + k_r[d_r]
每位置: 32768 float → 576 float
```

**L2: KV Cache 量化 (2-4×)**
```
latent 用 int4 存储 (而不是 fp16)
每位置: 576 × 2字节 → 576 × 0.5字节
```

**L3: 动态分配 (按需)**
```
不预分配 seq_len 大小, 只为实际使用的位置分配
对话只有 1000 个 token 时, 只占 1000 个位置的内存
```

**L4: 热冷分离 (核心优化)**

```
不是所有历史 token 都同等重要:

  热区 (RAM): 最近 8K token, fp16, 完整精度
  温区 (RAM): 8K~128K, int4 量化
  冷区 (SSD): 128K~1M, mmap 自动分页

三级结构:
  [热区 8K]    MLA fp16  → 0.4 GB RAM
  [温区 120K]  MLA int4  → 1.5 GB RAM
  [冷区 896K]  mmap SSD  → 0 GB RAM (OS 管理)
  ────────────────────
  总 RAM:              ~1.9 GB
```

**L5: 滑动窗口 + 远程摘要**
```
极端长上下文时:
  超过 1M 的早期内容 → 压缩成摘要向量
  不保留逐 token 的 KV, 只存一个 pooled 表示
```

### 代码结构

```c
typedef struct {
    float  *hot_latent;     // 热区: 最近 N 个 token, fp16
    uint8_t *warm_latent;   // 温区: int4 量化
    float  *cold_latent;    // 冷区: mmap 到 SSD
    int    *position_map;   // 位置 → 属于哪一级
    int     current_pos;
} CompressedKVCache;
```

---

## 11.7 性能优化路线 (目标 50 tok/s)

### 瓶颈分析

```
每 token 需要:
  计算: 13B × 2 FLOP = 26 GFLOP
  读取: 13B × 0.5字节(int4) = 6.5 GB 权重

M3 Pro 能力:
  GPU 计算: ~30 TOPS (int4) → 26G / 30T = 0.87ms (不是瓶颈!)
  内存带宽: 150 GB/s → 6.5G / 150G = 43ms (瓶颈!)
  SSD 带宽: 7 GB/s

结论: 瓶颈是内存带宽, 不是计算
```

### 八级优化路线

```
基线: mmap + CPU fp32                      →   0.1 tok/s
+ int4 量化 (权重 4× 压缩)                  →   0.5 tok/s
+ OpenMP (5 个 P 核并行)                    →   2 tok/s
+ Metal GPU (14 核 GPU 做 matmul)           →   8 tok/s
+ 专家预取 + madvise (I/O 隐藏在计算后)      →  15 tok/s
+ DSpark 投机解码 (1.85× 加速)              →  28 tok/s
+ GPU/CPU 异步流水线 (CPU 预取, GPU 计算)    →  35 tok/s
+ 专家亲和性缓存 (命中率 95%)                →  45 tok/s
+ MLA 低秩 KV Cache (省内存给专家缓存)       →  50 tok/s ✓
```

### 参考数据

| 硬件 | 引擎 | 速度 | 技术 |
|------|------|------|------|
| NVIDIA B200 | vLLM + DSpark | 120 tok/s | 全部优化 |
| M3 Pro (理论) | 极致优化 | ~50 tok/s | 8 重叠加 |
| M3 Pro (保守) | llama.cpp | ~10-20 tok/s | Metal + GGUF |

---

## 11.8 DSpark 投机解码

### 原理

```
传统: 每生成 1 个 token, 跑一次完整 forward (43 层)

DSpark:
  ① 草稿模块 (MTP) 快速预测 3 个候选 token
  ② 大模型一次 forward 验证这 3 个
  ③ 猜对的直接用 (通常 2-3 个都对)
  
  等于: 一次 forward 生成 2-3 个 token
  加速: 1.85× (DeepSeek 官方数据, 无损)
```

### 在 M3 Pro 上的应用

```
草稿模块很轻量 (几层 transformer):
  草稿模块常驻 RAM (~0.5 GB)
  快速猜 3 个 token (~10ms)
  
大模型验证:
  一次 forward 验证 3 个候选 (~150ms)
  
  总耗时: ~160ms 出 2-3 个 token
  等效速度: 2-3 × (1000/160) ≈ 15-19 tok/s

叠加其他优化: → 50 tok/s
```

---

## 11.8 GPU 加速：充分利用 M3 Pro 的 14 核 GPU

### 为什么 M3 Pro 的 GPU 特别适合

```
传统 PC (CPU + NVIDIA GPU):
  权重在 CPU 内存 → 拷贝到 GPU 显存 → GPU 计算 → 结果拷贝回 CPU
  拷贝开销大: PCIe 带宽只有 ~32 GB/s
  这个拷贝是最大的性能杀手

M3 Pro (统一内存架构):
  权重在 18GB 统一内存 → GPU 直接访问 → GPU 计算 → CPU 直接读结果
  零拷贝! CPU 和 GPU 看同一块内存
  带宽: ~150 GB/s (比 PCIe 快 5×)
```

### GPU 能提供多少算力

```
M3 Pro GPU 规格:
  14 核 GPU
  Metal 4 支持
  fp32: ~5 TFLOPS
  fp16/bf16: ~15 TFLOPS
  int4/int8: ~30 TOPS (估算)

对比 CPU:
  11 核 CPU fp32: ~100 GFLOPS = 0.1 TFLOPS
  GPU 比 CPU 快: 50-300× (取决于精度)

关键: 13B 激活参数的计算量 = 26 GFLOP/token
  CPU: 26G / 0.1T = 260ms → 4 tok/s
  GPU (fp16): 26G / 15T = 1.7ms → 588 tok/s (理论)
  GPU (int4): 26G / 30T = 0.87ms → 1149 tok/s (理论)

结论: GPU 计算能力完全不是瓶颈!
       真正瓶颈是内存带宽 (150 GB/s)
```

### GPU + CPU 异步流水线 (核心优化)

```
不只是"GPU 替代 CPU 算 matmul", 而是 GPU 和 CPU 各干各的:

时间线:
  CPU: [从SSD预取专家] [路由计算] [从SSD预取] [路由] ...
  GPU: [MLA Attention]  [MoE 专家计算] [MLA]   [MoE]  ...

CPU 负责: I/O (mmap/madvise) + 路由 (轻量计算) + KV Cache 管理
GPU 负责: MLA Attention + MoE 专家的矩阵乘法 (重量计算)

两者并行: CPU 预取下一层专家的同时, GPU 算当前层
→ I/O 延迟完全隐藏在 GPU 计算后面
→ 理论上速度只受 GPU 计算限制, 不受 SSD 限制
```

### Metal 实现方案

```
方案 A: Metal Compute Shaders (最佳性能, 最复杂)
  - 用 Metal Shading Language 写 matmul kernel
  - 数据在统一内存, GPU 直接访问 (零拷贝)
  - 需要写 .metal 文件 + Objective-C 桥接
  - 代码量: ~300 行 Metal shader + ~100 行桥接
  - 预期速度: 30-50 tok/s

方案 B: Accelerate 框架 (中等性能, 较简单)
  - macOS 自带的 vDSP 库, 底层用 SIMD/NEON
  - 纯 C 调用, 不需要 Metal 知识
  - 代码量: ~50 行 (替换 matmul)
  - 预期速度: 15-25 tok/s
  - 注: 严格说这不是 GPU, 是 CPU SIMD, 但 Apple 优化过

方案 C: OpenMP 多线程 (最简单, 性能一般)
  - 5 个 P 核并行 matmul
  - 代码量: 3 行 (#pragma omp parallel for)
  - 预期速度: 10-15 tok/s
  - 不用 GPU
```

### 方案 A 的 Metal shader 示意

```metal
// matmul.metal — GPU 矩阵乘法 kernel
#include <metal_stdlib>
using namespace metal;

kernel void matmul_int4(
    device const uint8_t *weights [[buffer(0)]],  // int4 打包权重
    device const float   *x       [[buffer(1)]],  // 输入向量
    device float         *out     [[buffer(2)]],  // 输出
    constant int         &n       [[buffer(3)]],  // 输入维度
    constant int         &d       [[buffer(4)]],  // 输出维度
    uint tid [[thread_position_in_grid]]
) {
    // 每个 GPU 线程算输出的一行
    if (tid >= d) return;
    float val = 0;
    for (int j = 0; j < n; j++) {
        // int4 解码: 两个权重打包在一个字节里
        uint8_t packed = weights[tid * n / 2 + j / 2];
        float w = (j % 2 == 0) ? (packed >> 4) - 8 : (packed & 0xF) - 8;
        val += w * x[j];
    }
    out[tid] = val;
}
```

```c
// C 侧调用 (Objective-C++ 桥接)
// 关键: 数据在统一内存, 不需要拷贝!
id<MTLBuffer> weightBuf = [device newBufferWithBytesNoCopy:
    weights_ptr length:size options:MTLResourceStorageModeShared];
// weights_ptr 就是 mmap 的地址, GPU 直接访问, 零拷贝
```

### 为什么统一内存是杀手锏

```
其他 GPU (NVIDIA/AMD):
  1. 权重从 SSD 加载到 CPU 内存 (mmap)
  2. 从 CPU 内存拷贝到 GPU 显存 (PCIe, 慢!)
  3. GPU 计算
  4. 结果拷贝回 CPU 内存 (PCIe, 慢!)
  → 拷贝开销可能比计算还大

M3 Pro 统一内存:
  1. 权重从 SSD 加载到统一内存 (mmap)
  2. GPU 直接访问 (零拷贝!)
  3. GPU 计算
  4. CPU 直接读结果 (零拷贝!)
  → 没有拷贝开销, GPU 和 CPU 无缝协作

这就是为什么 M3 Pro 18GB 能跑 284B 模型:
  mmap 142GB → 热页在统一内存 → GPU 直接算 → 不用拷贝
  换成 PC: mmap 142GB → 拷贝到 GPU (PCIe 慢) → 算 → 拷贝回 (PCIe 慢)
  PC 的拷贝开销会让一切慢 5-10×
```

### 实际 GPU 利用策略

```
DeepSeek V4 Flash 一层的前向传播:

  ┌─ CPU 做 (轻量) ────────────────────────┐
  │ 1. RMSNorm (维度小, CPU 够快)          │
  │ 2. Router 计算 top-6 (只算 256 个分数) │
  │ 3. madvise 预取下一层专家              │
  │ 4. KV Cache 管理 (热冷分离)            │
  └───────────────────────────────────────┘
  ┌─ GPU 做 (重量) ────────────────────────┐
  │ 1. MLA Attention 的矩阵乘法            │
  │    (W_DKV 压缩 + W_UK/W_UV 解压)      │
  │ 2. 6 个专家的 SwiGLU 计算              │
  │    (6 × gate/up/down = 18 个 matmul)  │
  │ 3. 共享专家的计算                      │
  └───────────────────────────────────────┘

CPU 和 GPU 重叠执行:
  时间 →
  CPU: [Norm+Route] [预取] [Norm+Route] [预取]
  GPU: [Attention]  [Experts]  [Attention]  [Experts]
       ↑ 当前层              ↑ 下一层 (专家已预取到位)
```

---

## 11.9 内存预算分析

### 128K 上下文 (编码辅助典型场景)

```
项目                         内存
──────────────────────────────────
DeepSeek 活跃权重 (int4)      6.5 GB
KV Cache 热区 (8K, MLA int4) 0.2 GB
KV Cache 温区 (120K, int4)   1.5 GB
专家缓存 (8 个热门)           3.0 GB
草稿模块 (DSpark)             0.5 GB
OS + 程序                    2.0 GB
──────────────────────────────────
总计                         ~13.7 GB
M3 Pro 内存                   18 GB
剩余                          ~4.3 GB ✓
```

### 1M 上下文 (极限场景)

```
项目                         RAM        SSD
──────────────────────────────────────────────
权重 (mmap, int4)            ~10 GB     142 GB
KV Cache 热区 (8K)           0.4 GB     —
KV Cache 温区 (120K)         1.5 GB     —
KV Cache 冷区 (896K)         ~0         11 GB
专家缓存                     3.0 GB     —
OS + 程序                    2.0 GB     —
──────────────────────────────────────────────
总计 RAM                     ~17 GB     153 GB SSD
M3 Pro                       18 GB      ✓
```

---

## 11.10 实现路线图

### 阶段 1: 架构研究 + Config
- 确认 DeepSeek V4 Flash 的精确 config 参数
- 确认 MLA 矩阵维度
- 确认 MoE 专家结构

### 阶段 2: MLA Attention (~200 行)
- 低秩压缩 (W_DKV)
- 解压 (W_UK, W_UV)
- 解耦 RoPE (W_KR, W_QR)
- 压缩 KV Cache (只存 latent)

### 阶段 3: MoE MLP (~150 行)
- Router (top-6 选择)
- Expert dispatch
- Shared expert (常驻)
- mmap 专家加载

### 阶段 4: int4 量化 (~100 行)
- 权重 int4 存储
- matmul int4 解码 + 计算

### 阶段 5: mmap + 性能优化 (~200 行)
- 大文件 mmap
- madvise 预取
- OpenMP 多线程
- Metal GPU (后续)

### 阶段 6: DSpark 投机解码 (~200 行)
- MTP 草稿模块
- 验证逻辑

总计: ~850 行新代码 + 现有 3300 行 = ~4150 行

---

## 参考资料

- [DeepSeek V4 Flash - HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- [Unsloth DeepSeek V4 指南](https://unsloth.ai/docs/zh/mo-xing/deepseek-v4)
- [DSpark 论文](https://arxiv.org/html/2607.05147v1)
- [MLA 原理详解](https://planetbanatt.net/articles/mla.html)
- [MLA 实现详解](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html)
- [DeepSeek V4 in vLLM](https://vllm.ai/blog/2026-04-24)
- [llama.cpp DeepSeek V4 支持](https://github.com/ggml-org/llama.cpp/issues/22319)
