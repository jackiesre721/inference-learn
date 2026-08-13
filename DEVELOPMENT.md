# DeepSeek V4 Flash 开发计划

> 分支: `deepseek-support`
> 目标: 在 18GB M3 Pro 上运行 284B MoE 模型, 达到 50 tok/s
> 预估: ~1300 行新代码, ~325K tokens, 7 个阶段

---

## 总览

```
阶段         内容              代码量    tokens    速度       依赖
────────────────────────────────────────────────────────────────
1. 研究      架构+Config       ~50行    ~20K      —          无
2. MLA       Attention 重写    ~200行   ~60K      0.5 tok/s  阶段1
3. MoE       专家路由+dispatch ~150行   ~45K      2 tok/s    阶段1-2
4. int4      量化              ~100行   ~30K      5 tok/s    阶段2-3
5. GPU       Metal+流水线      ~400行   ~80K      15 tok/s   阶段2-4
6. DSpark    投机解码          ~200行   ~50K      28 tok/s   阶段2-5
7. 测试      验证+文档         ~200行   ~40K      —          全部
────────────────────────────────────────────────────────────────
总计                           ~1300行  ~325K     50 tok/s
```

---

## 阶段 1: 架构研究 + Config (~20K tokens)

### 任务列表

| # | 任务 | 产出 | 验证 |
|---|------|------|------|
| 1.1 | 下载 DeepSeek V4 Flash 的 config.json | config.json | 文件存在 |
| 1.2 | 分析 config, 提取所有架构参数 | 参数表 | 和文档一致 |
| 1.3 | 确认 MLA 维度 (d_c, d_r, n_heads, d_head) | 维度表 | 数学验证 |
| 1.4 | 确认 MoE 结构 (n_experts, top_k, expert_dim) | 结构表 | 和论文一致 |
| 1.5 | 创建 deepseek.h 配置结构体 | deepseek.h | 编译通过 |
| 1.6 | 下载 tokenizer (可能是 tiktoken 风格) | tokenizer 文件 | 能加载 |

### 关键参数待确认

```
需要从 config.json 确认:
  hidden_size (d_model)
  num_hidden_layers
  num_attention_heads
  kv_lora_rank (d_c, MLA 压缩维度)
  q_lora_rank (d_c for query)
  qk_rope_head_dim (d_r, RoPE 维度)
  n_routed_experts
  num_experts_per_tok (top_k)
  moe_intermediate_size (每个专家的 MLP 宽度)
  n_shared_experts
  vocab_size
  rope_theta
  first_k_dense_replace (前几层是 dense MLP)
```

### 完成标准
- [ ] config.json 已下载
- [ ] deepseek.h 创建并能编译
- [ ] 所有维度参数已确认

---

## 阶段 2: MLA Attention (~60K tokens)

### 任务列表

| # | 任务 | 产出 | 验证 |
|---|------|------|------|
| 2.1 | 加载 MLA 权重 (W_DKV, W_UK, W_UV, W_KR, W_QR, W_O) | safetensors 加载 | 张量数量正确 |
| 2.2 | 实现 KV 压缩: c_kv = x @ W_DKV | compress 函数 | 维度正确 |
| 2.3 | 实现 KV 解压: K = c_kv @ W_UK, V = c_kv @ W_UV | decompress 函数 | 维度正确 |
| 2.4 | 实现解耦 RoPE: k_r = x @ W_KR, q_r = c_q @ W_QR | rope_decoupled 函数 | 旋转角度正确 |
| 2.5 | 实现 MLA attention 分数: score = (K·Q + k_r·q_r) / √(d_h+d_r) | attention 计算 | 数值合理 |
| 2.6 | 实现压缩 KV Cache (只存 c_kv + k_r) | cache 结构 | 内存减少 |
| 2.7 | 完整 MLA forward 一层 | mla_forward() | 能跑通 |
| 2.8 | 数值验证: 单 token logits 和 PyTorch 对比 | 验证脚本 | 误差 < 0.01 |

### 代码结构

```c
// deepseek.h
typedef struct {
    int d_model;       // hidden_size
    int n_heads;
    int d_head;
    int kv_lora_rank;  // d_c: MLA 压缩维度
    int q_lora_rank;   // query 压缩维度
    int qk_rope_dim;   // d_r: RoPE 解耦维度
} DeepSeekConfig;

typedef struct {
    float *W_DKV;      // (d_c, d_model) 压缩
    float *W_UK;       // (n_heads*d_head, d_c) 解压 K
    float *W_UV;       // (n_heads*d_head, d_c) 解压 V
    float *W_KR;       // (d_r, d_model) RoPE key
    float *W_QR;       // (d_r, q_c) RoPE query
    float *W_O;        // (d_model, n_heads*d_head) 输出
} MLAWeights;

// 压缩 KV Cache: 只存 latent, 不存完整 K/V
typedef struct {
    float *kv_cache;   // (n_layers, seq_len, d_c) 压缩 latent
    float *kr_cache;   // (n_layers, seq_len, d_r) RoPE key
} MLACache;
```

### 完成标准
- [ ] MLA forward 能跑通 (不崩)
- [ ] 单 token 数值和 PyTorch 误差 < 0.01
- [ ] KV Cache 内存比标准 attention 少 50×

---

## 阶段 3: MoE MLP (~45K tokens)

### 任务列表

| # | 任务 | 产出 | 验证 |
|---|------|------|------|
| 3.1 | 加载 MoE 权重 (W_router, 256 experts, shared expert) | safetensors 加载 | 张量数量正确 |
| 3.2 | 实现 Router: logits = x @ W_router → top-6 选择 | moe_route() | 选出 6 个专家 |
| 3.3 | 实现 softmax 归一化专家权重 | 归一化 | 权重和 = 1 |
| 3.4 | 实现单个 expert forward (SwiGLU) | expert_forward() | 和 Qwen MLP 类似 |
| 3.5 | 实现 shared expert forward (常驻) | shared_forward() | 每个 token 都用 |
| 3.6 | 实现 expert dispatch: 加载 6 个专家 + 计算 + 加权求和 | moe_forward() | 能跑通 |
| 3.7 | mmap 专家加载: 256 个专家在磁盘, 按需访问 | mmap 设置 | 内存不超 |
| 3.8 | 完整 MoE forward 一层 | moe_layer() | 数值合理 |
| 3.9 | 数值验证: 和 PyTorch 对比 | 验证脚本 | 误差 < 0.01 |

### 代码结构

```c
typedef struct {
    float *W_router;           // (n_experts, d_model) 路由权重
    ExpertWeights *experts;     // [n_experts] 256 个专家
    ExpertWeights shared;       // 共享专家 (常驻)
    int n_experts;              // 256
    int top_k;                  // 6
} MoELayer;

typedef struct {
    float *w_gate;   // (expert_dim, d_model)
    float *w_up;     // (expert_dim, d_model)
    float *w_down;   // (d_model, expert_dim)
} ExpertWeights;

// MoE 前向
void moe_forward(MoELayer *moe, float *x, float *out,
                 int *selected_experts, float *expert_weights);
```

### 完成标准
- [ ] MoE forward 能跑通 (不崩)
- [ ] Router 能正确选出 top-6
- [ ] 专家 dispatch 逻辑正确
- [ ] mmap 专家加载, 内存 < 18GB
- [ ] 数值和 PyTorch 误差 < 0.01

---

## 阶段 4: int4 量化 (~30K tokens)

### 任务列表

| # | 任务 | 产出 | 验证 |
|---|------|------|------|
| 4.1 | 编写 Python 转换脚本: bf16 → int4 GGUF | convert_int4.py | 文件生成 |
| 4.2 | 实现 int4 权重加载 (两个权重打包 1 字节) | int4 加载函数 | 解码正确 |
| 4.3 | 实现 matmul_int4: int4 权重 × fp16 输入 | matmul_int4() | 数值合理 |
| 4.4 | 精度验证: int4 vs fp32 输出对比 | 验证脚本 | 误差 < 0.1 |
| 4.5 | 性能对比: int4 vs fp32 速度 | 基准测试 | 快 2-3× |

### 完成标准
- [ ] int4 权重能正确加载和解码
- [ ] matmul_int4 数值误差 < 0.1
- [ ] 模型文件从 142GB(fp16) 缩到 ~71GB(int4)
- [ ] 速度比 fp32 快 2×

---

## 阶段 5: GPU 加速 — Metal (~80K tokens)

### 任务列表

| # | 任务 | 产出 | 验证 |
|---|------|------|------|
| 5.1 | 编写 Metal matmul shader (.metal 文件) | matmul.metal | 编译通过 |
| 5.2 | Obj-C++ 桥接: MTLDevice/Buffer/CommandQueue | metal_bridge.mm | 能调用 GPU |
| 5.3 | 统一内存零拷贝: newBufferWithBytesNoCopy | 零拷贝测试 | 无拷贝 |
| 5.4 | 替换 matmul: CPU 版 → GPU 版 | GPU matmul | 数值一致 |
| 5.5 | 实现 madvise 专家预取 | 预取逻辑 | 减少等待 |
| 5.6 | 实现 GPU/CPU 异步流水线 | 流水线 | I/O 隐藏 |
| 5.7 | OpenMP 作为 CPU 备选 (3 行 pragma) | omp 版本 | 快 5× |
| 5.8 | 性能基准测试 | 基准报告 | 15+ tok/s |

### 完成标准
- [ ] Metal shader 编译通过
- [ ] GPU matmul 和 CPU 结果一致
- [ ] 统一内存零拷贝验证
- [ ] GPU/CPU 流水线工作正常
- [ ] 速度达到 15+ tok/s

---

## 阶段 6: DSpark 投机解码 (~50K tokens)

### 任务列表

| # | 任务 | 产出 | 验证 |
|---|------|------|------|
| 6.1 | 加载 DSpark 草稿模块权重 | 草稿模型加载 | 能加载 |
| 6.2 | 实现草稿模块 forward (轻量 transformer) | draft_forward() | 能预测 3 个 token |
| 6.3 | 实现大模型批量验证 (一次 forward 验证 3 个) | verify() | 能验证 |
| 6.4 | 实现接受/拒绝逻辑 | accept/reject | 正确性验证 |
| 6.5 | 参数调优: --spec-draft-n-max | 参数 | 接受率 > 70% |
| 6.6 | 性能基准: 有/无 DSpark 对比 | 基准报告 | 1.85× 加速 |

### 完成标准
- [ ] 草稿模块能预测 3 个候选 token
- [ ] 大模型能批量验证
- [ ] 接受率 > 70%
- [ ] 整体加速 1.5-1.85×

---

## 阶段 7: 调试 + 测试 + 文档 (~40K tokens)

### 任务列表

| # | 任务 | 产出 | 验证 |
|---|------|------|------|
| 7.1 | 完整 forward 数值验证 (MLA+MoE+int4) | 验证脚本 | 误差 < 0.1 |
| 7.2 | 端到端文本生成测试 | 生成示例 | 输出连贯 |
| 7.3 | 编码辅助场景测试 | 测试报告 | 能写代码 |
| 7.4 | HTTP 服务器 + DeepSeek | server 测试 | API 可用 |
| 7.5 | 性能基准 (各阶段速度) | 基准报告 | 50 tok/s |
| 7.6 | 更新第 11 章文档 | 完整文档 | 图文并茂 |
| 7.7 | 创建 PR 合并到 main | PR | CI 通过 |

### 完成标准
- [ ] 端到端能生成连贯文本
- [ ] HTTP API 可用 (OpenAI 兼容)
- [ ] 性能达到目标
- [ ] 文档完整
- [ ] PR 创建并合并

---

## 开发节奏建议

### 每个 session 的结构

```
1. 回顾: 看上一个 session 做到哪了
2. 目标: 这个 session 做哪个任务的哪几个子任务
3. 实现: 写代码
4. 验证: 编译 + 测试
5. 提交: git commit + push
6. 记录: 更新本文档的进度
```

### 进度追踪

在每次 session 开始时, 把对应阶段的所有 `- [ ]` 更新为实际状态。

---

## 依赖关系图

```
阶段1 (研究+Config)
  ├──→ 阶段2 (MLA Attention)
  │      ├──→ 阶段3 (MoE MLP)           ← 阶段2-3 完成后可端到端跑
  │      │      ├──→ 阶段4 (int4 量化)
  │      │      │      ├──→ 阶段5 (GPU Metal)
  │      │      │      │      ├──→ 阶段6 (DSpark)
  │      │      │      │      │      ├──→ 阶段7 (测试+文档+PR)
  │      │      │      │      │      │
  │      │      │      │      │      └─ 可选: 合并 main
  │      │      │      │      │
  │      │      │      │      └─ 阶段6 也可以不用 GPU 直接做
  │      │      │      │
  │      │      │      └─ 阶段5 需要先有 int4 权重格式
  │      │      │
  │      │      └─ 阶段4 需要正确的 forward 才能验证
  │      │
  │      └─ 阶段3 需要 MLA 的输出作为 MoE 的输入
  │
  └─ 阶段1 是一切的基础
```

### 可以并行的任务

```
阶段4 (int4) 和 阶段5 的 OpenMP 部分 可以并行
阶段6 (DSpark) 的草稿模块可以提前准备
```

---

## 新增文件清单

```
deepseek-support 分支新增:
├── deepseek.h           — DeepSeek 配置 + MLA/MoE 数据结构
├── deepseek.c           — MLA attention + MoE MLP + forward 实现
├── matmul.metal         — Metal GPU int4 matmul shader
├── metal_bridge.mm      — Obj-C++ Metal 桥接层
├── convert_int4.py      — bf16 → int4 量化转换脚本
├── draft.h              — DSpark 草稿模块
├── draft.c              — 草稿模块实现
├── DEVELOPMENT.md       — 本文件 (开发计划)
└── docs/11-deepseek.md  — 技术方案文档 (已完成)
```

## 风险和备选

| 风险 | 影响 | 备选方案 |
|------|------|---------|
| 网络下载 142GB 模型太慢 | 阶段1 卡住 | 用 hf-mirror, 或先用小模型调试 |
| MLA 数值对不上 | 阶段2 卡住 | 用 PyTorch 逐层 dump 对比 |
| Metal shader 编译问题 | 阶段5 卡住 | 先用 OpenMP, Metal 后做 |
| DSpark 接受率低 | 阶段6 无效 | 调参, 或跳过用其他优化 |
| 内存不够 18GB | 全局 | 减小上下文, 增加专家换出 |

---

## 当前进度

```
阶段1: [ ] 未开始
阶段2: [ ] 未开始
阶段3: [ ] 未开始
阶段4: [ ] 未开始
阶段5: [ ] 未开始
阶段6: [ ] 未开始
阶段7: [ ] 未开始
```

文档: docs/11-deepseek.md [x] 已完成 (648 行)
