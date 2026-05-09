---
title: "Prefill/Decode 拆机：两个阶段坐在 roofline 的两边"
date: 2026-05-09T00:00:00+00:00
draft: false
summary: "Article 05 让两个阶段勉强共用一台引擎。这一篇要说的是：它俩本来就不该共用 —— prefill 是 compute-bound、decode 是 bandwidth-bound，长上下文还把这条沟越拉越宽。承认了这种 asymmetry，拆机就不再是优化，而是顺着公式来的唯一诚实答案。"
description: "用 roofline 的视角论证 prefill/decode 拆机。先把 arithmetic intensity 讲清楚，推出 transformer 一次 iteration 的强度 ≈ 一次 weight 加载被几个 token 共享，再扫一遍 context length 看 decode 怎么进一步往 ridge 之下掉，最后走一遍拆机后的形态和它带来的 KV 传输成本。"
tags: ["pd-disaggregation", "prefill", "decode", "roofline", "arithmetic-intensity", "kv-cache", "llm-serving", "mental-model"]
series: ["llm-stories"]
showToc: true
TocOpen: false
weight: 7
---

[Article 05](/llm_stories/posts/05-orca-and-chunked-prefill/) 收尾的时候引擎跳得很稳。ORCA 把"进出 batch"的边界问题修了；chunked prefill 给每次 iteration 的开销封了顶，长 prompt 没法一个人霸占整间屋子。每次 iteration 都有上限、每个 request 都大致公平，引擎呼吸均匀。

但那一篇文章末尾留了一根线，§7 第二条：

> *Decode 卡在 weight 读带宽上；prefill chunk 又是 compute-bound。也许它俩根本就不该共用同一组 GPU。*

这一篇就是顺着这根线往下扯。我们用 roofline 的角度量一下这条沟到底有多宽，看着它在 context length 增长时越拉越开，最后落到结构性的修法：让两个阶段彻底不共用一台机器。

起点其实有点尴尬：article 05 那套 piggyback chunked prefill 不是 prefill/decode 不匹配的*答案*，是个*妥协*。它把心跳抚平了，但底下的事实是 —— 一块 prefill chunk 和一个 decode token 想要 GPU 处于完全不同的 regime 里。共用同一次 forward，只不过逼着双方都退到对自己不合适的那一种。

---

## 1. Roofline，一页讲完

每张 GPU 上的每个 kernel，瓶颈都只在两件物理资源之一：

- **算力（compute）** —— tensor core 的峰值 FLOPs/s。
- **内存带宽（memory bandwidth）** —— HBM 把 bytes 送到 SM 的速率。

（这里讲的是 GPU *内部*的带宽，是 HBM 到 tensor core 之间那条管道。GPU 之间的带宽 —— NVLink、InfiniBand —— 是另一条轴线，等 TP/PP 出场我们再讲。）

### 字节实际放在哪：一张图

光说"内存带宽"很抽象。现代 GPU 有一套 **memory hierarchy** —— 几层缓存，越往上越小、越快。Tensor core 只能对 register 里的数据做运算，所以每一个 weight 字节、每一个 KV cache 字节，在被算到之前都得沿着这条 hierarchy 一路爬上来。

<svg viewBox="0 0 720 540" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;font-family:system-ui,sans-serif;display:block;margin:1.5rem auto">
  <text x="360" y="24" text-anchor="middle" font-size="14" fill="currentColor" font-weight="600">GPU memory hierarchy（H100 风格的数字）</text>
  <rect x="40" y="50" width="640" height="290" fill="none" stroke="currentColor" stroke-width="1.5" stroke-opacity="0.6" rx="6"/>
  <text x="60" y="74" font-size="13" fill="currentColor" opacity="0.75" font-weight="600">GPU die</text>
  <g>
    <rect x="70" y="95" width="150" height="120" fill="rgba(74,144,226,0.12)" stroke="#4a90e2" stroke-width="1.5" rx="4"/>
    <text x="145" y="115" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">SM 0</text>
    <text x="145" y="138" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.85">registers</text>
    <text x="145" y="158" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.85">SRAM (~256 KB)</text>
    <text x="145" y="178" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.85">tensor cores</text>
    <text x="145" y="202" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.6" font-style="italic">~30 TB/s 有效</text>
  </g>
  <g>
    <rect x="285" y="95" width="150" height="120" fill="rgba(74,144,226,0.12)" stroke="#4a90e2" stroke-width="1.5" rx="4"/>
    <text x="360" y="115" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">SM 1</text>
    <text x="360" y="138" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.85">registers</text>
    <text x="360" y="158" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.85">SRAM (~256 KB)</text>
    <text x="360" y="178" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.85">tensor cores</text>
  </g>
  <g>
    <rect x="500" y="95" width="150" height="120" fill="rgba(74,144,226,0.12)" stroke="#4a90e2" stroke-width="1.5" rx="4"/>
    <text x="575" y="115" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">SM 131</text>
    <text x="575" y="138" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.85">registers</text>
    <text x="575" y="158" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.85">SRAM (~256 KB)</text>
    <text x="575" y="178" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.85">tensor cores</text>
  </g>
  <text x="247" y="158" text-anchor="middle" font-size="18" fill="currentColor" opacity="0.55">⋯</text>
  <text x="467" y="158" text-anchor="middle" font-size="18" fill="currentColor" opacity="0.55">⋯</text>
  <line x1="145" y1="215" x2="360" y2="245" stroke="currentColor" stroke-opacity="0.5" stroke-width="1"/>
  <line x1="360" y1="215" x2="360" y2="245" stroke="currentColor" stroke-opacity="0.5" stroke-width="1"/>
  <line x1="575" y1="215" x2="360" y2="245" stroke="currentColor" stroke-opacity="0.5" stroke-width="1"/>
  <rect x="220" y="245" width="280" height="70" fill="rgba(245,166,35,0.18)" stroke="#f5a623" stroke-width="1.5" rx="4"/>
  <text x="360" y="268" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">L2 cache</text>
  <text x="360" y="287" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.85">~50 MB 共享</text>
  <text x="360" y="304" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.6" font-style="italic">~5 TB/s</text>
  <line x1="360" y1="340" x2="360" y2="395" stroke="currentColor" stroke-width="2"/>
  <polygon points="355,395 365,395 360,408" fill="currentColor"/>
  <text x="375" y="375" font-size="12" fill="currentColor" font-weight="600">3.35 TB/s</text>
  <text x="375" y="392" font-size="10" fill="currentColor" opacity="0.65" font-style="italic">HBM 带宽</text>
  <rect x="120" y="415" width="480" height="100" fill="rgba(126,211,33,0.15)" stroke="#7ed321" stroke-width="1.5" rx="4"/>
  <text x="360" y="440" text-anchor="middle" font-size="13" fill="currentColor" font-weight="600">HBM — 80 GB</text>
  <text x="360" y="463" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.85">模型权重 · KV cache · 跨 kernel 的 activation</text>
  <text x="360" y="490" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.6" font-style="italic">够装下模型和这一批的状态，但层级里最慢的一档</text>
</svg>

数字是 H100 风格的；其他 GPU 绝对值不一样，但*形状* —— 顶层和底层在容量和速度上差三到四个数量级 —— 是普遍的。

什么东西放在哪：

- **HBM** 装常驻的东西：模型权重（Llama-2-7B 是 14 GB）、每个 request 的 KV cache、跨 kernel 留存的 activation。容量大，相对慢。
- **L2 cache** 是几个 SM 共享的一块小 scratch —— 多个 SM 都在读重叠数据时有用，但只有 ~50 MB，远装不下权重或 KV。
- **SRAM（per-SM shared memory）** 是 kernel 当下正在动的那一块 weight、Q、K 的暂存区。FlashAttention 那套花招的核心就是把 attention score 矩阵压在 SRAM 里，不让它溢到 HBM。
- **Register** 是 tensor core 真正读 operand 的地方。每个 SM 几百 KB，访问只要一个 cycle。

所以当你看到"kernel 从 HBM 加载了 14 GB 权重"的时候，路径是 HBM → L2 → SRAM → register → tensor core。一层比一层小、一层比一层快。**3.35 TB/s 是这条链子最*底*的那一档** —— 也是一次 transformer iteration 没法绕过的那个瓶颈，因为权重比 HBM 之上每一层都大。

### compute-bound 和 bandwidth-bound 在物理上到底是什么

矩阵乘按 **tile** 工作：从 HBM 把一块 A、一块 B 装到 SRAM 里，在 register 里相乘（每个元素背后是很多 FLOPs）、累加，下一块。同一块 weight tile 在被换出之前，会被很多次输出行复用。

- **compute-bound** 是 tensor core 跑满的状态。当前 tile 消耗得足够快，HBM 把下一块送来都来得及。带宽有富余。*每个 weight 字节加载一次，被很多次 FLOPs 复用。*
- **bandwidth-bound** 是 HBM 送不上下一块。tensor core 已经把当前的吃完了，干等着字节到。*每个字节被复用的 FLOPs 太少，分摊不下加载这一次的成本。*

判断你在哪种 regime 里的那个数，恰好就是 **每从 HBM 拉出一个 byte，做了多少 FLOPs** —— 这就是 intensity。也是为什么 roofline 这条规则没什么回旋余地：它不是经验观察，是上面这套 hierarchy 的直接推论。

### Roofline 这条规则

哪种资源是瓶颈，由一个数决定：**arithmetic intensity** `I`，FLOPs 数和从 HBM 加载的 bytes 数的比值：

```
I  =  FLOPs done / bytes loaded     （单位：FLOPs/byte）
```

硬件这边对应有一个数，叫 **ridge point** `R`：

```
R  =  peak FLOPs/s / peak HBM bandwidth   （单位：FLOPs/byte）
```

H100 SXM5：fp16 GEMM 持续算力 ~500 TFLOPs/s，HBM3 带宽 3.35 TB/s → **R ≈ 150 FLOPs/byte**。

规则：

- `I > R` → **compute-bound**。算力是瓶颈；带宽有富余。
- `I < R` → **bandwidth-bound**。字节是瓶颈；tensor core 在等数据。

就这一条。剩下整篇文章其实就是在反复问两个问题：

1. 一次 prefill iteration 和一次 decode iteration 的 `I` 各是多少？
2. context length 增长时 `I` 怎么变？

---

## 2. 记号和单次 iteration 的开销模型

先把符号定下来。**全文假设 fp16**（每个参数 2 byte、cache 里每个数也 2 byte）。换更低精度的 dtype 会让数字变，但故事不变。

| 符号 | 含义 | 单位 |
|---|---|---|
| `Π` | 参数总数 | 无量纲 |
| `K_tok` | 一个 context token 在 KV cache 里的字节数（所有层的 K + V 加起来） | bytes/token |
| `T` | 这次 iteration 里的 token 总数 | tokens |
| `B` | 这次 iteration 里在跑的 request 数 | 无量纲 |
| `L` | 每个 request 的平均 context length | tokens |
| `C` | prefill chunk 大小（每个 chunk 的新 token 数） | tokens |
| `R` | 硬件 ridge point | FLOPs/byte |

（`K_tok` 是把所有层加起来的总和 —— 一个 context token *在整张网络的 KV cache* 里要占多少字节，不是每层。）

先点一下：transformer block 里大部分算力和几乎全部参数都在它的 **matmul 层** —— QKV projection、attention 的 output projection、FFN 的 up/down projection。Attention 本身（softmax over scores 那一步）和 pointwise 操作（layernorm、GeLU、residual add）只占总 FLOPs 的一小块（除非上下文极长）。所以下面说"每个 token 多少 FLOPs"或者"weight bytes"的时候，意思都是 matmul —— 那才是开销所在。

对一次走在大小为 `Π` 的模型上的 iteration，有两个物理量要追，两个都**关于 `Π` 线性**：

- **从 HBM 拉出来的 weight bytes：** 每个参数 fp16 = 2 byte，一次 iteration 读一遍 → `2Π` bytes。Llama-2-7B（`Π = 7B`）就是 14 GB。一次 iteration 付一次，跟塞了多少 token 没关系。
- **每个 token 走完整张网络做的 FLOPs：** 一个 token 走过一层 matmul，和那层每个参数都做一次 multiply-accumulate（每个参数 2 FLOPs）。把整张网络的 matmul 加起来，每个 token `2Π` FLOPs —— Llama-2-7B 就是 14 GFLOPs/token。一次 iteration 处理 `T` 个 token，做 `2Π · T` FLOPs。token 在 matmul 里互不干扰（只在 attention 里互相看到），所以同一次 iteration 里两个 token 的算力是单个 token 的两倍 —— 但 weight 加载只付一次。

加上 KV cache 读取，把两件事一起写下来：

```
bytes_loaded  =  2Π                  (weights, 一次 iteration 付一次)
              +  K_tok · L · B       (KV cache, 每个 request 读自己那 L 行)

FLOPs_done    =  2Π · T              (T = 这次 iteration 里的 token 数)
```

塞进 intensity 的定义里：

```
I  =  2Π · T  /  (2Π + K_tok · L · B)
```

盯着这条公式看一会儿 —— 这一节余下的内容就是把它读仔细。分母两项、分子一项；按顺序走一遍，整个 prefill/decode 故事就出来了。

### 第一步：先假装 KV 这一项是零

在 `L` 极短、或者一段对话刚开始还没什么 context 的时候，分母由 `2Π` 主导，公式塌成：

```
I ≈ T
```

intensity 就是 **共享同一次 weight 加载的 token 数**。prefill 和 decode 在这里就分了道：

- **Prefill iteration**：`T = C = 2048` 个 token → `I ≈ 2000` → 远高于现代 ridge point（~150） → **compute-bound**。
- **Decode iteration**：`T = B`（同时在 decode 的 request 数，一般几十到一百多）→ `I ≈ B` → 远低于 ridge → **bandwidth-bound**。

同一张 GPU、同一个模型、同一个 kernel。唯一区别是这次 iteration 装了多少 token。Prefill 把一次 weight 加载分摊到几千个 token 上；decode 分摊到 `B` 个上。从第一次 iteration 起，他们就坐在 ridge 的两边 —— 而且差距不小：intensity 上至少差一个数量级。

直觉的修法是把 decode 的 batch 推得更大 —— 把 `B` 推到 intensity 越过 ridge 为止。要清掉 `R = 150`，得 `B ≥ 150`。下一步说为什么这条路走不通。

### 第二步：把 KV 那一项打开

context 一长，`K_tok · L · B` 就开始往分母里加。两项相等的位置（**crossover**）：

```
L · B  =  2Π / K_tok
```

Llama-2-7B（`Π = 7B`、`K_tok ≈ 512 KB`）下，`L · B ≈ 27 k`。decode batch `B = 32` 的话，crossover 落在 `L ≈ 850` token。

850 这个数，放到今天的标准里小到吓人，值得停一下。生产环境里的 prompt 现在动不动就是**几万** token：超长的 system prompt 和工具定义、RAG 灌进来的文档、累积的多轮对话、agentic chain 那种 input/output ratio 经常 100:1 起步的工作流。前沿模型出 200 k – 2 M 的 context window，是因为真实的 workload 真的会塞满。所以"过了 crossover"根本不是 corner case，而是**中位 request**。

过了 crossover，公式向另一个方向化简：

```
I ≈ 2Π · T / (K_tok · L · B)
```

这里的约分关系开始决定命运：

- **Decode**（`T = B`）：`I ≈ 2Π / (K_tok · L)`。`B` 上下消掉 —— *在长 context 下，把 decode 的 batch 推大不再能提升 intensity。*多收的 request 只是按比例多付 KV 读带宽。再加上 KV 内存预算，`B` 还涨不太大就先把卡撑爆。所以"把 batch 推大"这一招在第一步行不通的同时，到了第二步还会再行不通一次。
- **Prefill**（`T = C`）：`I ≈ 2Π · C / (K_tok · L · B)`。没东西约掉 —— `C` 老老实实留在分子里。Prefill 一直 compute-bound，到夸张的 context 长度都还守得住。

### 同一条公式，两件事

1. **Prefill 是 compute-bound、decode 是 bandwidth-bound。** 在 context 接近零的时候就成立，完全由"同一次 weight 加载分摊到几个 token"决定。两个阶段从一开始就坐在 ridge 的两边。
2. **长 context 把这条沟拉得更宽。** 第二项带宽成本（KV 读）从分母里冒出来，过了 crossover 就主导（生产流量基本都过 crossover）。它打的主要是 decode，prefill 几乎没事。

§3 用 Llama-2-7B 上的具体数字把这两件事坐实。

---

## 3. 一个模型、两个阶段、两张表

把公式落到地面上：在一个具体模型 + 一张具体 GPU 上，扫一遍 `L`。

**Llama-2-7B（MHA、32 层、32 head、head_dim 128、fp16）on H100：**

- weight bytes `2Π = 14 GB`
- `K_tok = 2 (K,V) · 32 层 · 32 head · 128 head_dim · 2 byte ≈ 512 KB/token`
- ridge `R ≈ 150 FLOPs/byte`

### Decode at B = 32

| `L` | weight bytes | KV bytes | total | `I = 2Π·B / total` | regime |
|---:|---:|---:|---:|---:|---|
| 1 k | 14 GB | 16 GB | 30 GB | ~15 | bandwidth-bound（weights ≈ KV） |
| 4 k | 14 GB | 64 GB | 78 GB | ~5.7 | bandwidth-bound（KV 主导） |
| 16 k | 14 GB | 256 GB | 270 GB | ~1.7 | 严重 bandwidth-bound |
| 64 k | 14 GB | 1.0 TB | 1.0 TB | ~0.4 | cache 在一张 H100 上*装不下* |

（分子 `2Π · B = 448 GFLOPs` —— 钉死的。是分母在炸。）

注意几件事：

- **Intensity 跌得很快。** 从 L=1k 的 ~15 跌到 L=64k 的 ~0.4 —— 单一个 context 维度上就掉了一个数量级以上。
- **内存预算先于带宽爆。** L=16k、B=32 时单 KV 就 256 GB，远超 H100 的 80 GB。PagedAttention 的存在一部分就是为了管这件事；`B` 在长 context 下被迫*往下压*，结果 intensity 又被进一步拖坏。（Llama-2-7B 用的是 MHA；现代 GQA/MLA 把 `K_tok` 砍掉 4–8 倍，主要就是为了把这堵墙往后推。）
- **主导的字节种类会变。** 短 L 时 weight 主导，长 L 时 KV 主导。两边都是 bandwidth-bound，但解法不同 —— batch 推大对 weight 带宽有用；GQA/MLA/FlashDecoding 对 KV 带宽有用。

### Prefill at C = 2048

Chunked prefill 拿 `C` 个新 token 跑，面对一段长度 `S` 的前缀（所以 `T = C` 个 token 的算力，读 `S` 个 token 的 cache KV）：

```
I_prefill = 2Π · C / (2Π + K_tok · S)
```

分子按 `C` 走 —— 每个加载的字节都被分摊到几千个 token 的算力上。

| 前缀 `S` | weight bytes | KV bytes | total | `I` | regime |
|---:|---:|---:|---:|---:|---|
| 4 k | 14 GB | 2 GB | 16 GB | ~1800 | compute-bound（高出 ridge ×12） |
| 64 k | 14 GB | 32 GB | 46 GB | ~620 | compute-bound（×4） |
| 256 k | 14 GB | 128 GB | 142 GB | ~200 | 还是 compute-bound（×1.3） |
| 1 M | 14 GB | 512 GB | 526 GB | ~55 | 终于跌到 ridge 之下 —— 但已经 100 万 token 了 |

Prefill 一直 compute-bound 到极端的 context 长度。哪怕真的跌到 ridge 下面，也远远没到 decode 在*常见* context 长度下那种 bandwidth-bound 的程度。

asymmetry，干净地说：

> **Prefill 把每一字节带宽分摊到 `C ≈ 2000` 个 token 上；decode 是每个 request 一个 token。长 context 把刀子往 decode 这边拧，prefill 几乎没动。**

同一个模型，同一张 GPU。两个阶段。两条完全不同的命运曲线。

---

## 4. 为什么一台引擎没法把两边都伺候好

把 article 05 那台引擎 —— continuous batching、chunked prefill、piggyback iteration —— 拿过来问一句：你怎么 sizing？

- **按 prefill sizing：** 给 GPU 选高 FLOPs 的型号。Decode 跑在一种 ~90% 算力天生用不上的硬件上，因为它是 bandwidth-bound。你为 decode 物理上用不到的 tensor core 付钱。
- **按 decode sizing：** 选少一点、按 HBM 带宽和容量来挑的 GPU。Prefill 跑在缺 FLOPs 的机器上，时间被拖长。**TTFT** 上去。
- **混跑：** 每次 iteration 把 prefill chunk 和 decode token 装在一起。**TBT** 被这次 iteration 里 prefill chunk 吃掉的那点算力扣作人质。Chunked prefill 给这条卡了上限 —— 那就是 article 05 的全部目的 —— 但这个上限不是免费的。共用一台引擎的一次 decode iteration，要为 `C` 行根本对自己没用的 prefill 算力买单。

更深一层：**workload 的瓶颈 profile 是双峰的，引擎是单峰的。**没有一种 sizing、没有一种 parallelism 策略、没有一种 batch policy，能同时把两个阶段都伺候好。两个阶段拉满的是不同的物理资源、追的是不同的 SLO（TTFT vs TBT），一个 scheduler、一个旋钮，没法同时满足两个 SLO 在两种 regime 下。

那就别拼了。建两个 pool。

---

## 5. 拆开

<svg viewBox="0 0 760 260" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;font-family:system-ui,sans-serif;display:block;margin:1.5rem auto">
  <defs>
    <marker id="arr-split-zh" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="currentColor"/>
    </marker>
  </defs>
  <text x="40" y="120" text-anchor="end" font-size="12" fill="currentColor" opacity="0.85">prompt</text>
  <line x1="48" y1="115" x2="100" y2="115" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-split-zh)"/>
  <rect x="105" y="65" width="225" height="130" fill="rgba(74,144,226,0.15)" stroke="#4a90e2" stroke-width="1.5" rx="6"/>
  <text x="217" y="92" text-anchor="middle" font-size="14" fill="currentColor" font-weight="600">Prefill pool</text>
  <text x="217" y="120" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.85">compute-bound</text>
  <text x="217" y="140" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.85">目标是 TTFT</text>
  <text x="217" y="160" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.85">无状态</text>
  <line x1="332" y1="115" x2="428" y2="115" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-split-zh)"/>
  <text x="380" y="100" text-anchor="middle" font-size="11" fill="currentColor" font-weight="600">KV cache transfer</text>
  <text x="380" y="138" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.65" font-style="italic">每个 request,</text>
  <text x="380" y="153" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.65" font-style="italic">L_p · K_tok bytes</text>
  <rect x="430" y="65" width="225" height="130" fill="rgba(245,166,35,0.18)" stroke="#f5a623" stroke-width="1.5" rx="6"/>
  <text x="542" y="92" text-anchor="middle" font-size="14" fill="currentColor" font-weight="600">Decode pool</text>
  <text x="542" y="120" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.85">bandwidth-bound</text>
  <text x="542" y="140" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.85">目标是 TBT</text>
  <text x="542" y="160" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.85">持有长寿 KV</text>
  <line x1="657" y1="115" x2="710" y2="115" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-split-zh)"/>
  <text x="720" y="120" text-anchor="start" font-size="12" fill="currentColor" opacity="0.85">tokens</text>
</svg>

一个 request 的生命中间多了一跳：

1. **Prefill pool** 收下 prompt，对全部 `L_p` 个 token 跑 chunked prefill，产出整个 request 的 KV cache 加上第一个生成 token。
2. **KV cache 传输** 把这 `L_p · K_tok` byte 从 prefill GPU 内存搬到 decode GPU 内存。
3. **Decode pool** 接到 KV cache，把 request 塞进自己的 continuous-batching 池子，一直 decode 到 EOS，把 token 流式吐回给用户。

两个 pool、两套调度、两个 SLO 目标。妥协没了。每个 pool 现在可以针对一个*单一*目标自由地选 parallelism、batch policy、硬件搭配、scheduling 纪律。这份自由就是大头收益 —— 各自具体怎么用它，留给系列后面的文章。

新成本是中间这一跳。我们在 §6 给它定价。

---

## 6. 新成本：KV cache 传输

拆开两台引擎之后，每个 request 都要把 KV cache 从一边搬到另一边一次。这是真成本，先估个数。

Llama-2-7B（`K_tok ≈ 512 KB`）一段 4 k-token 的 prompt：

```
每个 request 的 KV bytes  =  L_p · K_tok  =  4096 · 512 KB  ≈  2 GB
```

每个 request 2 GB。每秒几百个 request（不算高的生产负载）的话，两个 pool 之间的*总* east-west 流量轻松能到几百 GB/s。中间那条 fabric 得吃得下这个量。

fabric 长什么样、一次传输要多久：

| Fabric | 带宽 | 传 2 GB 要多久 |
|---|---:|---:|
| NVLink（节点内） | ~900 GB/s | ~2 ms |
| NVLink-network / NVSwitch fabric（集群内） | ~400 GB/s | ~5 ms |
| InfiniBand HDR（跨节点） | ~50 GB/s | ~40 ms |
| PCIe Gen5（host 中转） | ~64 GB/s | ~30 ms |

所以两个 pool 同在一个 NVLink domain 里的话，这一跳几乎免费；隔了 IB 的话，是真的税。40 ms 加在 TTFT 上能感觉到，5 ms 是无所谓。

由此马上冒出几个工程旋钮（每一个都够独立成篇 —— 这里我们只点出来，不解决）：

- **Layer-streaming overlap.** 别等 prefill 全跑完再开始传。每一层的 K、V 是按顺序产出的；后面层还在算的时候，前面层的 KV 就已经能往那边发了。做得好的话，传输几乎完全藏在 prefill 算力背后。
- **GPUDirect RDMA.** 字节直接在两块 GPU 的 HBM 之间走，不绕 CPU 内存。省掉一次拷贝、一次 context switch。
- **拓扑感知调度.** 把同一个 request 的 prefill 和 decode 排到拓扑上靠近的 pool —— 同机架、同 NVLink domain —— 把 fabric 那一档压低。
- **前缀复用.** 两个 request 共享一段长前缀的话，只需要算和传 suffix 那段的 KV。生产系统（Mooncake 是个写得比较细的例子）把这件事做成了内存层级的问题：热前缀在 HBM、温前缀在 DRAM、冷前缀在 SSD。
- **GQA / MLA 直接砍单价.** 把 `K_tok` 砍掉 4–8 倍，传输也跟着砍 4–8 倍。一般不把它叫做拆机优化，但实际上是。

每一条底下都能再开一篇文章。这一节的 takeaway 就是：**这次传输是拆机的代价**，但是付得起的 —— 有上限、能工程化、相对于 TTFT/TBT 上的收益来说很小。

用户感受：

- **TTFT** = prefill time + transfer time + 第一次 decode iteration。transfer 是真的在里面，但量级是几 ms 到几十 ms。
- **TBT** = 纯 decode，不会被 prefill 抢算力。decode pool 的每次 iteration 都只装 decode 工作，所以 TBT 平稳到 decode 硬件能给到的极限。

这桩交易就是想要的那种：TTFT 上一次性吃个小亏，换取整段生成里 TBT 又稳又可预测。用户对 TBT 的感受比对 TTFT 重得多 —— TTFT 是一次顿挫，TBT 是每一次顿挫。

---

## 7. 之后还有哪些新问题

Article 05 是给 iteration 封顶。Article 06 是把它拆开。§2 那条公式逼着我们答了"*为什么*要拆"；这一篇大部分篇幅都在做这件事。"*怎么*让它真跑起来"是另一个问题，大部分实际工程量都落在这一边 —— §6 应该读成一扇门，不是终点站，而是一片更大工程面的可见尖端。

在那扇门口站一会儿。两块 GPU，可能在不同机架、可能挂在不同的内存层级下，要在能藏在 prefill 延迟里的时间内，把以 GB 计的状态搬过去。这条管道里每一个选择都有自己很真实的设计空间：

- **字节走哪条 fabric** —— NVLink 还是 NVSwitch 还是 InfiniBand 还是 PCIe —— 单次传输的成本能差出近两个数量级（§6 的表格）。你建出来的集群拓扑长什么样，全看这道选择。
- **request 之间 KV cache 住在哪里** —— HBM 还是 DRAM 还是 SSD —— 让拆机引擎变成了一套分层的内存系统。Mooncake 那套前缀池是其中一种；还有别的实现，invalidation 和 locality 行为各不相同。
- **传输怎么和算力 overlap** —— layer-by-layer streaming、GPUDirect RDMA、双缓冲队列 —— 这些是让一跳"端到端看不见"还是"在 TTFT 里非常显眼"的分水岭。
- **request 怎么在两边 pool 之间路由** —— fabric 局部性感知调度、前缀缓存命中、decode 容量追踪 —— 在 article 05 的所有调度问题之上又叠了一层。

每一条都能独立成篇，系列下一篇要接的就是这根线 —— 跑一套拆机 serving 系统的工程问题。*然后* 我们才能干净地问拆机最终允许我们问的那些优化问题：每个 pool *想要*什么，既然它已经获得了专门化的自由？Pipeline parallelism 给 prefill、Tensor parallelism 给 decode、PagedAttention、GQA/MLA、FlashDecoding、speculative decoding —— 每一项在 pool 拆开之后都有了一个干净的位置，按顺序我们一个一个走。

每次都是同一种语法：找出瓶颈、把 workload 切到每块只看见绑住自己那一种瓶颈的程度、按块去优化。拆机是这种切法里最大的一刀。系列接下来要做的，是这一刀切出来之后该做的工程和优化。
