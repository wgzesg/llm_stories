---
title: "在一个 transformer block 中完整走完一遍 Tensor Parallelism"
date: 2026-04-29T00:00:00+00:00
draft: false
summary: "把 article 01 的两种 cut 方式拿到一整个 transformer block 上跑一遍，盯着每一步每张 GPU 上的 shape。先把一种 cut 用到所有 matmul 上 —— 通信爆炸，每个 block 四次 gather。再把两种 cut 配成一对，刚好对上 widen-narrow 的架构节奏，落到每个 block 两次 all-reduce。"
description: "怎么把一整个 transformer block 切上两张 GPU。先全用 column-parallel，看一下为什么每个 block 要付四次 gather，再配上 row-parallel，自然走到 Megatron 那个每个 block 两次 all-reduce 的经典 pattern。每一步的 shape 都标在表里。"
tags: ["tensor-parallelism", "transformers", "llm-serving", "megatron", "mental-model"]
series: ["llm-stories"]
showToc: true
TocOpen: false
weight: 3
---

[Article 01](/llm_stories/posts/01-tensor-parallelism-mental-model/) 留给我们两种把一次 matmul 切上两张 GPU 的方式。直接看它们*在做什么*，比记 paper 里那些名字省事得多。先把两种摆一起对一下：

|                          | **Strategy A** —— *切 `fx`*                  | **Strategy B** —— *切行*                            |
|--------------------------|--------------------------------------------------------|----------------------------------------------------------------|
| 切的是矩阵的什么        | 列（每列是一个 `fx`）                                  | 行（每行是一个基向量）                                          |
| 每张 GPU 拿到的**输入** | 每个 token 的**完整**输入向量                          | 每个 token 输入特征的**一半**                                    |
| 每张 GPU 算出的**输出** | 每个 token 输出特征的**一半**                           | 每个 token 完整输出的一个 **partial sum**                       |
| 怎么合起来              | **concatenate**（免费）                                | **all-reduce**（一次通信）                                     |
| 别名                    | column-parallel                                        | row-parallel                                                    |

每一列最精简的读法：**A = "full in, half out"**，**B = "half in, sum out"**。后面所有东西都建立在这两行字上。

但一个真实的 transformer block 不是单次 matmul，是 **四次**，再加一些 pointwise 的胶水。下一个自然冒出来的问题就是：**一整个 block 该怎么切上两张 GPU？**

第一反应有一个非常自然的方案，*差点*就跑通了。我们先把它搭出来，看清楚它在哪儿坏掉，再让"补这个洞"的过程把我们带到 Megatron 那个经典 pattern。整篇都用一组小数字盯着 shape 走，免得讨论变空气。

---

## 1. 起手：一组装得下的小数

两张 GPU，叫 **G1** 和 **G2**。一个小 batch 配一个小模型：

| | 值 |
|---|---|
| **batch** `n` | 4 个 token |
| **model dim** `d` | 512 |
| **heads** `h` | 8 |
| **per-head dim** `d_head` | 64 |
| **attention dim** `k = h · d_head` | 512 |
| **FFN hidden** | `4d` = 2048 |

每个 token 是一行 512 个数。整个 batch 就是 `[n × d] = [4 × 512]`。

把一个 transformer block 拍平：

```
        ← input: [4 × 512]
    │
  LayerNorm
    │
  QKV projection      d → 3k       ← matmul   weight  [d × 3k] = [512 × 1536]
    │
  attention                          (Q 和 K 互相混；不是一次新的 matmul)
    │
  output projection   k → d        ← matmul   weight  [k × d]  = [512 × 512]
    │
    + residual
    │
  LayerNorm
    │
  FFN up-projection   d → 4d       ← matmul   weight  [d × 4d] = [512 × 2048]
    │
  activation (GeLU)                  (pointwise)
    │
  FFN down-projection 4d → d       ← matmul   weight  [4d × d] = [2048 × 512]
    │
    + residual
    │
```

**四次 matmul**，加一些胶水。

> **顺嘴说一下 pointwise 这些胶水：为什么两张 GPU 都要算一遍？** LayerNorm、activation、residual add 这些都是 **pointwise** 的。它们不在乎数据怎么分布在 GPU 上，*只要本地有它要的那份就行*。在 TP 里我们偷个懒：**只要某份数据完整地在两张 GPU 上都有，就让两张 GPU 各跑一遍这个 pointwise op**。同样输入、同样输出，重复算。为什么不让一张 GPU 算完再 broadcast？因为 **瓶颈一直是通信，不是算力**。pointwise 在 GPU 上对几千个数做一遍，几乎不花时间；跨 GPU 发数据是真金白银的 latency 和 bandwidth。多花点便宜的算力比省那点通信划算。把这条记着 —— 后面的 trace 表里你会看到每个 LN 和 residual 步骤都标着 "(冗余)"，就是这个意思。

所以这个 block 整个 TP 故事，全在那四次 matmul 上。两张 GPU，四个 cut 要做。开搞。

---

## 2. v1 —— 把 Strategy A（full → half）用到每个 matmul 上

最自然的第一步是什么？回到 article 01，Strategy A 长这样：

- **便宜**的那种 cut（concatenate，matmul 内部不需要 all-reduce）；
- 用在 QKV 上**正好落在 head 边界**：`k = 8 · 64 = 512`，每张 GPU 256，正好一人 4 个 head；
- 而且"完整输入进，半个输出出"也是更好脑补的那一版。

那就把 A 一路用到底，四次 matmul 全是 A。一步一步走过这个 block，盯着每张 GPU 手上有什么 —— 它的 **weight 切片**、**输入**、**输出**：

<table class="tp-trace">
<thead>
<tr><th>Step</th><th>GPU 1</th><th>GPU 2</th></tr>
</thead>
<tbody>
<tr>
  <td class="step-label">input</td>
  <td><code>[4×512]</code> 完整</td>
  <td><code>[4×512]</code> 完整</td>
</tr>
<tr>
  <td class="step-label">LayerNorm <span class="note">(冗余)</span></td>
  <td>in <code>[4×512]</code> → out <code>[4×512]</code></td>
  <td>in <code>[4×512]</code> → out <code>[4×512]</code></td>
</tr>
<tr>
  <td class="step-label">QKV proj (A)</td>
  <td>W <code>[512×768]</code> (heads 1–4)<br>in <code>[4×512]</code> → out <code>[4×768]</code><br><span class="note">= heads 1–4 的 Q+K+V，每份 <code>[4×256]</code></span></td>
  <td>W <code>[512×768]</code> (heads 5–8)<br>in <code>[4×512]</code> → out <code>[4×768]</code><br><span class="note">= heads 5–8 的 Q+K+V，每份 <code>[4×256]</code></span></td>
</tr>
<tr>
  <td class="step-label">attention</td>
  <td>heads 1–4<br>in <code>[4×768]</code> → out <code>[4×256]</code></td>
  <td>heads 5–8<br>in <code>[4×768]</code> → out <code>[4×256]</code></td>
</tr>
<tr class="sync">
  <td colspan="3"><span class="star">★</span> GATHER #1 —— output proj 要的是完整 <code>k=512</code>，每张 GPU 只有 256，凑出 <code>[4×512]</code></td>
</tr>
<tr>
  <td class="step-label">output proj (A)</td>
  <td>W <code>[512×256]</code><br>in <code>[4×512]</code> → out <code>[4×256]</code></td>
  <td>W <code>[512×256]</code><br>in <code>[4×512]</code> → out <code>[4×256]</code></td>
</tr>
<tr class="sync">
  <td colspan="3"><span class="star">★</span> GATHER #2 —— residual 要完整 <code>d=512</code>，输出只有半个 <code>d=256</code>，凑出 <code>[4×512]</code></td>
</tr>
<tr>
  <td class="step-label">+ residual</td>
  <td><code>[4×512]</code> → <code>[4×512]</code></td>
  <td><code>[4×512]</code> → <code>[4×512]</code></td>
</tr>
<tr>
  <td class="step-label">LayerNorm <span class="note">(冗余)</span></td>
  <td>in <code>[4×512]</code> → out <code>[4×512]</code></td>
  <td>in <code>[4×512]</code> → out <code>[4×512]</code></td>
</tr>
<tr>
  <td class="step-label">FFN-up (A)</td>
  <td>W <code>[512×1024]</code><br>in <code>[4×512]</code> → out <code>[4×1024]</code></td>
  <td>W <code>[512×1024]</code><br>in <code>[4×512]</code> → out <code>[4×1024]</code></td>
</tr>
<tr>
  <td class="step-label">activation <span class="note">(pointwise)</span></td>
  <td><code>[4×1024]</code> → <code>[4×1024]</code></td>
  <td><code>[4×1024]</code> → <code>[4×1024]</code></td>
</tr>
<tr class="sync">
  <td colspan="3"><span class="star">★</span> GATHER #3 —— FFN-down 要完整 <code>4d=2048</code>，每张 GPU 只有 1024，凑出 <code>[4×2048]</code></td>
</tr>
<tr>
  <td class="step-label">FFN-down (A)</td>
  <td>W <code>[2048×256]</code><br>in <code>[4×2048]</code> → out <code>[4×256]</code></td>
  <td>W <code>[2048×256]</code><br>in <code>[4×2048]</code> → out <code>[4×256]</code></td>
</tr>
<tr class="sync">
  <td colspan="3"><span class="star">★</span> GATHER #4 —— residual 要完整 <code>d=512</code>，输出只有半个 <code>d=256</code>，凑出 <code>[4×512]</code></td>
</tr>
<tr>
  <td class="step-label">+ residual</td>
  <td><code>[4×512]</code> → <code>[4×512]</code></td>
  <td><code>[4×512]</code> → <code>[4×512]</code></td>
</tr>
</tbody>
</table>

**每个 block 四次跨 GPU 的 gather。**

其中两次发生在下一个 A 风格的 matmul 之前 —— 它要的是完整输入。另外两次在 residual add 之前 —— 它要完整向量，但我们刚算出来的是半个。本质都是同一个原因：**A 产出半个输出，下游几乎所有东西要的都是完整输入。**

---

## 3. v1 的代价

跨 GPU 通信是分布式计算里 *慢* 的那一环。整个 TP 设计的目标，就是把它压到尽可能少。v1 的现状是：几乎每一个需要完整 feature 的算子前面，都得付一次 gather。

一个 32 层的模型，光一次 forward 就 ~130 次跨 GPU 通信。太多了。

问题就变成了：

> **能不能把 gather 省掉？**

每次 gather 的存在都是一个原因：下一个算子要完整向量，但 Strategy A 给的是半个。我们真正想要的，是一个 *愿意* 直接吃半个输出的 matmul。

article 01 已经把它递到我们手上了。

---

## 4. v2 —— 让 Strategy A 配上 Strategy B（half → sum）

换一个角度看这两种 strategy：

- **Strategy A** *输出* 一个 **half**。
- **Strategy B** *输入* 一个 **half**。

**形状一样。** A 的输出正好是 B 想要的输入。两者咬合上，中间一点通信都不需要。

把 v1 里的 "A → gather → A" 换成 "A → B"。B 直接吃下半个输出，而通信代价只剩下 B 末尾那一次 —— 把 partial sum 加成完整输出的 all-reduce，给后面的 residual 和 LN 用。

把这个套路用到整个 block，每个 A 配一个 B：

<table class="tp-trace">
<thead>
<tr><th>Step</th><th>GPU 1</th><th>GPU 2</th></tr>
</thead>
<tbody>
<tr>
  <td class="step-label">input</td>
  <td><code>[4×512]</code> 完整</td>
  <td><code>[4×512]</code> 完整</td>
</tr>
<tr>
  <td class="step-label">LayerNorm <span class="note">(冗余)</span></td>
  <td>in <code>[4×512]</code> → out <code>[4×512]</code></td>
  <td>in <code>[4×512]</code> → out <code>[4×512]</code></td>
</tr>
<tr>
  <td class="step-label">QKV proj (A)</td>
  <td>W <code>[512×768]</code> (heads 1–4)<br>in <code>[4×512]</code> → out <code>[4×768]</code><br><span class="note">= heads 1–4 的 Q+K+V，每份 <code>[4×256]</code></span></td>
  <td>W <code>[512×768]</code> (heads 5–8)<br>in <code>[4×512]</code> → out <code>[4×768]</code><br><span class="note">= heads 5–8 的 Q+K+V，每份 <code>[4×256]</code></span></td>
</tr>
<tr>
  <td class="step-label">attention</td>
  <td>heads 1–4<br>in <code>[4×768]</code> → out <code>[4×256]</code></td>
  <td>heads 5–8<br>in <code>[4×768]</code> → out <code>[4×256]</code></td>
</tr>
<tr>
  <td class="step-label">output proj (B)</td>
  <td>W <code>[256×512]</code><br>in <code>[4×256]</code> → out <code>[4×512]</code> <span class="note">(部分和)</span></td>
  <td>W <code>[256×512]</code><br>in <code>[4×256]</code> → out <code>[4×512]</code> <span class="note">(部分和)</span></td>
</tr>
<tr class="sync">
  <td colspan="3"><span class="star">★</span> ALL-REDUCE #1 —— 把两张 GPU 上的两个 <code>[4×512]</code> 部分和加起来，凑出完整 <code>[4×512]</code>（residual + LN 要用）</td>
</tr>
<tr>
  <td class="step-label">+ residual</td>
  <td><code>[4×512]</code> → <code>[4×512]</code></td>
  <td><code>[4×512]</code> → <code>[4×512]</code></td>
</tr>
<tr>
  <td class="step-label">LayerNorm <span class="note">(冗余)</span></td>
  <td>in <code>[4×512]</code> → out <code>[4×512]</code></td>
  <td>in <code>[4×512]</code> → out <code>[4×512]</code></td>
</tr>
<tr>
  <td class="step-label">FFN-up (A)</td>
  <td>W <code>[512×1024]</code><br>in <code>[4×512]</code> → out <code>[4×1024]</code></td>
  <td>W <code>[512×1024]</code><br>in <code>[4×512]</code> → out <code>[4×1024]</code></td>
</tr>
<tr>
  <td class="step-label">activation <span class="note">(pointwise)</span></td>
  <td><code>[4×1024]</code> → <code>[4×1024]</code></td>
  <td><code>[4×1024]</code> → <code>[4×1024]</code></td>
</tr>
<tr>
  <td class="step-label">FFN-down (B)</td>
  <td>W <code>[1024×512]</code><br>in <code>[4×1024]</code> → out <code>[4×512]</code> <span class="note">(部分和)</span></td>
  <td>W <code>[1024×512]</code><br>in <code>[4×1024]</code> → out <code>[4×512]</code> <span class="note">(部分和)</span></td>
</tr>
<tr class="sync">
  <td colspan="3"><span class="star">★</span> ALL-REDUCE #2 —— 把两张 GPU 上的两个 <code>[4×512]</code> 部分和加起来，凑出完整 <code>[4×512]</code>（residual + LN 要用）</td>
</tr>
<tr>
  <td class="step-label">+ residual</td>
  <td><code>[4×512]</code> → <code>[4×512]</code></td>
  <td><code>[4×512]</code> → <code>[4×512]</code></td>
</tr>
</tbody>
</table>

**每个 block 两次 all-reduce。**

这就是 Megatron pattern。没人告诉我们答案，我们自己一步一步走进去的。

---

## 5. 没料到的对偶性

article 01 把 A 和 B 当两种独立 strategy 介绍 —— 同一个矩阵的两种读法。但把它们并排放着看，盯着每种的输入输出形状：

- **A** 拿 **完整输入**，吐 **半个输出**。
- **B** 拿 **半个输入**，吐 **完整 sum** 作为输出。

它们不是两种 strategy，是 **同一次往返的两半**。A 的输出 shape *就是* B 的输入 shape；B 的输出 shape（all-reduce 之后）*就是* A 的输入 shape。你发明 A 的同时，其实就把 B 当成它的回程一起发明出来了。

再回头看这个 block 在做什么：

- **Attention** 是 *widen*（QKV：`d → k`）后跟 *narrow*（output proj：`k → d`）。
- **FFN** 也是 *widen*（`d → 4d`）后跟 *narrow*（`4d → d`）。

widen 那里输出 feature 多，正好分到多张 GPU —— A 派得上用场。narrow 那里输入 feature 多，正好把输入切到多张 GPU，输出小再加回来 —— B 派得上用场。

这个 block 不是"碰巧"对 A→B 友好，是 **结构上**就对 A→B 友好：两对 widen-narrow，中间用 pointwise 粘起来。"Megatron pattern" 不是哪个人坐下来设计出来的算法，它就是唯一一个尊重架构本身做的事的通信 pattern。A、B 对偶和 widen-narrow 节奏，是同一件事讲两遍。

提一句通信代价：一次 gather 和一次 all-reduce 在每张 GPU 上搬的数据量差不多（all-reduce 内部大致是 reduce-scatter 加 all-gather）。v1 每个 block **4 次 gather**；v2 每个 block **2 次 all-reduce** —— 通信砍一半，模型一行没改。

---

## 6. 为什么 cut 必须落在 head 边界

v2 trace 偷偷假设了一件事：QKV 的 column cut 是沿着 **head 边界** 把 `k = 512` 切成两块各 256，每张 GPU 正好拿 4 整个 head。这个假设干的活比看起来多得多。试一下反事实就知道。

想象一下 **single-head attention** —— 同样 `k = 512`，但只有一个 head，没有 head 结构。Strategy A 照旧用到 QKV 上：每张 GPU 拿到形状都是 `[4 × 256]` 的 `Q, K, V`。开始算 attention。

第一步是 `Q Kᵀ`。每张 GPU 算 `Q_half @ K_halfᵀ`，得到一个 `[4 × 4]` 的矩阵 —— 但这个矩阵是它手上那 256 个 feature 的 **partial sum**。真正的 scores 得把两张 GPU 上的 partial 加起来才行。

接下来麻烦了：下一步是 **softmax**。Softmax 是非线性的，本地算完再合不行 —— `softmax(a) + softmax(b) ≠ softmax(a + b)`。reduction 必须发生在 softmax *之前*。也就是说，attention 中间会硬塞进一次同步：

> ★ ALL-REDUCE，对 `[n × n]` 的 scores，发生在 softmax 之前。

这是 v2 那两次之外的第三次 all-reduce。Megatron pattern 直接塌成 *三* 次同步，而新增的这次的 tensor 大小随 sequence length 平方增长 —— 你最不想让它变大的那个。

修法不在算法层，在结构层：不要让 cut 跨过 head。**每个 head 的 `Q Kᵀ` 必须完整地落在一张 GPU 上**，partial sum 的问题就不会出现。multi-head attention 把这个白送给我们了：head 在定义上就互相独立，head 边界天然是切点；只要 `h` 能被 GPU 数整除，对 `k = h · d_head` 的 column split 就正好落在两个 head 之间。

所以 multi-head 不是系统人捡了 modeler 的便宜。它是 v2 能存在的 **结构性前提**。cut 落到 head 内部，softmax 立刻逼出一次毁掉一切的同步；cut 落在 head 之间，非线性就保持本地。Megatron pattern 不是 *碰巧* 在 multi-head 架构上能跑 —— 它要求 multi-head 架构。

---

## 7. 这一篇打开了哪些门

到这里，**一个 block** 跑在两张 GPU 上，每次 forward 两次 all-reduce。下一轮"等下，那……"的问题就开始了：

- **如果 block 很多、GPU 也很多呢？** TP 切的是 block *内部*。切 *跨* block —— 把整个 block 摆到不同 GPU 上，让 microbatch 流过去 —— 是另一回事。**Pipeline parallelism**，下一篇见。
- **如果 FFN 被换成 expert 呢？** column-then-row 这一套对每个 expert 的 matmul 还是适用，但把 token 路由到正确的 expert 又引入一种新通信。**MoE**，再下一篇。
- **如果 batch 里 sequence 长度差异巨大呢？** 通信 pattern 不变，但 attention 那边要处理变长 sequence —— continuous batching 由此登场。

同样的语法。每条都自成一篇 walk-through。
