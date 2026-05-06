---
title: "一次 forward 怎么塞下很多个 request"
date: 2026-05-03T00:00:00+00:00
draft: false
summary: "很多个用户同时打过来，prompt 长度还都不一样。把一整个 transformer block 拿到一个 flatten 起来的多 request tensor 上跑一遍，看哪些 layer 是白送、哪些得真动手 —— 顺便看一下 TP 这边到底要不要改。"
description: "怎么把多个并发的 prefill request batch 起来一次 forward 跑完。把整块 block 摆到 flatten 之后的多 request tensor 上走一遍，看清楚哪些地方 batching 是白送、哪些不是。"
tags: ["batching", "varlen-attention", "selective-batching", "llm-serving", "mental-model"]
series: ["llm-stories"]
showToc: true
TocOpen: false
weight: 4
---

[Article 02](/llm_stories/posts/02-tp-through-a-full-block/) 把一个 transformer block 跑在两张 GPU 上，每层两次 all-reduce 就拿下了。但真实的 serving 系统不会只有一个用户 —— 一堆 prompt 同时打进来，长度还各不相同。一个 50 token 的 *"现在几点了"* 就坐在一份 5,000 token 的论文草稿旁边。

这一篇要追两个问题：

1. **怎么把不等长的 request 高效地塞进一次 forward？** 最自然的做法 —— 把所有 prompt 都 pad 到最长那个，按固定 batch 跑 —— 在短的那些上浪费一大堆算力。肯定有更聪明的办法。
2. **TP 这边要不要知道 batch 这件事？** 还是说 batching 的招数和切模型的故事可以彼此独立？

我们继续沿用 article 02 的 setup，盯着每一层在面对多个 request 时做了什么。

---

## 1. Setup

数字跟 article 02 一样：

| | 值 |
|---|---|
| GPUs | 2 张 (TP=2) |
| layers | 8 |
| `d` (model dim) | 512 |
| `h` (heads) | 8，每张 GPU 4 个 |
| `d_head` | 64 |
| `k = h · d_head` | 512 |
| FFN hidden | `4d` = 2048 |

讨论里我们用两个具体的 request：**request A** 长度 10，**request B** 长度 30。

这一篇明确握着三条假设：

- **只考虑 prefill。** 我们只算每个 request 的 prompt 那次 forward。逐 token 的 decode 还没来 —— 那是 article 04 的事。
- **每个 request 都装得下一个 batch。** 一个 batch 装的是 ≥1 个*整* request，从来不会装"半个"。article 05 用 chunked prefill 来放松这条。
- **暂时不引入 KV cache。** KV cache 是 decode 阶段让*后面*的 token 能看回之前 token 的那个东西。在只有 prefill 的世界里，我们算完输出直接发出去，没什么需要存下来的。KV cache 跟着 article 04 一起到。

这样把空间维度的故事讲干净。*时间*维度的故事（跨 iteration 的 continuous batching）是另一篇。

---

## 2. 先看一个 request：`N` 只是个 tensor 维度

在两个 request 之前，先回忆一下 article 02 v2 里一个 request 长什么样。一个长度 `N` 的 prefill 走过这个 block，shape 就是 `[N × 512]`。从那篇的 trace 里可以看到：

- 8 层 × 每层 2 次 all-reduce = 每次 forward **16 次 all-reduce**。
- 每次 all-reduce 跨 GPU 搬的都是 `[N × 512]`。

值得停一下的是：**`N` 只出现在 tensor shape 里，从来没出现在通信次数里。** 不管 `N=10` 还是 `N=10,000`，你都做正好 16 次 all-reduce。区别只是每次搬的字节数多还是少。

也就是说，给一个 request 加更多的 token 在通信成本上是"白送"的 —— 总字节数线性地涨，但*同步事件*的次数没增加。

这是个不错的性质。下一个要问的是：当多出来的这些 token 来自*不同 request* 的时候，这条性质还能不能保住。

---

## 3. 自然的做法和更聪明的做法

**自然做法：pad 到最长。** A 和 B 摞起来变成 `[2 × 30 × 512]`。A 拿到 20 个 padding token，模型还是要照算。Linear 那边的浪费温和（matmul 大了 2×）。Attention 那边就重了 —— 每个 request 的 attention 是 `O(L²)` 的活，A 的 attention 要算 `30² = 900` 次（per head per layer），实际只需要 `10² = 100` 次。**单 A 一个就多算了 9 倍**，padding token 还对最终结果没贡献。

**更聪明的做法：flatten。** 把 A 和 B 的 token 拼成一个 shape 为 `[(10+30) × 512] = [40 × 512]` 的 tensor。没有 padding，没有 batch 维度 —— 就是一长串 token。

但问题立刻冒出来：**这个 flatten 之后的 tensor 走过一整个 forward 的每一步，还能算出对的结果吗？** 有些步骤显然没问题。有些得想一想。一步一步走过去看看。

---

## 4. 从头到尾把这个 block 走一遍

从输入 `[40 × 512]` 开始，把这个 block 的每一步过一遍。每一步都问：当输入里同时有多个 request 的 token 时，它能不能算出正确答案？

| 步骤 | 它做什么 | 在 `[40 × 512]` 上？ |
|---|---|---|
| LayerNorm | 每行各自归一化 | ✓ 直接没问题 |
| QKV proj (linear) | 跟共享的 `W` 做 matmul | 需要分析 |
| Attention | 逐 request 的 sequence-mixing | 需要分析 |
| Output proj (linear) | 跟共享的 `W` 做 matmul | 需要分析 |
| Residual add | 每行做加法 | ✓ 直接没问题 |
| LayerNorm | 每行各自归一化 | ✓ 直接没问题 |
| FFN-up (linear) | 跟共享的 `W` 做 matmul | 需要分析 |
| Activation (GeLU) | 每个元素的非线性 | ✓ 直接没问题 |
| FFN-down (linear) | 跟共享的 `W` 做 matmul | 需要分析 |
| Residual add | 每行做加法 | ✓ 直接没问题 |

一半的步骤一上来就打钩了。**Pointwise 算子** —— LayerNorm、GeLU、residual add —— 处理每行都是独立的。第 `i` 行属于 request A 还是 request B，对它们来说是不可见的。它们逐 token，互不混合。所以 batching 是白送的。

剩下五步要细看：四个 linear matmul（QKV、output、FFN-up、FFN-down）外加一个 attention block。

但有个便利：**四个 linear matmul 的结构完全一样** —— `Y = X @ W`，`W` 是被所有行共享的。所以只要弄清楚 *一个* linear 在 flatten batching 下的行为，四个全都跟着确定下来。Attention 在每层只有一个。

整个 batching 的问题就缩成两条：

1. **一个 linear layer 在 `[40 × 512]` 上算出来对吗？**
2. **Attention 在 `[40 × 512]` 上算出来对吗？**

§5 处理 linear，§6 处理 attention。这两条解决，整个 block 就解决了。

---

## 5. Linear layer：简单的那一半

回到 [article 01](/llm_stories/posts/01-tensor-parallelism-mental-model/) 是怎么看 linear layer 的。weight matrix 是一排小小的 **feature extractor** —— 每个 `fx` 是它自己一个不透明的小函数，吃一个 token 的 `d` 维 feature vector，吐一个数。一个 `k` 维输出的 linear layer，就是 `k` 个这样的 extractor 并排站着，对同一个 token 一齐工作。

```
token  ⇒  [ fx1   fx2   fx3   ...   fxk ]   ⇒   [ fx1(token), fx2(token), ..., fxk(token) ]
```

值得停一下的事：每个 `fx` 看的是 **一个 token**，吐出 **一个数**。它不偷瞄下一个 token，也不偷瞄上一个 token。它不知道这个 token 来自哪一个对话。**整个 math 里就没有让 request 边界进来的入口**，因为它一次只看一个 token。

那把一个 flat tensor `[40 × 512]` —— 40 个 token 摞起来 —— 喂给它，它就把每个 `fx` 在每个 token 上跑一遍。40 个 token、每个 `k` 个 extractor，填满一个 `[40 × k]` 的输出。前 10 行是 request A、后 30 行是 request B 这件事，对这次操作来说**根本看不到**；它从一开始就没有混淆它们的可能性。

这就是 linear layer 在 batching 下白送的根本原因。它们不是"奇迹般地"能 batch —— 它们本来就是逐 token 的。我们只是让它跑了更多个 token 而已。

**TP=2 的情况：**跟 article 02 一样没变。`fx` 还是按 head 切到两张 GPU 上，每张 GPU 拥有一半：

- G1 在 `[40 × 512]` 上跑 heads 1–4 的 `fx` → `[40 × 768]`
- G2 在 `[40 × 512]` 上跑 heads 5–8 的 `fx` → `[40 × 768]`

all-reduce 的 shape 从 `[N × 512]` 变成 `[40 × 512]`，但 **all-reduce 的次数没变**。同样的通信 pattern，每次搬的字节多了。

而且这个论证对 block 里所有四个 linear matmul 都适用 —— QKV、output、FFN-up、FFN-down —— **所有 linear 全都搞定了。** 只剩一步。

---

## 6. Attention：难的那一半

为什么 attention 不一样？因为 attention 是 **sequence-mixing**。每个 token 的输出依赖 sequence 里 *所有* token，不只是它自己那一行：

```
out[i, :] = softmax( Q[i, :] @ K.T / √d_head ) @ V
```

这里的 `K.T` 和 `V` 是横扫整个 sequence 的。如果 `K` 和 `V` 来自一个同时装着 A 和 B token 的 tensor，那 A 里第 `i` 个 token 默认就会去 attend B 的 token —— 反过来也一样。math *能*跑通，但答案是错的：A 的输出会跟 B 的 key、value 混在一起，这不是模型训练时学到的东西。

我们需要一种办法：让 request A 的 attention 严格只在 A 的 token 范围内做，B 的也严格只在 B 的范围内 —— 但底下的 flat tensor 还是共享的。

### 6.1 自然做法 —— 先算，再 mask

最直接的修法：把整个 `[40 × 40]` 的 attention matrix 当成 40 个 token 是一个 sequence 一样算出来，然后把跨 request 的那些位置 mask 掉（softmax 之前设成 `-∞`，让它们贡献为零）。

这个 flat token buffer 长这样：

<svg viewBox="0 0 500 560" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;font-family:system-ui,sans-serif;display:block;margin:1.5rem auto">
  <text x="250" y="25" text-anchor="middle" font-size="14" fill="currentColor" font-weight="600">flat token tensor: [40 tokens × 512]</text>
  <text x="250" y="46" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.65">each row is one token of d=512 features</text>
  <rect x="180" y="70" width="140" height="110" fill="rgba(74,144,226,0.25)" stroke="#4a90e2" stroke-width="2"/>
  <text x="250" y="118" text-anchor="middle" font-size="14" fill="currentColor" font-weight="600">request A</text>
  <text x="250" y="138" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.75">10 tokens</text>
  <rect x="180" y="180" width="140" height="330" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="2"/>
  <text x="250" y="338" text-anchor="middle" font-size="14" fill="currentColor" font-weight="600">request B</text>
  <text x="250" y="358" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.75">30 tokens</text>
  <text x="335" y="74" font-size="11" fill="currentColor" opacity="0.7">row 0</text>
  <text x="335" y="184" font-size="11" fill="currentColor" opacity="0.7">row 10</text>
  <text x="335" y="514" font-size="11" fill="currentColor" opacity="0.7">row 40</text>
  <text x="250" y="540" text-anchor="middle" font-size="13" fill="currentColor" font-family="ui-monospace,monospace">cu_seqlens = [0, 10, 40]</text>
</svg>

完整的 attention matrix，跨 request 的 block 被 mask 掉之后：

<svg viewBox="0 0 700 680" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;font-family:system-ui,sans-serif;display:block;margin:1.5rem auto">
  <defs>
    <pattern id="hatch-naive" patternUnits="userSpaceOnUse" width="10" height="10" patternTransform="rotate(45)">
      <rect width="10" height="10" fill="rgba(150,150,150,0.06)"/>
      <line x1="0" y1="0" x2="0" y2="10" stroke="rgba(150,150,150,0.45)" stroke-width="2"/>
    </pattern>
  </defs>
  <text x="350" y="25" text-anchor="middle" font-size="14" fill="currentColor" font-weight="600">naive: compute full 40×40, mask cross-request blocks</text>
  <text x="200" y="115" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.75">keys 0..9</text>
  <text x="440" y="115" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.75">keys 10..39</text>
  <text transform="translate(115,200) rotate(-90)" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.75">queries 0..9</text>
  <text transform="translate(115,440) rotate(-90)" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.75">queries 10..39</text>
  <rect x="140" y="140" width="120" height="120" fill="rgba(74,144,226,0.25)" stroke="#4a90e2" stroke-width="2"/>
  <text x="200" y="195" text-anchor="middle" font-size="14" fill="currentColor" font-weight="600">A → A</text>
  <text x="200" y="215" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.75">10 × 10</text>
  <rect x="260" y="140" width="360" height="120" fill="url(#hatch-naive)" stroke="rgba(150,150,150,0.5)" stroke-width="1.5" stroke-dasharray="4 4"/>
  <text x="440" y="195" text-anchor="middle" font-size="13" fill="currentColor" opacity="0.7">masked to −∞</text>
  <text x="440" y="215" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.6">10 × 30</text>
  <rect x="140" y="260" width="120" height="360" fill="url(#hatch-naive)" stroke="rgba(150,150,150,0.5)" stroke-width="1.5" stroke-dasharray="4 4"/>
  <text x="200" y="435" text-anchor="middle" font-size="13" fill="currentColor" opacity="0.7">masked</text>
  <text x="200" y="455" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.6">30 × 10</text>
  <rect x="260" y="260" width="360" height="360" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="2"/>
  <text x="440" y="435" text-anchor="middle" font-size="14" fill="currentColor" font-weight="600">B → B</text>
  <text x="440" y="455" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.75">30 × 30</text>
  <text x="350" y="660" text-anchor="middle" font-size="12" fill="currentColor" font-family="ui-monospace,monospace">computed: 1600   useful: 1000   wasted: 600</text>
</svg>

跑是跑得通，但太浪费了。off-diagonal 那两块 —— `10 × 30` 和 `30 × 10`，加起来 600 个位置 —— 算出来立刻就被丢掉。一旦并发的 request 多起来情况只会更糟：`R` 个长度都是 `L` 的 request，你算了 `(RL)²`，但只用得上 `R · L²`。跨 request 的活按 `R²` 涨，有用的活只按 `R` 涨。serving 系统里 `R` 轻易就能上百，这种做法根本撑不住。

### 6.2 Varlen 的想法 —— 直接跳过，不要 mask

不要先算再 mask，干脆 *只* 算 diagonal 的那些 block。一个 request 一个 request 地循环，每次在它在 flat buffer 里那一段上跑普通 attention：

<svg viewBox="0 0 700 680" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;font-family:system-ui,sans-serif;display:block;margin:1.5rem auto">
  <text x="350" y="25" text-anchor="middle" font-size="14" fill="currentColor" font-weight="600">varlen: compute only the diagonal blocks</text>
  <text x="200" y="115" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.75">keys 0..9</text>
  <text x="440" y="115" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.75">keys 10..39</text>
  <text transform="translate(115,200) rotate(-90)" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.75">queries 0..9</text>
  <text transform="translate(115,440) rotate(-90)" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.75">queries 10..39</text>
  <rect x="140" y="140" width="120" height="120" fill="rgba(74,144,226,0.25)" stroke="#4a90e2" stroke-width="2"/>
  <text x="200" y="195" text-anchor="middle" font-size="14" fill="currentColor" font-weight="600">A → A</text>
  <text x="200" y="215" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.75">10 × 10</text>
  <rect x="260" y="260" width="360" height="360" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="2"/>
  <text x="440" y="435" text-anchor="middle" font-size="14" fill="currentColor" font-weight="600">B → B</text>
  <text x="440" y="455" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.75">30 × 30</text>
  <text x="440" y="200" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.4" font-style="italic">(not computed)</text>
  <text x="200" y="440" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.4" font-style="italic">(not computed)</text>
  <text x="350" y="660" text-anchor="middle" font-size="12" fill="currentColor" font-family="ui-monospace,monospace">computed: 1000 — no waste</text>
</svg>

这就是 **variable-length attention** kernel —— 简称 varlen。它吃一个 flat tensor 加一组 request 边界（`cu_seqlens`，cumulative sequence lengths），然后一个 request 一个 request 地走：

```python
# cu_seqlens = [0, 10, 40]   # request A 占 [0,10)，B 占 [10,40)
for i in range(num_requests):
    s, e = cu_seqlens[i], cu_seqlens[i+1]
    Q_i = Q[s:e]
    K_i = K[s:e]
    V_i = V[s:e]
    scores_i = (Q_i @ K_i.T) / sqrt(d_head)        # L_i × L_i
    probs_i  = softmax(scores_i + causal_mask_i)
    out[s:e] = probs_i @ V_i                       # 写回 flat buffer
```

把这个循环画出来，就是顺着 flat 的 Q、K、V stack 从上往下走：

<svg viewBox="0 0 800 580" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;font-family:system-ui,sans-serif;display:block;margin:1.5rem auto">
  <defs>
    <marker id="arrow-blue" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
      <path d="M 0 0 L 8 4.5 L 0 9 Z" fill="#4a90e2"/>
    </marker>
    <marker id="arrow-amber" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
      <path d="M 0 0 L 8 4.5 L 0 9 Z" fill="#f5a623"/>
    </marker>
  </defs>
  <text x="400" y="25" text-anchor="middle" font-size="14" fill="currentColor" font-weight="600">varlen walks the flat Q, K, V stacks request-by-request</text>
  <text x="260" y="68" text-anchor="middle" font-size="13" fill="currentColor" font-weight="600">Q</text>
  <text x="360" y="68" text-anchor="middle" font-size="13" fill="currentColor" font-weight="600">K</text>
  <text x="460" y="68" text-anchor="middle" font-size="13" fill="currentColor" font-weight="600">V</text>
  <rect x="220" y="80" width="80" height="100" fill="rgba(74,144,226,0.4)" stroke="#4a90e2" stroke-width="2.5"/>
  <rect x="320" y="80" width="80" height="100" fill="rgba(74,144,226,0.4)" stroke="#4a90e2" stroke-width="2.5"/>
  <rect x="420" y="80" width="80" height="100" fill="rgba(74,144,226,0.4)" stroke="#4a90e2" stroke-width="2.5"/>
  <text x="260" y="125" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">A</text>
  <text x="260" y="142" text-anchor="middle" font-size="9" fill="currentColor" opacity="0.75">[0:10]</text>
  <text x="360" y="125" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">A</text>
  <text x="360" y="142" text-anchor="middle" font-size="9" fill="currentColor" opacity="0.75">[0:10]</text>
  <text x="460" y="125" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">A</text>
  <text x="460" y="142" text-anchor="middle" font-size="9" fill="currentColor" opacity="0.75">[0:10]</text>
  <rect x="220" y="180" width="80" height="300" fill="rgba(245,166,35,0.4)" stroke="#f5a623" stroke-width="2.5"/>
  <rect x="320" y="180" width="80" height="300" fill="rgba(245,166,35,0.4)" stroke="#f5a623" stroke-width="2.5"/>
  <rect x="420" y="180" width="80" height="300" fill="rgba(245,166,35,0.4)" stroke="#f5a623" stroke-width="2.5"/>
  <text x="260" y="325" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">B</text>
  <text x="260" y="342" text-anchor="middle" font-size="9" fill="currentColor" opacity="0.75">[10:40]</text>
  <text x="360" y="325" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">B</text>
  <text x="360" y="342" text-anchor="middle" font-size="9" fill="currentColor" opacity="0.75">[10:40]</text>
  <text x="460" y="325" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">B</text>
  <text x="460" y="342" text-anchor="middle" font-size="9" fill="currentColor" opacity="0.75">[10:40]</text>
  <text x="40" y="115" font-size="13" fill="#4a90e2" font-weight="700">i = 0</text>
  <text x="40" y="135" font-size="11" fill="currentColor" opacity="0.85">read slice [0:10]</text>
  <line x1="160" y1="130" x2="215" y2="130" stroke="#4a90e2" stroke-width="2" marker-end="url(#arrow-blue)"/>
  <text x="40" y="320" font-size="13" fill="#f5a623" font-weight="700">i = 1</text>
  <text x="40" y="340" font-size="11" fill="currentColor" opacity="0.85">read slice [10:40]</text>
  <line x1="160" y1="335" x2="215" y2="335" stroke="#f5a623" stroke-width="2" marker-end="url(#arrow-amber)"/>
  <text x="530" y="115" font-size="13" fill="#4a90e2" font-weight="600">step 1 — compute</text>
  <text x="530" y="135" font-size="11" fill="currentColor" opacity="0.85" font-family="ui-monospace,monospace">scores = Q_A @ K_A.T</text>
  <text x="530" y="153" font-size="11" fill="currentColor" opacity="0.85" font-family="ui-monospace,monospace">probs  = softmax(scores)</text>
  <text x="530" y="171" font-size="11" fill="currentColor" opacity="0.85" font-family="ui-monospace,monospace">out[0:10] = probs @ V_A</text>
  <text x="530" y="320" font-size="13" fill="#f5a623" font-weight="600">step 2 — compute</text>
  <text x="530" y="340" font-size="11" fill="currentColor" opacity="0.85" font-family="ui-monospace,monospace">scores = Q_B @ K_B.T</text>
  <text x="530" y="358" font-size="11" fill="currentColor" opacity="0.85" font-family="ui-monospace,monospace">probs  = softmax(scores)</text>
  <text x="530" y="376" font-size="11" fill="currentColor" opacity="0.85" font-family="ui-monospace,monospace">out[10:40] = probs @ V_B</text>
  <text x="400" y="525" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.7" font-style="italic">flat Q, K, V tensors in HBM — varlen kernel slices [s:e] for each request, top to bottom</text>
</svg>

三件值得注意的事：

- 跨 request 的 block 不是被 mask 掉，是**根本没算**。kernel 整块跳过去。
- 每次循环里的 score matrix 大小恰好是*那个 request* 的 —— `[L_i × L_i]`，不是 `[40 × 40]`。所以 score 那块的内存占用一直很小。
- flat 输出 buffer 是这样填的：每个 request 算出来的 attention output 被写回到它自己那一段。

`cu_seqlens` 是 kernel 唯一需要知道的"关于 request 的"信息。剩下的就是 flat tensor 的切片操作。

（实际跑的 kernel 不会真的在 Python 里循环 —— 它把循环放进 GPU、一次 launch 全部搞定，不会每个 request 多付一次 launch overhead。数学内容跟上面这段循环一样，优化版只是表达得更高效。高性能 attention kernel 是后面专门一篇的事。）

### 6.3 在 TP=2 下

每张 GPU 还是各自管 article 02 那 4 个 head。varlen kernel 跑在每张 GPU 自己本地的 Q、K、V 上 —— 对*它的*那 4 个 head，对所有 request 的 token。G1 不需要知道 G2 在 attention 里干什么；G2 的 head 是 G2 的事。article 02 靠的那条"head 之间互相独立"的性质，在这个循环里照样成立。**没有新增任何通信。**

linear（§5）和 attention（§6）都搞定了，整个 block 在 batch 这件事上就都对了。

---

## 7. 退一步看：TP 完全没动到

值得停一下的"意外之喜"。看一下整次 batched forward 里 TP 看到的东西：

- 一个 shape 是 `[tokens × hidden]` 的 tensor 流过每一层。
- weight 按 head 切。
- all-reduce 在 `[tokens × hidden]` 的 partial sum 上做。
- 每个 block 16 次同步事件，跟 article 02 一模一样。

**TP 完全没看到 request 边界。** 那个 flat tensor 在 TP 眼里长的样子，跟 40 个 token 来自 1 个 request 还是 50 个 request，没有任何区别。request 边界只在一个地方进入 —— varlen attention kernel 里的 `cu_seqlens` 参数 —— 而这个参数完全在每张 GPU 自己的本地切片里用，没有引发任何通信。

也就是说，request batching 和 TP 是**两条互相不挡道的轴，唯一相交的地方在 attention kernel 内部**：

- **TP** 回答的是：*模型怎么切到多张 GPU 上？*
- **Request batching** 回答的是：*token 怎么打包进一次 forward？*

这两个问题彼此不约束。我们没有为这一点做过设计 —— 它是从下面两条本来就成立的事实里掉出来的：

- linear layer 是逐 token 的（所以即便在一张 GPU 上也看不到 request 边界）。
- multi-head attention 的 head 互相独立（所以每张 GPU 自己的 per-head varlen 循环根本不需要跟其他 GPU 说话）。

article 02 的收尾说，multi-head attention 是 modeler 留给做系统的人的一份礼物，让 TP 通信免费。这里能看到这份礼物又往前走了一层：让 TP 通信免费的那条 head 独立性，*同样*让 request batching 通信免费。两件本来无关的招数因为同一条架构性质，刚好可以白嫖式地组合起来。

---

## 8. 计算量分析

把 request flatten 起来之后，时间到底花在哪里，老实讲几句。

**Linear layer** 看起来很爽。一份 weight 从 HBM 读上来，被摊到 flat tensor 的 `(N+M)` 个 token 上摊薄。塞进去的 token 越多，GPU 越接近 compute 的峰值。这就是为什么激进的 prefill batching 在 throughput 上是稳赚不赔的。

**Attention** 这边复杂一些。每个 request 的 `Q_i K_i.T` 是它自己那次 matmul，没法像 linear 那样把所有 request 融成一次大 GEMM。现代 varlen kernel 把循环放进 GPU 一次 launch，所以不会按 request 数付 launch overhead。但每个 request 的 attention 还是 `O(L_i²)`，瓶颈长什么样很大程度上取决于 request 长度的*分布*。

想象两种 batch，总 token 数一样：

- **10 个 request × 每个 1,000 token** —— attention 的活是 `10 × 1000² = 10⁷`（per head per layer）。大。**Attention 主导**整次 forward。
- **1,000 个 request × 每个 10 token** —— attention 的活是 `1000 × 10² = 10⁵`，少了 100 倍。**Linear 主导。**

其实就是 §6.2 那张 varlen 方块图，被推到了两个极端。把全部 10,000 个 token 摆在 attention 矩阵的一条边上，varlen 真正会算的只有 per-request 那些对角块，剩下的都是跨 request 的格子，直接跳过：

<svg viewBox="0 0 760 480" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;font-family:system-ui,sans-serif;display:block;margin:1.5rem auto">
  <defs>
    <pattern id="hatch-cost-zh" patternUnits="userSpaceOnUse" width="10" height="10" patternTransform="rotate(45)">
      <rect width="10" height="10" fill="rgba(150,150,150,0.06)"/>
      <line x1="0" y1="0" x2="0" y2="10" stroke="rgba(150,150,150,0.4)" stroke-width="1.5"/>
    </pattern>
  </defs>
  <text x="380" y="25" text-anchor="middle" font-size="14" fill="currentColor" font-weight="600">总 token 数都是 10,000，attention 的活差很多</text>
  <text x="380" y="46" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.65">彩色 = per-request 真正算的部分；斜纹 = 跨 request，varlen 直接跳过</text>
  <rect x="60" y="80" width="280" height="280" fill="url(#hatch-cost-zh)" stroke="rgba(150,150,150,0.5)" stroke-width="1.5"/>
  <g fill="rgba(74,144,226,0.45)" stroke="#4a90e2" stroke-width="1">
    <rect x="60"  y="80"  width="28" height="28"/>
    <rect x="88"  y="108" width="28" height="28"/>
    <rect x="116" y="136" width="28" height="28"/>
    <rect x="144" y="164" width="28" height="28"/>
    <rect x="172" y="192" width="28" height="28"/>
    <rect x="200" y="220" width="28" height="28"/>
    <rect x="228" y="248" width="28" height="28"/>
    <rect x="256" y="276" width="28" height="28"/>
    <rect x="284" y="304" width="28" height="28"/>
    <rect x="312" y="332" width="28" height="28"/>
  </g>
  <text x="200" y="385" text-anchor="middle" font-size="13" fill="currentColor" font-weight="600">10 个 request × 每个 1,000 token</text>
  <text x="200" y="408" text-anchor="middle" font-size="12" fill="currentColor" font-family="ui-monospace,monospace">10 × 1000² = 10⁷</text>
  <text x="200" y="430" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.75">attention 主导整次 forward</text>
  <rect x="440" y="80" width="280" height="280" fill="url(#hatch-cost-zh)" stroke="rgba(150,150,150,0.5)" stroke-width="1.5"/>
  <line x1="440" y1="80" x2="720" y2="360" stroke="#f5a623" stroke-width="2.5" stroke-linecap="square"/>
  <text x="580" y="385" text-anchor="middle" font-size="13" fill="currentColor" font-weight="600">1,000 个 request × 每个 10 token</text>
  <text x="580" y="408" text-anchor="middle" font-size="12" fill="currentColor" font-family="ui-monospace,monospace">1000 × 10² = 10⁵</text>
  <text x="580" y="430" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.75">对角块缩成一根细线；linear 主导</text>
</svg>

外框一样大，总 token 数一样，但彩色那一块——真正会算的部分——从左到右缩了 100 倍。attention 的 L² scaling 意味着：长上下文 batch 是 attention-compute-bound；短 request 多的 batch 是 linear-bandwidth-bound。flatten 这套招在两种 regime 下都一样，但瓶颈换了位置。

这也是 decode 那一篇的预告：当每个"request"一次只生成一个 token 时，per-request 的 `Q_i K_i.T` 退化成一个 `1 × L_kv` 的向量乘上一个 `L_kv × d_head` 的矩阵。arithmetic intensity 掉到 ~1，per-request 的 matmul 不再"够大"，整次 forward 变成被 weight 的 bandwidth 卡住的状态。完全不一样的优化目标 —— 这就是为什么 decode 自成一篇。

---

## 9. 这一篇打开了哪些门

到这里，多个并发 prefill request 在 TP 模型上要怎么跑，我们已经有一套方案了：把 token flatten 成一个 tensor，每个 linear layer 都是一次大 matmul，每个 attention block 都是一次 varlen attention。模型本身的 TP 通信 pattern 不变。naive padding 的浪费消失了。每个 request 拿到的算力都是它真正需要的那么多 —— 不多不少。

下面三个跟进的问题各自值得一篇：

- **如果一个 request 要生成很多个输出 token 呢？** Prefill 是一个 prompt 一次过完。Decode 多了一个逐 token 的阶段，瓶颈完全不同，还多了一个新结构（KV cache）来记住之前的 token。**Article 04 —— decode 和跨 iteration 的 continuous batching。**
- **如果某个 request 长到一个 batch 装不下呢？** "每个 request 都装得下"这条假设有时候就是会破。修法是 *chunked prefill* —— 把 prompt 切成一段一段处理，沿路把 KV cache 一段段建起来。**Article 05。**
- **varlen attention kernel 在 GPU 上到底怎么跑得快？** 我们整篇用的是 naive 的 attention 数学。高性能那一版（FlashAttention）干脆不让 score matrix 实例化出来，用 tiled online-softmax 的递推。这是 kernel 层级的深入，值得专门一篇，会在系列后面出现。

每次都是同一个语法：从这一篇拿走一条假设，放掉，看看会发生什么。
