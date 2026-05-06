---
title: "ORCA 和 chunked prefill：把每次 iteration 的开销摆平"
date: 2026-05-06T00:00:00+00:00
draft: false
summary: "很多 request 同时在跑，结束时间各不相同，有的还带着比一次 decode 大 1000 倍的 prefill。每次 iteration 的开销因此摇摆得厉害。ORCA 那种 iteration-level 调度先收拾一半问题；chunked prefill 再给最大的那次 iteration 封顶，让短任务不被拖在长任务后面。"
description: "iteration-level 调度（ORCA）和 chunked prefill 是怎么把每次 iteration 的开销抚平的。先把开销的方差讲一遍，看 ORCA 修了什么、留了什么，再看 chunked prefill 怎么给最长的 iteration 封顶。"
tags: ["orca", "continuous-batching", "chunked-prefill", "kv-cache", "llm-serving", "mental-model"]
series: ["llm-stories"]
showToc: true
TocOpen: false
weight: 6
---

[Article 04](/llm_stories/posts/04-batching-many-requests/) 把"很多个 prefill 塞进一次 forward"这件事干净地解决掉了。但 prefill 只是一个 request 一生中的前半段。prompt 嚼完之后，request 进入 decode 阶段 —— 一次产出一个 token，有时跑上几百步，直到撞到 EOS 才停。真实的 serving 引擎看不到那种干净整齐的 prefill 批次；它看到的是一锅乱炖：刚到的 prompt、跑到一半的 decode、马上要结束的 request，全在同一时刻共用同一张 GPU。

这一篇要走进这片混乱。生成流程的基础和 KV cache 的机制我们直接借用 [Article 01](/llm_stories/posts/01-llm-end-to-end/) 里讲过的，假设你已经熟悉。

（先把一个名字钉在脑子里，后面会反复用到：**一次 iteration 就是从模型的第 0 层一路 forward 到第 `L−1` 层、走一整遍。**喂进去的内容可以是某个 prompt 的一块、可以是好几个 request 的 decode 步、或者两者混着 —— 不管是什么，一次 iteration 把它们送过 `L` 层一遍。）

想象一下引擎里的几秒钟。几十个 request 正在同时跑：有些还在嚼 prompt、有些已经 decode 到第 50 个 token、有些再吐 1000 步就要结束、还有些刚刚进来。新 request 进来，旧 request 走人。scheduler 的工作就是在伺候好每一个 request 的同时，把 GPU 塞得尽可能满。

由此引出两个问题：

1. **request 的到达时间和结束时间各不相同。** 引擎要怎么在不让谁在生命的起点或终点卡住的前提下，把 GPU 塞满？
2. **每次 iteration 的开销能差出 1000 倍。** 一次纯 decode 的 forward 跑几毫秒就完事；如果其中混了一个 100 k-token 的 prefill，就要好几秒。怎么让每次 iteration 的开销大致平稳，scheduler 才好规划？

两个答案 —— iteration-level 调度（ORCA）和 chunked prefill —— 用的是同一条直觉：**我们要抚平的是每次 iteration 的开销，而不是每个 request 的开销。** ORCA 收拾"到达"和"结束"这两端的边界；chunked prefill 接着给一次 iteration 能装多少东西封顶。

---

## 1. 朴素 batching 在哪些地方崩

想得到的最简单的 scheduler：有空位时挑出 `B` 个 request，一起过 prefill 和 decode，等*最后一个*跑完再把所有人的输出还回去，然后再挑下一批。这就是 *request-level* batching —— batch 是调度单位，batch 里有谁，在被收进来的那一刻就定死了。

它撞上了真实流量的两条铁律。

**request 的到达时间不一样。** 一个 batch 跑了 5 秒，第 200 毫秒进来的新 request 没法加进去 —— batch 里有谁，是在它开跑那一刻定死的。新 request 只能在队里等这一整批跑完。GPU 也许完全装得下再多一个 decoder，但引擎就是不放人进来。新 request 的 **TTFT**（*time-to-first-token* —— 你按下回车到 ChatGPT 的第一个字蹦出来之间的那段空白）从几毫秒膨胀到几秒，*纯粹是干等*。这就是 **convoy effect**：新到的人，就排在当前那批人里最慢那个的后面。

**request 的结束时间不一样。** 一个 batch 里 request A 想要 50 个 output token，request B 想要 1000 个。两个一起 decode。第 50 步之后 A 已经做完了 —— 但它的 slot 收不回来给别人，因为 batch 的形状被钉死了，要等所有人都跑完才能动。接下来 B 还要再 decode ~5 秒，A 的算力 slot 就一直空着。更糟的是，A 已经生成好的那些 token，也得卡到 batch 边界才能还给用户。这就是 **frozen batch size** 问题：寿命短的 request 把最长那个邻居的寿命付了*两遍* —— 一遍是返回延迟，一遍是 GPU 空转。

两个失败都来自同一个根因：**一个静态 batch 只有一个共享寿命，由它所有成员里最长的那个决定。** 比这个最大值短的人在浪费；在开跑之后到达的人在干等。

scheduler 被绑在了错误的粒度上。现实是按 *iteration* 这个粒度在动 —— 每次 forward，每个正在跑的 request 都产出一个 token（或者一块 prefill）。但 scheduler 在按 *batch* 这个粒度做决策 —— 几千次 iteration 才决策一次。当然跟不上。

---

## 2. ORCA：按 iteration 调度，不按 batch

[ORCA 那篇论文](https://www.usenix.org/system/files/osdi22-yu.pdf) 给的修法说起来很简单，后果却很大：把 **iteration** —— 端到端走完 `L` 层的一次 forward —— 当成调度单位。正在跑的 request 集合不再是收进来时定死的名单，而是 scheduler 在每次 forward 之间都在动手整理的一个活东西。

两次 iteration 之间，scheduler 可以：

- **踢人。** 上一次 iteration 出 EOS 的任何 request，slot 立刻释放。
- **进人。** 从队列里挑一个新 request 进来。它在第一次 iteration 就把 prompt 的若干行喂进去做 prefill。
- **带人。** 还在 decode 中间的 request 接着跑，每个这一轮贡献正好 1 行 Q。

这三件事全是 scheduler 在 host 上做的 metadata bookkeeping —— 不动 GPU，只更新每个 request 的元数据。它们在两次 iteration 之间、GPU 还在忙上一次 forward 的同时跑完。

这意味着一次 iteration *里都装了什么* 比 Article 04 灵活了一档。Article 04 的 iteration 是同质的 —— 所有 request 都在 prefill，每个都贡献 prompt 的几行。ORCA 之下，一次 iteration 同时承载处于不同阶段的 request。举个具体例子：

| Request | 状态 | 这次 iter 的 Q 行数 | kv_length |
|---|---|---:|---:|
| A | 第一次 iter 的 prefill，4096-token prompt | 4096 | 4096 |
| B | decode 中间，第 51 步 | 1 | 1500 |
| C | decode 中间，第 200 步 | 1 | 1700 |

这次 iteration 的 Q 总行数：`4096 + 1 + 1 = 4098`。varlen kernel 沿这个 flat tensor 一个 request 一个 request 地走过去，算出三个互相独立的 score block：A 的 `4096 × 4096`（下三角 —— A 在给自己的 token 做 prefill）、B 的 `1 × 1500`、C 的 `1 × 1700`。每个 request 只读*自己*的 KV cache —— request 之间不会互相串。

为了支持这种混合方式，Article 04 那个 `cu_seqlens`（只记 Q 行边界）泛化成每个 request 一个三元组：

```
(q_start, q_end, kv_length)   per request
```

`q_rows = q_end - q_start` 是这个 request 这一轮贡献的 Q 行数。`kv_length` 是这次 iteration 把新的 K、V 追加进去之后，这个 request 完整的 attention 上下文 —— 也就是包含了之前 cache 里的所有内容。Q 行数和 kv_length 不再被强制相等 —— 一个 decoder 是 `q_rows = 1` 配 `kv_length = 1500`，一个全新的 prefill 是两者都等于 4096。

kernel 这边就这一个改动。ORCA 真正的贡献不是一个新的 attention kernel —— 而是一种**调度纪律**：不要把一个 batch 跑到底，每次 iteration 都重新决定谁在里面。kernel 的活在 Article 04 里就已经准备好了；之前缺的是"按 iteration 一次次去用它"这条策略。

现代 serving 系统说的 **continuous batching**，指的就是这件事。

拿到的好处：

- *Convoy effect* 没了 —— 新来的人下一次 iteration 就能进来；等的是一次 iteration（几毫秒），不是一个 batch（几秒）。
- *Frozen batch size* 没了 —— iteration `t` 释放出来的 slot，iteration `t+1` 就能填上；某个 request 跑完，EOS 一采样到就把结果还给用户，不必等到很远的 batch 边界。

两个问题都没了。漂亮的一仗 —— 漂亮到让人想就此宣布"调度问题搞定"然后翻篇。但我们偷偷跳过了一件事。

---

## 3. 下一个问题：iteration 自身的开销也摇摆得厉害

ORCA 把"到达"和"结束"这两端的边界问题修了，靠的是把 iteration 升格成调度单位。但把 iteration 当成调度单位，同时也意味着它变成了**整台引擎的心跳**。所有正在跑的 request —— 不管是 decoder 还是 prefiller —— 每次 iteration 都各自往前走一小步。所以如果 iteration `t` 用了 6 ms、iteration `t+1` 用了 8 秒，那么*任何*正在跑的 decoder 的两个相邻 token 之间，都隔了 8 秒。一次 iteration 的 wall time 不再只是 GPU 内部花了多少算力的私事；它是这一轮里引擎里所有人共同的延迟下限。

那 iteration 的 wall time 到底能波动多大？我们以 **Llama-2-7B** 在单张 H100 上跑为锚，把几种现实里 scheduler 真会拼出来的 iteration mix 套进成本模型走一遍。

<details>
<summary><em>下面用到的 FLOP 和 wall-time 公式（点开看）</em></summary>

Llama-2-7B：multi-head attention，32 层，hidden 4096，head dim 128。Forward 的开销有两个结构上不同的项。

- **Linears**：每行 token 过一遍模型的 forward 开销大致是 `2P` FLOPs，`P ≈ 7×10⁹` —— 也就是每行 token 大约 **14 GFLOPs**。
- **Attention**：每对 (q, k) 在整个网络里的代价 ≈ `4 · d_head · heads · layers = 4·128·32·32` ≈ **每对 0.52 MFLOPs**。一段长度 `L` 的 prefill 配 causal mask：约 `L²/2` 对 ≈ `2.6×10⁵ · L²` FLOPs。一个 decode step 对一份大小为 `M` 的 cache：`M` 对 ≈ `0.52·M` MFLOPs。

H100 有效算力：fp16 compute-bound 任务 ~500 TFLOPs/s，read-bound 任务 ~3.35 TB/s HBM 带宽。一次 decode step 的主要开销是把整个网络的 weight *读一遍*（fp16 下 ~14 GB），不是 FLOPs 本身 —— 所以 decode 是 bandwidth-bound 的，**每步约 5–7 ms**，由 `bytes ÷ HBM` 决定。

</details>

底子是 8 个正在跑的 decoder，每个 context ~1 k。在同一次 iteration 里再塞进一个新 request，看几种情况：

| Iteration mix（8 decode + …）| Linear | Attn | Total | Wall time |
|---|---:|---:|---:|---:|
| 什么也不加（纯 decode） | ~110 GF | ~4 GF | ~115 GF | **~6 ms**（被 weight 带宽卡住）|
| + 1 k-token prefill | ~14 TF | ~0.3 TF | ~14 TF | **~30 ms** |
| + 4 k-token prefill | ~57 TF | ~4 TF | ~61 TF | **~120 ms** |
| + 16 k-token prefill | ~225 TF | ~67 TF | ~290 TF | **~580 ms** |
| + 100 k-token prefill | ~1.4 PF | ~2.6 PF | ~4 PF | **~8 s** |

三个值得注意的规律：

- **Linears 跟 iteration 的总 token 数线性增长。**
- **Attention 跟单个 request 的 prefill 长度平方增长** —— 短的时候可以忽略，到 100 k 这个量级开始压倒一切。
- **scheduler 合理拼出来的 iteration**，wall time 之间能差到 **大约 1300 倍**。

正是最后这个数字把引擎搞崩。要感受一下它的含义：想象你正在用 ChatGPT，下一段正以每秒 ~150 token 顺顺地流着，突然 —— 在你能看见的范围里没有任何理由 —— 模型在一个写到一半的单词上**冻住了八秒钟**才接着写。你的对话本身一点没变。在你看不到的某个地方发生的事是：另一个用户往他自己的会话里粘了一份 10 万 token 的文档，你这一步 decode iteration 刚好被拼进了和他那个 prefill 同一次 forward。对 ORCA 来说这次 iteration 拼起来很正当 —— 两边都是合法的工作 —— 但 wall time 由他那个 prefill 决定，账由你来付。

这种 head-of-line blocking 有两种味道，两种都发生在*一次 iteration 之内*，不是跨 batch。

### 3.1 正在跑的 decoder 的 TBT 尖峰

上面那个场景有个名字：**TBT**（*time-between-tokens*） —— 一个 decode request 相邻两个 output token 之间的等待时间，决定了用户感受到的"匀速流式"那种体验。一个被 100k prefill 拖累的 iteration，会让所有恰好跟它共一轮的正在跑的 decoder 的 TBT 暴增 **~1300 倍**。

静态 batch 不会出这种问题 —— 但静态 batch 有它自己的灾难。ORCA 并没有破坏什么；它只是让一种本来就存在的差异*在 iteration 这个层级上浮出水面*，结果就是：它一出现，就同时砸在引擎里每个人头上。

### 3.2 短 prefill 跟长 prefill 一起跑出来的 TTFT 尖峰

两个新 request 同一轮 iteration 一起到了：一个 prompt 100 token，一个 prompt 10 k token。ORCA 高高兴兴把它俩一起塞进同一次 forward —— 都是要做 prefill、都没有跑到一半的状态要照顾，往一次 iteration 里多塞东西本来就是 kernel 设计来做的事。但这次 forward 的 wall time 由长的那个邻居定：

| Forward 内容 | Linear | Attn | Wall time |
|---|---:|---:|---:|
| 单跑 100-token prefill | ~1.4 GF | ~3 MF | **~4 ms** |
| 单跑 10 k-token prefill | ~140 TF | ~26 TF | **~320 ms** |
| 100 + 10 k 一起跑 | ~141 TF | ~26 TF | **~330 ms** |

短 request 的 TTFT 从单跑时的 ~4 ms 退化到了和长邻居一起跑的 ~330 ms —— **差了 ~80 倍**，纯粹因为它俩共了一次 forward。从短 request 的角度看：整个网络对所有人都在全速跑，*除了它*；这个延迟在*它自己的*请求里找不到任何理由可以解释。这是结构性的 —— iteration 的 wall time 被里面最大那块决定，剩下的人就要陪着付。

### 3.3 同一个根因

3.1 和 3.2 都来自同一个结构性事实：**一次 iteration 的 wall time 由它里面最大那一块工作决定。** ORCA 可以决定一块工作*要不要*进这次 iteration，但决定不了一块工作*有多大*。在最大那一块被封顶之前，iteration 这个心跳就会跳乱。

要把心跳搞稳，就得给最大那一块封顶。这正是 chunked prefill 做的事 —— 而 KV cache 已经替我们把工具准备好了。

---

## 4. Chunked prefill：给最大那块封顶

如果长 prefill 是问题，那为什么不干脆*把它切开*？

结构上其实没什么不让切的理由 —— KV cache 让"切开"这件事变得 trivial。chunk 0 跑完之后，它每一层的 K、V 已经存在 cache 里了。chunk 1 的 attention 直接读就行，跟 decode step 读 cache 是同一个动作。整段 prefill 一次性跑出来的数学和这个一模一样，结构上就是相等的；唯一的差别是这些活*在什么时候*被做。

所以：把一段长 prompt 切成大小为 `C` 的 chunk，每次 iteration 带一块走。走一遍：一个长度为 `N` 的 prompt 在做 prefill 时会发生什么 ——

- 这个 prompt 变成 `⌈N/C⌉` 次 iteration。
- iteration 0 prefill token `[0, C)`。它的 attention 就是 Article 04 那种纯 prefill —— `[C × C]` 的下三角 score block。每一层的 K、V 存进 cache。
- iteration 1 prefill token `[C, 2C)`。它的 attention 现在 Q 行来自这个新 chunk，K、V 行来自*两边*：cache 里的前缀，加上这一块刚算出来的 K、V。score block：`[C × 2C]`。
- ……
- iteration `k` prefill token `[kC, (k+1)C)`。score block：`[C × (k+1)C]`。

chunk `k` 上的 mask 有两块：

- 对着 **cache 里前缀** 的那块 `[C × kC]` 全开 —— 不 mask。前缀里的每个 token 都比这块里任何 token 早，causality 允许全看。
- 对着**这块自己 token** 的那块 `[C × C]` 是下三角 —— chunk 内部要保持 causal。

<svg viewBox="0 0 700 480" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;font-family:system-ui,sans-serif;display:block;margin:1.5rem auto">
  <defs>
    <pattern id="hatch-mask" patternUnits="userSpaceOnUse" width="10" height="10" patternTransform="rotate(45)">
      <rect width="10" height="10" fill="rgba(150,150,150,0.06)"/>
      <line x1="0" y1="0" x2="0" y2="10" stroke="rgba(150,150,150,0.45)" stroke-width="2"/>
    </pattern>
  </defs>
  <text x="350" y="25" text-anchor="middle" font-size="14" fill="currentColor" font-weight="600">chunk k of size C, prefix S = kC: scores [C × (S+C)]</text>
  <text x="270" y="85" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.75">cached prefix keys (S)</text>
  <text x="520" y="85" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.75">this chunk's keys (C)</text>
  <text transform="translate(95,240) rotate(-90)" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.75">queries from chunk k (C rows)</text>
  <rect x="120" y="100" width="300" height="280" fill="rgba(74,144,226,0.25)" stroke="#4a90e2" stroke-width="2"/>
  <text x="270" y="232" text-anchor="middle" font-size="14" fill="currentColor" font-weight="600">attends to cached prefix</text>
  <text x="270" y="252" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.75">all visible — no mask</text>
  <text x="270" y="270" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.7" font-family="ui-monospace,monospace">[C × S]</text>
  <rect x="420" y="100" width="200" height="280" fill="none" stroke="#f5a623" stroke-width="2"/>
  <polygon points="420,100 420,380 620,380" fill="rgba(245,166,35,0.45)"/>
  <polygon points="420,100 620,100 620,380" fill="url(#hatch-mask)"/>
  <text x="470" y="345" font-size="13" fill="currentColor" font-weight="600">causal</text>
  <text x="455" y="362" font-size="11" fill="currentColor" opacity="0.75" font-family="ui-monospace,monospace">[C × C], lower-tri</text>
  <text x="555" y="135" font-size="11" fill="currentColor" opacity="0.55">masked</text>
  <text x="350" y="425" text-anchor="middle" font-size="12" fill="currentColor" font-family="ui-monospace,monospace">q_rows = C, kv_length = S + C</text>
  <text x="350" y="447" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.65" font-style="italic">prefix block fully unmasked; new-token block lower-triangular</text>
</svg>

在一个大小为 `C`、前缀长度 `S = kC` 的 chunk 上走一遍 block：

| 步骤 | 它做什么 | 碰 cache 吗？|
|---|---|---|
| LayerNorm | 按行做 | 不碰 |
| QKV proj | 在 `[C × hidden]` 上 matmul → Q、K、V 各 `[C × heads × d_head]` | 不碰 |
| 把 K、V 追加到 cache | 把这一层的 K、V concat 到这个 request 的 cache 里 | 碰（写入）|
| Attention | Q `[C × heads × d_head]`、K 和 V `[(S+C) × heads × d_head]`。Scores `[C × (S+C)]`，mask 如上。 | 碰（读完整前缀）|
| Output proj | matmul | 不碰 |
| Residual + LayerNorm + FFN-up + GeLU + FFN-down + Residual | 按行做 | 不碰 |

相比 Article 04 唯一变化的是 **attention**，几个形状被泛化了：

- Q 行数是 `C`，不再是"这个 request 的整个长度"。
- K、V 行数是 `S + C`，不再等于 Q —— 前缀现在住在 cache 里。
- score block 是矩形 `[C × (S+C)]`，不再是方阵。

Linears、residuals、layernorms、按元素的操作都是按行做的，根本看不见 cache。它们在 `[C × hidden]` 上一行一行处理，跟任何别的 `C` 行 batch 没有区别。

这个 score block 的形状值得停下来想一想：它是一个混合体。左边那块 —— 这个 chunk 的 query 对前缀 cache —— 长得和*把 `C` 个 decode step 摞在一起*那种 score block 一模一样：对前面所有 token 全开 attention。右边那块 —— chunk 对自己 —— 是一个普通的 causal prefill 的 `[C × C]` block。**decode 和 prefill 是同一个形状的两个极端，chunked prefill 是这个谱系上的任何一点。**

事后再看就一目了然：**decode 不过是 `C = 1` 的 chunked prefill。** 同一套机器，不同的旋钮取值而已。

---

## 5. Piggyback：prefill chunk 跟 decode 共用一次 iteration

现在前面所有的零件要拼起来了。Article 04 那套 flat-tensor + varlen kernel 不在乎一个 request 的切片*是哪种*工作。对 kernel 来说，一个 request 的切片就是 `(q_rows, kv_length)` —— 不管它是在 decode（q_rows = 1）、是在 prefill 自己的第一个 chunk（q_rows = C, kv_length = C），还是在跑中间某个 chunk（q_rows = C, kv_length = S + C），形状都是这一个。

所以一次 iteration 可以承载下面这些东西，全部打包进一个 flat tensor：

```
Iteration content:
  - Request E: prefill chunk 7 of 50    →  1024 Q rows,  kv_length = 8 × 1024 = 8192
  - Request A: decode step 51           →     1 Q row,   kv_length = 1500
  - Request B: decode step 200          →     1 Q row,   kv_length = 1700
  - Request C: decode step 75           →     1 Q row,   kv_length = 1100

Total Q rows in this iteration: 1024 + 3 = 1027
```

varlen kernel 把每个 request 的切片各自走一遍。TP 还是完全没动到。

这就是 **piggyback chunked prefill**：长 prefill 和正在跑的 decode 在一次 forward 里共存。scheduler 的工作变成了一种 bin-packing —— 每次 iteration 有一个预算（比如"Q 不超过 2048 行、iteration 不超过 50 ms"），用 decode step 和 prefill chunk 的任意组合把它填满。一段长 prompt 变成一连串 chunk 大小的贡献，每次 iteration 一块，和当时在跑的所有 decode 一起跑。短 prefill 一次就能跑完。decode 总是塞得下。§3 里那 1300 倍的波动塌缩成一个稳定的 iteration profile —— 大概 2 到 3 倍 —— 容易规划，引擎的心跳又稳了。

`C` 是 scheduler 这边新增的旋钮：

- **`C` 小** → iteration 时间更均匀、正在跑的 decoder 的 TBT 更低；但每个 chunk 的 cache 重读更多，linears 的 MFU 更低（小 GEMM 离峰值更远）。
- **`C` 大** → cache 重读更少、MFU 更高；但 iteration 的 wall time 又开始往上爬，整车人的 TBT 又开始劣化。

真实系统挑 `C` 一般落在 **256–8192** 这个范围，通常跟"每次 iteration 最多多少 token-row"的预算挂钩，预算的目标是控住 TBT 上限。举个具体的：在"每次 iteration ≤ 50 ms、最多 2048 个 Q 行"这套预算下，一段 100 k-token 的 prompt 会被 prefill 成 `100 000 / 2048 ≈ 49` 次 iteration，每一次都跟当时手上的 decode 一起跑。

---

## 6. 成本分析

下面三件事值得停下来想一想，每一条都不是小开销。

**总算力没省。** chunk `k = 0 … N/C − 1` 各自的 `C · (k+1)C` 个 causal pair 加起来等于 `N²/2`。chunked prefill *把* attention 的活在 iteration 之间*重新分配*，但没有把它减少。

**KV 这边的 HBM 带宽消耗涨了。** chunk `k` 在每一层每一次 attention 都要*再读* `kC` 行 cache。所有 chunk 加起来：≈ `N²/(2C)` 行 cache 累计流量，而不切的 prefill 只用 ~`N` 行（tiled attention 把 cache 流过去恰好一遍）。`N = 100 k`、`C = 2048` 时，同一段 prompt 的累计 cache 读带宽大约多了 **25 倍** —— 这是 chunking 为了"把 iteration 封顶"付的价。这也是为什么 `C` 不能无限做小：到某个点之后，带宽税会盖过可调度性带来的收益。

**`C` 小的时候每次 iteration 的 MFU 会掉。** `C` 小的 iteration，linears 的 matmul 跑得离峰值远 —— tensor core 能咬的行数太少。真实 serving 引擎调 `C` 时找的是一个甜点：让 iteration 时间能压到 TBT 目标、又不至于把 MFU 浪费太多。

这三条加起来解释了 `C` 一般落在 **256–8192** 这个区间的原因。没有标准答案；区间具体落在哪儿，取决于模型本身的算力/带宽特性、以及引擎对 TBT 和吞吐的目标。

---

## 7. 之后还有哪些新问题

到这里，我们手里有了一个真正的 serving 循环：prefill、decode、混合 iteration、单次 iteration 的开销有上限、没有空转的 slot。还有几条假设是漏的，每一条都给后面的文章埋下种子。

- **KV cache 的物理布局。** 我们一直默默假设每个 request 的 cache 在每一层都是一段连续的内存。一旦 `B` 涨起来、上下文长度又千差万别，这件事很快就难看 —— 碎片、eviction、分配开销。**PagedAttention** 把 cache 当虚拟内存来处理；下一篇文章。
- **两种 regime 共用一台引擎。** Decode 卡在 weight 读带宽上；prefill chunk 又是 compute-bound。也许它俩根本就不该共用同一组 GPU。**Prefill/decode 拆分** 探索的就是把它们丢到不同 replica 上去跑。
- **head 不总是独立的。** GQA、MLA 以及"减少 KV head"那一家子的其他成员，会大幅压缩 cache —— batch 更大、上下文更长 —— 但也引入一些此前我们可以无视的共享模式。一个子系列。
- **一个 request 的 cache 都装不下一张 GPU。** 上下文长到 KV cache 自己就装不下一张卡的时候，sequence/context parallelism 会把*一个* request 切到多张 GPU 上。一篇专门的文章，挪到后面。

每次都用同一种语法：松开一条假设，看看会掉出什么。
