---
title: "Roadmap：这个系列要写些什么"
date: 2026-04-29T00:00:00+00:00
draft: false
summary: "这个系列是什么，以及一份还在生长的文章地图 —— 已发布、在写、留给未来的自己去填的坑都列在这。"
description: "LLM Stories 系列的活地图。记录已发表的文章，以及每一篇会开出来的下一批问题。"
tags: ["meta"]
series: ["llm-stories"]
showToc: false
weight: 1
---

## 这个系列是什么

这是一份学习笔记 —— 我自己在搞清楚现代 LLM 到底是怎么被 serve 起来的，主要靠跟 Claude 聊，然后把真正想明白的部分写下来。文章本身是用一种"发现之旅"的、比较自信的口吻写的，但底下其实就是一个普通人在公开学习。

下面这张表是**活的** —— 文章会随着发布翻状态，每次讨论挖出新坑就会往里加。

---

## 文章列表

| # | 标题 | 状态 | 链接 |
|---|---|---|---|
| 01 | LLM 从头到尾走一遍 —— 鸟瞰整张网络、单个 block 的内部、生成一段文字的完整过程，以及由此自然冒出的、贯穿整个系列的问题 | `[next]` | — |
| 02 | Tensor parallelism 心智模型：从零搭起 | `[done]` | [阅读 →](/llm_stories/zh/posts/02-tensor-parallelism-mental-model/) |
| 03 | 在一个 transformer block 中完整走完一遍 Tensor Parallelism —— 先全用 column-parallel 看通信怎么爆掉，再配上 row-parallel 落到每个 block 两次 all-reduce | `[done]` | [阅读 →](/llm_stories/zh/posts/03-tp-through-a-full-block/) |
| 04 | 一次 forward 怎么塞下很多个 request —— varlen attention，只考虑 prefill，TP 完全没动到 | `[done]` | [阅读 →](/llm_stories/zh/posts/04-batching-many-requests/) |
| 05 | ORCA 和 chunked prefill —— iteration-level 调度先把进出 batch 的边界问题解决掉；chunked prefill 再给每次 iteration 的开销封顶，免得一个长 prompt 把整台引擎卡住 | `[done] (EN)` | [read →](/llm_stories/posts/05-orca-and-chunked-prefill/) |
| 06 | Pipeline parallelism —— 切 *跨* block 而不是 block *内部*，以及它带来的 bubble | `[planned]` | — |
| 07 | MoE 和 expert parallelism —— FFN 变成 routed 之后改了什么 | `[planned]` | — |
| 08 | Prefill 和 decode 拆机 —— 两个阶段瓶颈不一样，干脆不再共用一个 engine | `[planned]` | — |
| 09 | PagedAttention —— 把 KV cache 当虚拟内存做，分块而不是连续，跨 request 还能 copy-on-write | `[planned]` | — |
| 10 | Sequence 和 context parallelism —— 把一个 *request* 切到多张 GPU 上，ring attention，长上下文那一招 | `[planned]` | — |
| 11 | FlashAttention —— tiled online softmax，为什么 `[L × L]` 的 score matrix 根本不需要存在 | `[speculative]` | — |
| 12 | FlashDecoding —— bandwidth 压力下让 `1 × L_kv` 的 decode-attention 跑快 | `[speculative]` | — |
| 13 | GQA 和 MLA —— 更少的 KV head、更小的 KV cache、更快的 decode（以及模型代价） | `[speculative]` | — |
| 14 | Speculative decoding —— 一个小模型出 draft，大模型来 verify，两次 forward 价钱办一次的事 | `[speculative]` | — |
| 15 | KV 压缩 —— 量化、eviction，能丢的和不能丢的 | `[speculative]` | — |

## 状态说明

`[done]` 已发布、有链接 · `[done] (EN)` 英文版已发布，中文版还在路上 · `[next]` 正在写 · `[planned]` 排队中，肯定会写到 · `[speculative]` 一个值得挖的坑 —— 可能填、可能不填，但问题本身有意思

---

## 几条贯穿多篇的线索

下面这些观察会在多篇文章里反复出现，读的时候可以放在脑子里：

- **TP 出乎意料地不挡道。** Request batching 没动到它（Article 04），continuous batching + chunked prefill 也没动到（Article 05）。但 PP 和 MoE *会* 跟 TP 有一些有意思的互动 —— 这也是为什么这两篇在地图上排得比较靠前。
- **KV cache 是文章 05 之后的连结组织。** 它一旦在 decode 里登场就不会离开了；它也是长上下文之所以困难的根源。
- **Decode 把瓶颈翻了个面。** 02–04 都假设 prefill，那时候 compute 是主要约束。一旦 decode 进来（Article 05 起），weight 读取的 bandwidth 变成卡脖子的那个 —— 这也是后面几乎每个优化（FlashDecoding、GQA、prefill/decode 拆机、speculative decoding）的动机。
- **Modeler 的选择反复在 serving 这边承重。** Multi-head 的独立性让 TP 通信免费，也让 request batching 通信免费，到 GQA/MLA 那篇还会再出现一次。这是一条值得留意的反复出现的主题。
