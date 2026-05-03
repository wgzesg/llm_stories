---
title: "Roadmap：这个系列要写些什么"
date: 2026-04-29T00:00:00+00:00
draft: false
summary: "这个系列是什么，以及一份还在生长的文章地图 —— 已发布、在写、在排队的都列在这。"
description: "LLM Stories 系列的活地图。记录已发表的文章，以及每一篇会开出来的下一批问题。"
tags: ["meta"]
series: ["llm-stories"]
showToc: false
weight: 1
---

## 这个系列是什么

这是一份学习笔记 —— 我自己在搞清楚现代 LLM 到底是怎么被 serve 起来的，主要靠跟 Claude 聊，然后把真正想明白的部分写下来。文章本身是用一种"发现之旅"的、比较自信的口吻写的，但底下其实就是一个普通人在公开学习。

下面这张表是**活的** —— 文章会从 `[wip]` 翻成 `[done]`，新的方向冒出来时，roadmap 会跟着长。

## 文章列表

| # | 标题 | 状态 | 链接 |
|---|---|---|---|
| 01 | Tensor parallelism 心智模型：从零搭起 | `[done]` | [阅读 →](/llm_stories/zh/posts/01-tensor-parallelism-mental-model/) |
| 02 | Pipeline parallelism 在 vanilla attn + FFN block 里怎么走 —— Megatron 在一个 block 里 column-wise 和 row-wise TP 的交错 | `[wip]` | — |
| 03 | MoE —— 用 experts 替掉 FFN 之后，前面那套要怎么改 | `[planned]` | — |
| 04 | Continuous batching 和不等长 sequence —— 会把 vanilla attention 改成什么样 | `[planned]` | — |

## 状态说明

`[done]` 已发布、有链接 · `[wip]` 正在写 · `[planned]` 排队中
