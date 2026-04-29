---
title: "Roadmap"
date: 2026-04-29T00:00:00+00:00
draft: false
summary: "What this series is, and a living map of the articles — shipped, in progress, and on deck."
description: "A living roadmap of the LLM Stories series. Tracks shipped articles and the questions each one opens up next."
tags: ["meta"]
series: ["llm-stories"]
showToc: false
weight: 0
---

## About this series

These are learning notes — me working through how modern LLMs are actually served, mostly by talking to Claude and writing up the parts that finally clicked. The articles themselves are written in a confident "discovery journey" voice, but the project underneath is just someone learning in public.

The map below is **alive**. Articles flip from `[wip]` to `[done]` as they ship, and each one usually ends by surfacing two or three new questions — those become the next branches. So the roadmap will keep growing sideways as much as forward.

## The map

```
01  Tensor parallelism, built from scratch in your head        [done]
    → /posts/01-tensor-parallelism-mental-model/
    │
    └─ opens: "ok, but what about stacking layers across GPUs?"
       │
       ▼
02  Pipeline parallelism in a vanilla attn + FFN block         [wip]
    walking through Megatron's interleaving of column-wise
    and row-wise TP across the block
    │
    ├─ opens: "what if FFN is sparse?"
    │  │
    │  ▼
    │  03  MoE — what changes when experts replace FFN          [planned]
    │
    └─ opens: "what if the batch has uneven sequence lengths?"
       │
       ▼
       04  Continuous batching & variable seq lengths           [planned]
           and what that does to vanilla attention
```

## Status legend

`[done]` shipped & linked · `[wip]` actively drafting · `[planned]` next up
