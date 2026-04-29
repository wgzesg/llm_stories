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

The list below is **alive** — articles flip from `[wip]` to `[done]` as they ship, and the roadmap grows as new directions surface.

## Articles

| # | Title | Status | Link |
|---|---|---|---|
| 01 | Tensor parallelism, built from scratch in your head | `[done]` | [read →](/llm_stories/posts/01-tensor-parallelism-mental-model/) |
| 02 | Pipeline parallelism in a vanilla attn + FFN block — walking through Megatron's column/row-wise TP interleaving | `[wip]` | — |
| 03 | MoE — what changes when experts replace FFN | `[planned]` | — |
| 04 | Continuous batching & variable sequence lengths — and what that does to vanilla attention | `[planned]` | — |

## Status legend

`[done]` shipped & linked · `[wip]` actively drafting · `[planned]` next up
