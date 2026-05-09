---
title: "Roadmap"
date: 2026-04-29T00:00:00+00:00
draft: false
summary: "What this series is, and a living map of the articles — shipped, in progress, and the holes we've dug for our future selves to fill."
description: "A living roadmap of the LLM Stories series. Tracks shipped articles and the questions each one opens up next."
tags: ["meta"]
series: ["llm-stories"]
showToc: false
weight: 1
---

## About this series

These are learning notes — me working through how modern LLMs are actually served, mostly by talking to Claude and writing up the parts that finally clicked. The articles themselves are written in a confident "discovery journey" voice, but the project underneath is just someone learning in public.

The list below is **alive** — articles flip status as they ship, and the roadmap grows whenever a discussion surfaces a hole worth digging.

---

## Articles

| # | Title | Status | Link |
|---|---|---|---|
| 01 | An LLM, end to end — bird's-eye stack, one block, the generation loop, and the questions the rest of the series picks up | `[next]` | — |
| 02 | Tensor parallelism, built from scratch in your head | `[done]` | [read →](/llm_stories/posts/02-tensor-parallelism-mental-model/) |
| 03 | Walking TP through a full block — start column-parallel everywhere, watch the comm explode, pair with row-parallel until two all-reduces per block fall out | `[done]` | [read →](/llm_stories/posts/03-tp-through-a-full-block/) |
| 04 | How to batch many requests through one forward pass — varlen attention, prefill only, TP turns out to be untouched | `[done]` | [read →](/llm_stories/posts/04-batching-many-requests/) |
| 05 | ORCA and chunked prefill — iteration-level scheduling solves the boundary problems; chunked prefill bounds the iteration so a long prompt can't hijack the engine's heartbeat | `[done]` | [read →](/llm_stories/posts/05-orca-and-chunked-prefill/) |
| 06 | Prefill and decode disaggregation — two phases on opposite sides of the roofline; once you accept the asymmetry, sharing a GPU pool is no longer a compromise but a fight against the formula | `[done]` | [read →](/llm_stories/posts/06-prefill-decode-disaggregation/) |
| 07 | The engineering of disaggregation — KV cache transfer across fabrics (NVLink, NVSwitch, IB, PCIe), tiered memory pools (HBM, DRAM, SSD), overlap with prefill, topology-aware routing | `[next]` | — |
| 08 | Pipeline parallelism — the cut *across* blocks instead of within one, and the bubble it creates; why the prefill pool wants it | `[planned]` | — |
| 09 | MoE and expert parallelism — what changes when FFN becomes routed | `[planned]` | — |
| 10 | PagedAttention — the KV cache as virtual memory, blocks instead of contiguous slabs, copy-on-write across requests | `[planned]` | — |
| 11 | Sequence and context parallelism — splitting one *request* across GPUs, ring attention, the long-context move | `[planned]` | — |
| 12 | FlashAttention — tiled online softmax, why the `[L × L]` score matrix never has to exist | `[speculative]` | — |
| 13 | FlashDecoding — making the `1 × L_kv` decode-attention call fast under bandwidth pressure | `[speculative]` | — |
| 14 | GQA and MLA — fewer KV heads, smaller KV cache, faster decode (and what it costs the model) | `[speculative]` | — |
| 15 | Speculative decoding — a draft model proposes, the big model verifies, two passes for the price of one | `[speculative]` | — |
| 16 | KV compression — quantization, eviction policies, what we can drop and what we can't | `[speculative]` | — |

## Status legend

`[done]` shipped & linked · `[next]` actively drafting · `[planned]` on deck, will get there · `[speculative]` a hole worth digging — may or may not get filled, but the question is interesting

---

## Recurring threads worth flagging

A few observations that keep showing up across articles, worth keeping in the back of your mind as you read:

- **TP turns out to be remarkably non-disruptive.** Request batching didn't disturb it (Article 04), and continuous batching + chunked prefill didn't either (Article 05). PP and MoE *do* interact with TP in interesting ways — that's why those come up next.
- **The KV cache is the connective tissue** between articles 05 onward. It enters with decode and never really leaves; it's also the thing that makes long contexts hard.
- **Decode flips the bottleneck profile.** Articles 02–04 assume prefill, where compute dominates. Once decode is in scope (Article 05 onward), bandwidth on weight reads becomes the binding constraint — and that's what motivates almost every later optimization (FlashDecoding, GQA, prefill/decode disaggregation, speculative decoding).
- **Modelers' choices keep load-bearing for serving** in ways that weren't designed in. Multi-head independence made TP comm-free; it also made request batching comm-free; it'll show up again when we look at GQA/MLA. Worth tracking as a recurring theme.
