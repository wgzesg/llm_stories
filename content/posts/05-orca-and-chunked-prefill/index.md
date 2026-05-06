---
title: "ORCA and Chunked Prefill: Evening Out the Iteration"
date: 2026-05-06T00:00:00+00:00
draft: false
summary: "Many requests, each finishing at a different time, and some carrying prefills 1000× the size of a decode step. Per-iteration cost swings wildly. ORCA-style iteration-level scheduling fixes one half; chunked prefill bounds the largest iteration so short work isn't dragged behind long work."
description: "How iteration-level scheduling (ORCA) and chunked prefill flatten per-iteration cost. We walk the cost variance, see what ORCA fixes and what it leaves open, and watch chunked prefill bound the longest iteration."
tags: ["orca", "continuous-batching", "chunked-prefill", "kv-cache", "llm-serving", "mental-model"]
series: ["llm-stories"]
showToc: true
TocOpen: false
weight: 6
---

[Article 04](/llm_stories/posts/04-batching-many-requests/) left us with one forward pass that batches many prefills cleanly. But prefill is just the front half of a request's life. Once the prompt is consumed, the request enters a decode phase — generating one token at a time, sometimes for hundreds of steps, until it lands on an EOS. A real serving engine doesn't see neat prefill batches; it sees a turbulent mix of arriving prompts, ongoing decodes, and finishing requests, all sharing the same GPU at every moment.

This article steps into that mess. We'll lean on [Article 01](/llm_stories/posts/01-llm-end-to-end/) for the basic generation flow and the KV cache mechanics — assume both are familiar.

(One name to fix in your head before we start, since the rest of the article leans on it: **an iteration is one end-to-end forward pass through all `L` layers of the model.** Whatever rows of input we feed in — a chunk of one prompt, decode steps for many requests, or a mix — an iteration runs them through layer 0, layer 1, all the way to layer `L−1`, once.)

Picture a few seconds in the engine's life. Dozens of requests in flight: some still munching prompts, some 50 tokens into decoding, some about to terminate at step 1000, some that just walked in. New requests arrive. Old requests finish. The scheduler's job is to keep the GPU as full as it can while doing right by every one of them.

Two questions fall out:

1. **Requests arrive and finish at different times.** How does the engine keep the GPU packed without stalling anyone at the start or end of their life?
2. **Per-iteration cost can swing 1000×.** A forward of pure decodes runs in a handful of milliseconds; a forward that includes a 100 k-token prefill takes seconds. How do we keep iterations roughly uniform so the scheduler can plan?

Both answers — iteration-level scheduling (ORCA) and chunked prefill — pull on the same insight: **what we want to even out is the per-iteration cost, not the per-request cost.** ORCA cleans up the arrival/finish boundaries; chunked prefill then bounds what any single iteration can carry.

---

## 1. Where naive batching breaks down

The simplest scheduler imaginable: pick `B` requests when slots open, run them through prefill and decode together, return everyone's outputs when the *last* one is done, then pick the next batch. This is *request-level* batching — the batch is the scheduling unit, and a batch's membership is fixed at admission.

It fights two facts about real traffic.

**Requests arrive at different times.** A request that arrives 200 ms into a 5-second batch can't join — the batch's membership was set when it started. It sits in the queue until the entire current batch finishes. The GPU might have plenty of room for one more decoder, but the engine refuses to admit anyone. The arrival's **TTFT** (*time-to-first-token* — the pause between hitting enter and seeing ChatGPT's first word appear) inflates from a few milliseconds into seconds, *just from waiting*. This is the **convoy effect**: arrivals get queued behind the slowest member of whatever happens to be running.

**Requests finish at different times.** Inside a batch, request A might want 50 output tokens and request B might want 1000. Both are decoded together. After step 50, A is done — but its slot can't be reclaimed for someone else, because the batch's shape is frozen until everyone's finished. A's compute slot sits idle for the next ~5 seconds of B's continued decoding. Worse, A's tokens — already produced and ready — can't return to the user until the batch boundary either. This is the **frozen batch size** problem: short-lived requests pay the longest peer's lifespan twice over, once in stalled return and once in wasted GPU.

Both failures come from one root cause: **a static batch has one shared lifespan, set by `max` over its members.** Anyone shorter than the max wastes; anyone arriving after the start waits.

The scheduler is glued to the wrong granularity. Reality moves at the granularity of *iterations* — every forward pass produces a token (or a chunk of prefill) for each in-flight request. The scheduler is making decisions at the granularity of *batches* — once per several thousand iterations. Of course it can't keep up.

---

## 2. ORCA: schedule per iteration, not per batch

The fix from the [ORCA paper](https://www.usenix.org/system/files/osdi22-yu.pdf) is small to state and large in consequence: treat the **iteration** — one end-to-end forward through all `L` layers — as the scheduling unit. The set of in-flight requests becomes a living thing the scheduler curates between every forward, instead of a fixed roster set at admission.

Between iterations, the scheduler can:

- **Drop** any request that produced EOS in the last iteration. Its slot is free immediately.
- **Add** a new request from the queue. On its first iteration, it contributes its prompt rows for prefill.
- **Carry** mid-decode requests forward, each contributing exactly one Q row this iteration.

All three operations are pure scheduler bookkeeping — no GPU work, just updates to per-request metadata. They run between iterations on the host, while the GPU is busy with the previous forward.

What this means for one iteration's *contents* is a step up in flexibility from Article 04. Article 04's iterations were homogeneous — every request was prefilling, every request contributed prompt rows. Under ORCA, an iteration carries requests at different stages at the same time. Concretely:

| Request | State | Q rows this iter | kv_length |
|---|---|---:|---:|
| A | first-iter prefill, 4096-token prompt | 4096 | 4096 |
| B | mid-decode, step 51 | 1 | 1500 |
| C | mid-decode, step 200 | 1 | 1700 |

Total Q rows in this iteration: `4096 + 1 + 1 = 4098`. The varlen kernel walks the flat tensor request-by-request and computes three independent score blocks: A's `4096 × 4096` (lower-triangular — A is prefilling its own tokens), B's `1 × 1500`, C's `1 × 1700`. Each request reads only *its own* KV cache — no cross-request bleed.

To support this mix, Article 04's `cu_seqlens` (which only tracked Q-row boundaries) generalizes to one tuple per request:

```
(q_start, q_end, kv_length)   per request
```

`q_rows = q_end - q_start` is what this request contributes to *this* iteration's Q. `kv_length` is the request's full attention context after this iteration's K, V appends — which now includes any prior cache. The number of Q rows and the KV length are no longer forced to match — a decoder has `q_rows = 1` and `kv_length = 1500`, a fresh prefill has both equal at 4096.

That's the only kernel-side change. ORCA's contribution wasn't a new attention kernel — it was a **scheduling discipline**: don't run a batch to completion, choose membership at every iteration. The kernel work was already in place from Article 04; what was missing was the policy of using it iteration-by-iteration.

This is what modern serving systems mean by **continuous batching**.

What it gives us:

- *Convoy effect* dissolved — new arrivals join at the next iteration; the wait is one iteration (a few ms), not one batch (seconds).
- *Frozen batch size* dissolved — a slot freed at iteration `t` is filled at iteration `t+1`; a finished request returns its output as soon as its EOS is sampled, not at some far-off batch boundary.

Both problems gone. A clean win. So clean, in fact, that it's tempting to declare scheduling solved and move on. But there's something we glossed over.

---

## 3. The next problem: iterations themselves vary wildly

ORCA fixed the boundary problems — arrival and finish — by making the iteration the scheduling unit. But making the iteration the scheduling unit also makes it the **heartbeat of the engine**. Every in-flight request, decoder or prefiller, gets one bit of work done per iteration. So if iteration `t` takes 6 ms and iteration `t+1` takes 8 seconds, the gap between consecutive tokens for *any* in-flight decoder is 8 seconds. An iteration's wall time is no longer a private detail of how the GPU spends its compute; it's the latency floor for everyone in the engine that iteration.

So how variable is iteration wall time, really? Anchor on **Llama-2-7B** running on a single H100 and plug a few realistic iteration mixes through the cost model.

<details>
<summary><em>FLOP and wall-time formulas used (click to expand)</em></summary>

Llama-2-7B: multi-head attention, 32 layers, hidden 4096, head dim 128. Forward cost has two structurally different terms.

- **Linears**: forward cost per processed token-row is roughly `2P` FLOPs where `P ≈ 7×10⁹` — about **14 GFLOPs per token-row** that flows through the model.
- **Attention**: per (q,k) pair across the whole network ≈ `4 · d_head · heads · layers = 4·128·32·32` ≈ **0.52 MFLOPs per pair**. For a length-`L` prefill with causal mask: ~`L²/2` pairs ≈ `2.6×10⁵ · L²` FLOPs total. For a single decode step against a cache of size `M`: `M` pairs ≈ `0.52·M` MFLOPs.

H100 effective: ~500 TFLOPs/s fp16 for compute-bound work, ~3.35 TB/s HBM bandwidth for read-bound work. A decode step's main cost is *reading the weights once* across the whole network (~14 GB at fp16), not the FLOPs themselves — so decode is bandwidth-bound at about **5–7 ms per step**, set by `bytes ÷ HBM`.

</details>

Take a baseline of 8 in-flight decoders at ~1 k context each. Vary what one new request brings into the same iteration:

| Iteration mix (8 decodes + …) | Linear | Attn | Total | Wall time |
|---|---:|---:|---:|---:|
| nothing else (decodes only) | ~110 GF | ~4 GF | ~115 GF | **~6 ms** (bw-bound on weights) |
| + 1 k-token prefill | ~14 TF | ~0.3 TF | ~14 TF | **~30 ms** |
| + 4 k-token prefill | ~57 TF | ~4 TF | ~61 TF | **~120 ms** |
| + 16 k-token prefill | ~225 TF | ~67 TF | ~290 TF | **~580 ms** |
| + 100 k-token prefill | ~1.4 PF | ~2.6 PF | ~4 PF | **~8 s** |

Three patterns to notice:

- **Linears scale linearly** with the iteration's total token count.
- **Attention scales quadratically** in any single request's prefill length — negligible at small sizes, starts dominating around 100 k.
- **Wall-time swing** across iterations the scheduler might legitimately assemble is **roughly 1300×**.

That last number is what breaks the engine. To feel what it means: imagine you're using ChatGPT, your next paragraph streaming smoothly at ~150 tokens per second, and then — for no reason visible to you — the model freezes on a half-finished word for **eight seconds** before resuming. Nothing changed about your conversation. What happened, somewhere upstream, is that a different user pasted a 100 k-token document into their session, and your decode iteration got bundled into the same forward as their prefill. ORCA was happy to assemble that iteration — both were valid pieces of work — but the wall time was set by their prefill, and you paid for it.

Two flavors of this head-of-line blocking fall out, both inside a single iteration, not across batches.

### 3.1 TBT spike for in-flight decodes

The scenario above has a name: **TBT** (*time-between-tokens*) is the wait between consecutive output tokens for a decoding request — the steady-streaming feel a user expects. The bundled-with-a-100k-prefill iteration spikes TBT **~1300×** for every in-flight decoder that happens to share it.

A static batch wouldn't have done this — but a static batch had its own catastrophes. ORCA didn't break anything; it just made an existing variability *visible* at the iteration level, where it now hits everyone in the engine simultaneously.

### 3.2 TTFT spike for short prefills batched with long ones

Two new requests arrive in the same iteration: one with a 100-token prompt, one with a 10 k-token prompt. ORCA happily packs both into one forward — they both want prefill, no in-flight state to mind, and stuffing more into one iteration is exactly what the kernel is built for. But the forward's wall time is set by the long peer:

| Forward content | Linear | Attn | Wall time |
|---|---:|---:|---:|
| 100-token prefill alone | ~1.4 GF | ~3 MF | **~4 ms** |
| 10 k-token prefill alone | ~140 TF | ~26 TF | **~320 ms** |
| 100 + 10 k packed together | ~141 TF | ~26 TF | **~330 ms** |

The short request's TTFT degrades from ~4 ms (alone) to ~330 ms (batched with the long peer) — **~80× worse**, purely because they shared a forward. From the short request's perspective, the network was operating at full speed for everyone except them, and there's no reason in *their* request for the latency they're paying. It's structural — a side effect of the iteration's wall time being set by its largest member.

### 3.3 Same root cause

Both 3.1 and 3.2 come from one structural fact: **an iteration's wall time is set by its largest piece of work.** ORCA can decide *whether* a piece is in this iteration, but not *how big* a piece is. Until the largest piece is bounded, the iteration heartbeat skips.

To get a stable heartbeat back, we need to bound the largest piece. That's what chunked prefill does — and the KV cache already gives us the tool to do it.

---

## 4. Chunked prefill: cap the largest piece

If a long prefill is the problem, what stops us from just *splitting* it?

Nothing structural, as it turns out — the KV cache makes splitting trivial. Once chunk 0 has run, its K and V at every layer are already stored in the cache. Chunk 1's attention can read them just like a decode step would. The math is identical to running the whole prefill at once, by construction; the only difference is *when* the work happens.

So: split a long prompt into chunks of size `C`, and carry one chunk per iteration. Walk through what happens for a request prefilling a prompt of length `N`:

- The prompt becomes `⌈N/C⌉` iterations for that request.
- Iteration 0 prefills tokens `[0, C)`. Its attention is exactly Article 04's prefill — `[C × C]` lower-triangular score block. K and V get stored in the cache at every layer.
- Iteration 1 prefills tokens `[C, 2C)`. Its attention now has Q rows from the new chunk and K, V rows from *both* the cached prefix and this chunk's freshly projected K, V. Score block: `[C × 2C]`.
- ...
- Iteration `k` prefills tokens `[kC, (k+1)C)`. Score block: `[C × (k+1)C]`.

The mask on chunk `k` has two regions:

- The block attending to the **cached prefix** `[C × kC]` is fully unmasked. Every prefix token was emitted before any token in this chunk, so causality permits attending to all of them.
- The block attending to **this chunk's own tokens** `[C × C]` is lower-triangular — causal within the chunk.

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

Walk the block on a chunk of size `C`, with prefix size `S = kC`:

| Step | What it does | Touches cache? |
|---|---|---|
| LayerNorm | per-row | no |
| QKV proj | matmul on `[C × hidden]` → Q, K, V each `[C × heads × d_head]` | no |
| Append K, V to cache | concat layer's K, V into the per-request cache | yes (write) |
| Attention | Q `[C × heads × d_head]`, K and V `[(S+C) × heads × d_head]`. Scores `[C × (S+C)]` with mask above. | yes (read full prefix) |
| Output proj | matmul | no |
| Residual + LayerNorm + FFN-up + GeLU + FFN-down + Residual | per-row | no |

The only step that changes versus Article 04 is **attention**, with shapes generalized:

- Q row count is `C` instead of "the request's whole length".
- K, V row count is `S + C` instead of equal to Q — the prefix now lives in the cache.
- Score block is rectangular `[C × (S+C)]`, not square.

Linears, residuals, layernorms, and pointwise ops are per-token and don't notice the cache. They process `[C × hidden]` rows row-by-row, indistinguishable from any other batch of `C` tokens.

Worth pausing on the score block's shape: it's a hybrid. The left part — chunk's queries against cached prefix — looks exactly like the score block of `C` decode steps stacked together: full unmasked attention over all prior tokens. The right part — chunk against itself — is a normal causal prefill `[C × C]` block. **Decode and prefill are just two extremes of the same shape, and chunked prefill is any point along the spectrum.**

Which makes the punchline obvious in retrospect: **decode is just `C = 1` chunked prefill.** Same machinery, different value of one knob.

---

## 5. Piggyback: prefill chunks coexist with decodes

Here's where everything composes. The flat-tensor + varlen kernel from Article 04 doesn't care what kind of work each request's slice represents. To the kernel, a request slice is just `(q_rows, kv_length)` — same shape whether the request is decoding (q_rows = 1), prefilling its first chunk (q_rows = C, kv_length = C), or carrying a middle chunk (q_rows = C, kv_length = S + C).

So a single iteration can carry, all packed into one flat tensor:

```
Iteration content:
  - Request E: prefill chunk 7 of 50    →  1024 Q rows,  kv_length = 8 × 1024 = 8192
  - Request A: decode step 51           →     1 Q row,   kv_length = 1500
  - Request B: decode step 200          →     1 Q row,   kv_length = 1700
  - Request C: decode step 75           →     1 Q row,   kv_length = 1100

Total Q rows in this iteration: 1024 + 3 = 1027
```

The varlen kernel walks each request's slice independently. TP is still untouched.

This is **piggyback chunked prefill**: long prefills coexist with in-flight decodes inside one forward. The scheduler's job becomes a kind of bin-packing — at every iteration, fill a budget (say "no more than 2048 token-rows of Q, no iteration longer than 50 ms") with whatever mix of decode steps and prefill chunks fits. A long prompt becomes a stream of chunk-sized contributions, one per iteration, alongside whatever decodes are running. Short prefills fit in single iterations. Decodes always fit. The 1300× swing from §3 collapses into a stable iteration profile of maybe 2–3× — easily plannable, and the engine's heartbeat is steady again.

`C` is the new scheduler knob:

- **Smaller `C`** → more uniform iteration time, lower TBT for in-flight decodes; but more cache re-reads per chunk and lower MFU on the linears (small GEMMs run further below peak).
- **Larger `C`** → fewer cache re-reads, higher MFU; but iteration wall time creeps back up and TBT degrades for everyone else in the iteration.

Real systems pick `C` in the **256–8192** range, usually tied to a "max batched tokens per iteration" budget that targets a TBT ceiling. Concretely: under a budget of "≤ 50 ms per iteration, up to 2048 Q rows," a 100 k-token prompt prefills in `100 000 / 2048 ≈ 49` iterations, sharing each one with whatever decodes are currently running.

---

## 6. Cost intuitions

Three things worth pausing on, since they all bite.

**Total compute is preserved.** Sum over chunks `k = 0 … N/C − 1` of `C · (k+1)C` causal pairs equals `N²/2`. Chunked prefill *redistributes* attention work across iterations; it doesn't reduce it.

**HBM bandwidth on KV reads grows.** Chunk `k` re-reads `kC` rows of cache per layer per attention call. Summed over all chunks: ≈ `N²/(2C)` rows of cumulative cache traffic, vs ~`N` rows for an unchunked prefill (which streams the cache through tiled attention exactly once). For `N = 100 k` and `C = 2048`, that's about **25× more cumulative cache-read bandwidth** spent on the same prompt — the price chunking pays for keeping iterations bounded. It's also why `C` can't be made arbitrarily small: at some point the bandwidth tax overtakes the schedulability win.

**Per-iteration MFU dips at small `C`.** Small-`C` iterations run their linear matmuls below peak — fewer rows for the tensor cores to chew on. Real serving engines tune `C` to a sweet spot where iteration time meets the TBT target without leaving too much MFU on the table.

The three together explain the typical `C ∈ [256, 8192]` band. There's no single right answer; the band depends on the model's compute/bandwidth profile and the engine's TBT/throughput targets.

---

## 7. What this opens

A real serving loop now: prefill, decode, mixed iterations, bounded per-iteration cost, no idle slots. Some assumptions still leak, each seeding the next round of articles.

- **The KV cache's physical layout.** We've quietly assumed each request's cache is a contiguous slab per layer. As `B` grows and contexts vary, this gets ugly fast — fragmentation, eviction, allocation overhead. **PagedAttention** treats the cache as virtual memory; the next article.
- **Two regimes sharing one engine.** Decode is bandwidth-bound on weight reads; prefill chunks are compute-bound. Maybe they shouldn't share the same GPUs at all. **Prefill/decode disaggregation** explores running them on separate replicas.
- **Heads aren't always independent.** GQA, MLA, and the rest of the "fewer KV heads" family shrink the cache dramatically — bigger batches, longer contexts — but introduce sharing patterns we've been able to ignore so far. A whole sub-series.
- **One request's cache outgrows one GPU.** Once a context gets long enough that its KV cache alone won't fit on a single card, sequence/context parallelism splits *one request* across GPUs. Its own article, much later.

Same grammar each time: relax one assumption, see what falls out.
