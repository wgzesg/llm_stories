---
title: "Prefill and Decode Disaggregation: Two Phases on Opposite Sides of the Roofline"
date: 2026-05-09T00:00:00+00:00
draft: false
summary: "Article 05 left two phases politely sharing one engine. This article shows they shouldn't — prefill is compute-bound, decode is bandwidth-bound, and long context drives the gap wider, not smaller. Once we accept the asymmetry, splitting them is the structural fix."
description: "A roofline-first argument for prefill/decode disaggregation. Defines arithmetic intensity, derives that intensity ≈ tokens-per-iteration for transformers, sweeps context length to show decode falling further below the ridge as L grows, then walks through the split and the KV-transfer cost it introduces."
tags: ["pd-disaggregation", "prefill", "decode", "roofline", "arithmetic-intensity", "kv-cache", "llm-serving", "mental-model"]
series: ["llm-stories"]
showToc: true
TocOpen: false
weight: 7
---

[Article 05](/llm_stories/posts/05-orca-and-chunked-prefill/) ended with a smooth heartbeat. ORCA fixed the boundary problems by scheduling per iteration; chunked prefill capped the iteration so a long prompt couldn't hijack the room. Every iteration is bounded, every request is roughly fair, the engine breathes evenly.

But that article also left a thread dangling, in §7's second bullet:

> *Decode is bandwidth-bound on weight reads; prefill chunks are compute-bound. Maybe they shouldn't share the same GPUs at all.*

This article pulls on that thread. We'll measure the gap with the roofline model, watch it widen as context length grows, and end with the structural fix: stop putting the two phases on the same machine.

The starting point is uncomfortable: piggyback chunked prefill from Article 05 isn't a *solution* to the prefill/decode mismatch — it's a *compromise*. It flattens the heartbeat, but the underlying truth is that a prefill chunk and a decode token want the GPU to be in two different regimes. Sharing forces both to settle for the wrong one.

---

## 1. The roofline, in one page

Every kernel on every GPU is bottlenecked on one of two physical resources:

- **Compute** — the tensor cores' peak FLOPs/s.
- **Memory bandwidth** — the rate at which HBM can deliver bytes to the SMs.

(This is intra-GPU bandwidth, the wire from HBM to the tensor cores. Inter-GPU bandwidth — NVLink, InfiniBand — is a separate axis we'll meet later when TP and PP enter the story.)

### A concrete picture of where bytes live

The "memory bandwidth" number is opaque without a picture of the chip. A modern GPU has a **memory hierarchy** — several layers, each smaller and faster than the one below it. Tensor cores can only do math on data sitting in registers, so every byte of weight or KV cache has to traverse the hierarchy *up* before any compute happens.

<svg viewBox="0 0 720 540" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;font-family:system-ui,sans-serif;display:block;margin:1.5rem auto">
  <text x="360" y="24" text-anchor="middle" font-size="14" fill="currentColor" font-weight="600">GPU memory hierarchy (H100-flavored numbers)</text>
  <rect x="40" y="50" width="640" height="290" fill="none" stroke="currentColor" stroke-width="1.5" stroke-opacity="0.6" rx="6"/>
  <text x="60" y="74" font-size="13" fill="currentColor" opacity="0.75" font-weight="600">GPU die</text>
  <g>
    <rect x="70" y="95" width="150" height="120" fill="rgba(74,144,226,0.12)" stroke="#4a90e2" stroke-width="1.5" rx="4"/>
    <text x="145" y="115" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">SM 0</text>
    <text x="145" y="138" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.85">registers</text>
    <text x="145" y="158" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.85">SRAM (~256 KB)</text>
    <text x="145" y="178" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.85">tensor cores</text>
    <text x="145" y="202" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.6" font-style="italic">~30 TB/s effective</text>
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
  <text x="360" y="287" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.85">~50 MB shared</text>
  <text x="360" y="304" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.6" font-style="italic">~5 TB/s</text>
  <line x1="360" y1="340" x2="360" y2="395" stroke="currentColor" stroke-width="2"/>
  <polygon points="355,395 365,395 360,408" fill="currentColor"/>
  <text x="375" y="375" font-size="12" fill="currentColor" font-weight="600">3.35 TB/s</text>
  <text x="375" y="392" font-size="10" fill="currentColor" opacity="0.65" font-style="italic">HBM bandwidth</text>
  <rect x="120" y="415" width="480" height="100" fill="rgba(126,211,33,0.15)" stroke="#7ed321" stroke-width="1.5" rx="4"/>
  <text x="360" y="440" text-anchor="middle" font-size="13" fill="currentColor" font-weight="600">HBM — 80 GB</text>
  <text x="360" y="463" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.85">model weights · KV cache · activations between kernels</text>
  <text x="360" y="490" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.6" font-style="italic">large enough to hold the model and the batch's state, but the slowest tier</text>
</svg>

Numbers are H100-flavored; other GPUs differ in absolute terms but the *shape* — three to four orders of magnitude between top and bottom in both capacity and speed — is universal.

What's stored where:

- **HBM** holds the persistent stuff: model weights (14 GB for Llama-2-7B), every request's KV cache, and activations that survive between kernel launches. Big and slow-relative.
- **L2 cache** is a shared scratch — useful when many SMs read overlapping data, but it's only ~50 MB, far too small to hold weights or KV.
- **SRAM (per-SM shared memory)** is where a kernel stages the *current tile* of weights, queries, and keys it's working on. FlashAttention's whole trick is keeping the attention score matrix in SRAM so it never spills to HBM.
- **Registers** are where tensor cores actually read operands from. A few hundred KB per SM, accessible in a single cycle.

So when you read "the kernel loaded 14 GB of weights from HBM," the path is: HBM → L2 → SRAM → registers → tensor cores. Each layer is smaller and faster than the one below it, and **the 3.35 TB/s number is the *bottom* of that chain** — the one bottleneck that can't be cached around for a transformer iteration, because the weights are larger than every layer above HBM.

### What "compute-bound" vs "bandwidth-bound" actually means physically

A matrix multiply works in **tiles**: load a tile of A and a tile of B from HBM into SRAM, multiply them in registers (many FLOPs per element), accumulate, move on. The same tile of weights is reused across many output rows before being evicted.

- **Compute-bound** means the tensor cores are saturated. They consume the current tile fast enough that HBM can comfortably deliver the next tile in the background. Bandwidth has slack. *Each byte of weight, once loaded, is reused for many FLOPs.*
- **Bandwidth-bound** means HBM can't deliver the next tile fast enough. The tensor cores have already finished with the current one and sit idle waiting for bytes. *Each byte is reused for too few FLOPs to amortize the load.*

The number that decides which regime you're in is exactly **how many FLOPs you do per byte you pulled from HBM** — that's the intensity, and that's why the roofline rule is so unforgiving. It isn't an empirical observation; it's a direct consequence of the hierarchy above.

### The roofline rule

Which resource binds is decided by a single number: **arithmetic intensity** `I`, the ratio of FLOPs done to bytes loaded from HBM:

```
I  =  FLOPs done / bytes loaded     (units: FLOPs/byte)
```

The hardware has a matching number, the **ridge point** `R`:

```
R  =  peak FLOPs/s / peak HBM bandwidth   (units: FLOPs/byte)
```

For an H100 SXM5: ~500 TFLOPs/s sustained fp16 GEMM, 3.35 TB/s HBM3 → **R ≈ 150 FLOPs/byte**.

The rule:

- `I > R` → **compute-bound**. The arithmetic dominates; bandwidth has slack.
- `I < R` → **bandwidth-bound**. The bytes dominate; tensor cores idle waiting for data.

That's it. The whole rest of this article is two questions:

1. What's `I` for a prefill iteration vs. a decode iteration?
2. How does `I` change as context length grows?

---

## 2. Notation and the iteration cost model

Before any numbers, fix symbols. We'll assume **fp16 throughout** (2 bytes per parameter, 2 bytes per cached number). Lower-precision dtypes change the numerics but not the story.

| Symbol | Meaning | Units |
|---|---|---|
| `Π` | parameter count | dimensionless |
| `K_tok` | KV bytes stored per token (sum over all layers, both K and V) | bytes/token |
| `T` | total tokens in this iteration | tokens |
| `B` | in-flight requests in this iteration | dimensionless |
| `L` | average context length per request | tokens |
| `C` | prefill chunk size (new tokens per chunk) | tokens |
| `R` | hardware ridge point | FLOPs/byte |

(`K_tok` is the sum across all layers — what *one token of context* costs across the full network's KV cache, not per-layer.)

For one transformer iteration on a model of size `Π`, two physical quantities matter, and both are **linear in `Π`**:

- **Bytes pulled from HBM for weights:** every parameter is 2 bytes wide and the iteration reads each one once → `2Π` bytes. For Llama-2-7B (`Π = 7B`), 14 GB. Paid once per iteration, no matter how many tokens we packed in.
- **FLOPs to push one token through the network:** each token's pass through the model multiplies by every parameter once (2 FLOPs per multiply-accumulate) → `2Π` FLOPs/token. For Llama-2-7B, 14 GFLOPs/token. An iteration processing `T` tokens does `2Π · T` FLOPs — tokens don't interact in the matmul layers (only in attention), so each one costs the same `2Π` independently.

Add the KV-cache reads to the byte side and write the two together:

```
bytes_loaded  =  2Π                  (weights, paid once per iteration)
              +  K_tok · L · B       (KV cache, each request reads its own L rows)

FLOPs_done    =  2Π · T              (T = tokens in this iteration)
```

Plug into the intensity definition:

```
I  =  2Π · T  /  (2Π + K_tok · L · B)
```

Stare at this formula for a moment — the rest of this section is reading it carefully. The denominator has two terms, the numerator has one, and walking through them in sequence gives us the whole prefill/decode story.

### Part 1: pretend the KV term is zero

At very short `L`, or before any context has accumulated, the denominator is dominated by `2Π` and the formula collapses to:

```
I ≈ T
```

Intensity is literally **the number of tokens sharing one weight load.** This is where prefill and decode part ways:

- **Prefill iteration**: `T = C = 2048` tokens → `I ≈ 2000` → way above any modern ridge (~150) → **compute-bound**.
- **Decode iteration**: `T = B` (concurrent decoding requests, typically tens to low hundreds) → `I ≈ B` → way below ridge → **bandwidth-bound**.

Same hardware, same model, same kernel. The only difference is how many tokens the iteration is carrying. Prefill amortizes the weight load over thousands of tokens; decode amortizes it over `B`. They land on opposite sides of the ridge from the very first iteration — and not by a small margin: an order of magnitude or more in intensity.

The instinct is to fix this by batching decode harder — push `B` up until intensity clears the ridge. To clear `R = 150` you'd need `B ≥ 150`. The next part of the formula explains why that's not feasible.

### Part 2: turn the KV term back on

As context grows, `K_tok · L · B` adds to the denominator. The two denominator terms cross when:

```
L · B = 2Π / K_tok
```

For Llama-2-7B (`Π = 7B`, `K_tok ≈ 512 KB`), `L · B ≈ 27 k`. At decode batch `B = 32`, the crossover is at `L ≈ 850` tokens.

That number — 850 — is tiny by today's standards, and it's worth pausing on. Production prompts routinely run **tens of thousands** of tokens: large system prompts and tool definitions, RAG-injected documents, accumulated multi-turn conversations, agentic chains where input-to-output ratios commonly run 100:1 or higher. Frontier models ship with 200 k – 2 M context windows precisely because real workloads fill them. So "past the crossover" isn't a corner case — it's the median request.

Past the crossover, the formula approximates in the *other* direction:

```
I ≈ 2Π · T / (K_tok · L · B)
```

And here the cancellations matter:

- **Decode** (`T = B`): `I ≈ 2Π / (K_tok · L)`. The `B`s cancel — *increasing the decode batch no longer raises intensity at long context.* You just pay proportional KV reads for processing more requests in parallel. And before `B` can grow much, you run out of KV memory. So the "just batch harder" instinct from Part 1 fails twice over.
- **Prefill** (`T = C`): `I ≈ 2Π · C / (K_tok · L · B)`. Nothing cancels — `C` stays in the numerator. Prefill stays compute-bound out to absurd contexts.

### Two facts from one formula

1. **Prefill is compute-bound; decode is bandwidth-bound.** This holds even at zero context, set entirely by how many tokens share one weight load. They're on opposite sides of the ridge from the start.
2. **Long context widens the gap.** A second bandwidth cost — KV reads — emerges in the denominator and dominates past the crossover (which production traffic routinely sits past). It lands disproportionately on decode, while leaving prefill mostly untouched.

§3 confirms both with concrete numbers on Llama-2-7B.

---

## 3. The numbers, one model, two phases

To put the formula on the ground, sweep `L` for a single model on a single GPU.

**Llama-2-7B (MHA, 32 layers, 32 heads, head_dim 128, fp16) on H100:**

- weight bytes `2Π = 14 GB`
- `K_tok = 2 (K,V) · 32 layers · 32 heads · 128 head_dim · 2 bytes ≈ 512 KB/token`
- ridge `R ≈ 150 FLOPs/byte`

### Decode at B = 32

| `L` | weight bytes | KV bytes | total | `I = 2Π·B / total` | regime |
|---:|---:|---:|---:|---:|---|
| 1 k | 14 GB | 16 GB | 30 GB | ~15 | bandwidth-bound (weights ≈ KV) |
| 4 k | 14 GB | 64 GB | 78 GB | ~5.7 | bandwidth-bound (KV dominates) |
| 16 k | 14 GB | 256 GB | 270 GB | ~1.7 | catastrophically bandwidth-bound |
| 64 k | 14 GB | 1.0 TB | 1.0 TB | ~0.4 | the cache doesn't even *fit* on one H100 |

(Numerator `2Π · B = 448 GFLOPs` — pinned. The denominator is what blows up.)

Notice:

- **Intensity falls fast.** From ~15 at L=1k to ~0.4 at L=64k — more than an order of magnitude over a single dimension of context.
- **Memory budget bites before bandwidth does.** At L=16k, B=32 the KV alone is 256 GB, way past the H100's 80 GB. PagedAttention exists partly to manage this, and `B` is forced *down* at long context, which makes intensity worse. (Llama-2-7B uses MHA; modern GQA/MLA models cut `K_tok` by 4–8×, mostly to push this wall back.)
- **The dominant byte changes.** At small L, weights dominate. At large L, KV dominates. Both are bandwidth-bound, but the fix is different — bigger batch helps with weight pressure; GQA/MLA/FlashDecoding help with KV pressure.

### Prefill at C = 2048

Chunked prefill processes `C` new tokens against a prefix of size `S` (so `T = C` tokens of compute, reads `S` tokens of cached KV):

```
I_prefill = 2Π · C / (2Π + K_tok · S)
```

The numerator scales with `C` — every byte loaded is reused across thousands of tokens of math.

| prefix `S` | weight bytes | KV bytes | total | `I` | regime |
|---:|---:|---:|---:|---:|---|
| 4 k | 14 GB | 2 GB | 16 GB | ~1800 | compute-bound (×12 above ridge) |
| 64 k | 14 GB | 32 GB | 46 GB | ~620 | compute-bound (×4) |
| 256 k | 14 GB | 128 GB | 142 GB | ~200 | still compute-bound (×1.3) |
| 1 M | 14 GB | 512 GB | 526 GB | ~55 | finally below ridge — but we're at a million tokens |

Prefill stays compute-bound out to extreme contexts. Even where it crosses below the ridge, it's nowhere near as bandwidth-bound as decode is at *common* contexts.

The asymmetry, stated cleanly:

> **Each byte of bandwidth is amortized over `C ≈ 2000` tokens in prefill, but over 1 token per request in decode. Long context turns the screw on decode and barely touches prefill.**

Same model, same GPU. Two phases. Completely different fates.

---

## 4. Why one engine can't serve both well

Take the engine from article 05 — continuous batching, chunked prefill, piggyback iterations — and ask: how do you size it?

- **Size for prefill:** buy GPUs for FLOPs. Decode then runs on hardware where most of the compute is structurally unreachable, because decode is bandwidth-bound. You're paying for tensor cores decode physically can't use.
- **Size for decode:** buy fewer GPUs sized for HBM bandwidth and capacity. Prefill takes longer than it needs to. **TTFT** (time-to-first-token, the pause before the first word) inflates.
- **Mix:** every iteration packs prefill chunks and decode tokens. **TBT** (time-between-tokens, the cadence between words) is held hostage by however much compute the prefill chunks are eating. Chunked prefill bounds this — that was article 05's whole point — but it can't make the bound free. A decode iteration sharing the engine pays for `C` rows of prefill work that doesn't help any decode at all.

The deeper issue: **the workload's bottleneck profile is bimodal, but the engine is unimodal.** There's no single sizing, no single parallelism strategy, no single batch policy that's right for both phases simultaneously. The two phases stress different physical resources and have different SLOs (TTFT vs TBT), and one scheduler with one knob can't satisfy two SLOs against two regimes.

So you stop trying. You build two pools.

---

## 5. The split

<svg viewBox="0 0 760 260" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;font-family:system-ui,sans-serif;display:block;margin:1.5rem auto">
  <defs>
    <marker id="arr-split" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="currentColor"/>
    </marker>
  </defs>
  <text x="40" y="120" text-anchor="end" font-size="12" fill="currentColor" opacity="0.85">prompt</text>
  <line x1="48" y1="115" x2="100" y2="115" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-split)"/>
  <rect x="105" y="65" width="225" height="130" fill="rgba(74,144,226,0.15)" stroke="#4a90e2" stroke-width="1.5" rx="6"/>
  <text x="217" y="92" text-anchor="middle" font-size="14" fill="currentColor" font-weight="600">Prefill pool</text>
  <text x="217" y="120" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.85">compute-bound</text>
  <text x="217" y="140" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.85">optimizes TTFT</text>
  <text x="217" y="160" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.85">stateless</text>
  <line x1="332" y1="115" x2="428" y2="115" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-split)"/>
  <text x="380" y="100" text-anchor="middle" font-size="11" fill="currentColor" font-weight="600">KV cache transfer</text>
  <text x="380" y="138" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.65" font-style="italic">per request,</text>
  <text x="380" y="153" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.65" font-style="italic">L_p · K_tok bytes</text>
  <rect x="430" y="65" width="225" height="130" fill="rgba(245,166,35,0.18)" stroke="#f5a623" stroke-width="1.5" rx="6"/>
  <text x="542" y="92" text-anchor="middle" font-size="14" fill="currentColor" font-weight="600">Decode pool</text>
  <text x="542" y="120" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.85">bandwidth-bound</text>
  <text x="542" y="140" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.85">optimizes TBT</text>
  <text x="542" y="160" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.85">holds long-lived KV</text>
  <line x1="657" y1="115" x2="710" y2="115" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-split)"/>
  <text x="720" y="120" text-anchor="start" font-size="12" fill="currentColor" opacity="0.85">tokens</text>
</svg>

A request's lifecycle now has a hop in the middle:

1. **Prefill pool** receives the prompt, runs chunked prefill across all `L_p` tokens, produces the request's full KV cache plus the first generated token.
2. **KV cache transfer** ships those `L_p · K_tok` bytes from prefill GPU memory to a decode GPU's memory.
3. **Decode pool** receives the KV cache, slots the request into its continuous-batching pool, and runs decode iterations until EOS, streaming tokens back to the user.

Two pools, two scheduling regimes, two SLO targets. The compromise is gone. Each pool is now free to pick its own parallelism, batch policy, hardware mix, and scheduling discipline against a *single* objective. That freedom is most of the win — what each pool actually does with it is the subject of later articles in this series.

The handoff is the new cost. We'll price it in §6.

---

## 6. The new cost: KV cache transfer

Splitting the engines means KV moves between machines, once per request. That's a real cost — let's price it.

For Llama-2-7B (`K_tok ≈ 512 KB`) and a 4 k-token prompt:

```
KV bytes per request  =  L_p · K_tok  =  4096 · 512 KB  ≈  2 GB
```

That's per request. At a few hundred requests per second (modest production load), the *aggregate* east-west traffic between the two pools can run into hundreds of GB/s. Whichever fabric connects them needs to handle it.

What that fabric looks like, and what one transfer costs:

| Fabric | Bandwidth | 2 GB transfer time |
|---|---:|---:|
| NVLink (intra-node) | ~900 GB/s | ~2 ms |
| NVLink-network / NVSwitch fabric (cluster) | ~400 GB/s | ~5 ms |
| InfiniBand HDR (cross-node) | ~50 GB/s | ~40 ms |
| PCIe Gen5 (host-mediated) | ~64 GB/s | ~30 ms |

So the handoff is cheap if your pools are co-located in the same NVLink domain, and a real tax if they're across an IB hop. A 40 ms hit on TTFT is meaningful; a 5 ms hit is not.

A few engineering knobs that immediately fall out (each could justify its own article — we're surfacing, not solving):

- **Layer-streaming overlap.** Don't wait for prefill to finish to start the transfer. Each layer's K, V are produced in order; ship them while later layers are still computing. Done well, the transfer is mostly hidden behind prefill compute.
- **GPUDirect RDMA.** Move bytes directly between GPU HBMs without bouncing through CPU memory. Saves a copy and a context switch.
- **Topology awareness.** Schedule prefill and decode for the same request onto pools that are close — same rack, same NVLink domain — to minimize fabric class.
- **Prefix reuse.** If two requests share a long prefix, you only need to compute and transfer the suffix's KV. Production systems (Mooncake at Moonshot is a well-documented example) turn this into a memory-hierarchy problem: hot prefixes in HBM, warm in DRAM, cold on SSD.
- **GQA / MLA shrink the bill directly.** Cutting `K_tok` by 4–8× cuts the transfer by 4–8×. This isn't usually framed as a disaggregation optimization, but it is one.

There's a real article's worth of detail under each of those. For now the takeaway is just that **the transfer is the price of disaggregation**, and it's payable — bounded, well-engineered, and small relative to the wins on TTFT and TBT.

What the user feels:

- **TTFT** = prefill time + transfer time + first decode iter. Transfer is a real but small component (a few ms to tens of ms).
- **TBT** = pure decode, no prefill contention. The decode pool's iterations only ever contain decode work, so TBT is as smooth as the decode hardware alone can make it.

The trade is the one you want: a small one-time tax on TTFT in exchange for clean, predictable TBT throughout the generation. Users feel TBT far more than TTFT — TTFT is one pause, TBT is every pause.

---

## 7. What this opens

Article 05 ended by capping the iteration. Article 06 ends by splitting it. The formula in §2 forces the *why*; this article has spent most of its pages on that argument. The *how* is a different question, and §6 should be read as a doorway, not a destination — the visible tip of a much larger engineering surface.

Stand at that doorway for a second. Two GPUs, possibly in different racks, possibly under different memory tiers, have to move gigabytes of state per request fast enough to disappear behind prefill latency. Every choice in that pipeline has its own real design space:

- **Which fabric carries the bytes** — NVLink vs NVSwitch vs InfiniBand vs PCIe — sets a per-transfer cost that ranges across nearly two orders of magnitude (§6's table). The cluster topology you build looks completely different depending on the answer.
- **Where the KV cache lives between requests** — HBM vs DRAM vs SSD — turns the disaggregated engine into a tiered memory system. Mooncake-style prefix pools are one way; there are others, with different invalidation and locality behaviors.
- **How the transfer overlaps with compute** — layer-by-layer streaming, GPUDirect RDMA, double-buffered queues — is what makes the handoff invisible end-to-end vs. dominant in TTFT.
- **How requests are routed across pools** — fabric-locality-aware scheduling, prefix-cache hits, decode capacity tracking — is its own scheduling problem on top of everything in article 05.

Each of those is a real article on its own, and the next piece in this series picks up the thread — the engineering of running a disaggregated serving stack. *Then* we'll be in a position to ask the optimization questions disaggregation finally lets us ask cleanly: what does each pool *want*, now that it's free to specialize? Pipeline parallelism for prefill, tensor parallelism for decode, paging, GQA/MLA, FlashDecoding, speculative decoding — each has a clean home once the pools are split, and we'll work through them in turn.

Same grammar each time: name the bottleneck, factor the workload until each piece sees only the bottleneck that binds it, optimize per piece. Disaggregation was the biggest factoring move available. The next stretch of this series is the engineering and the optimization that the cuts unlock.
