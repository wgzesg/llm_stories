---
title: "Walking Tensor Parallelism Through a Full Block"
date: 2026-04-29T00:00:00+00:00
draft: false
summary: "Walk article 01's two cuts through a full transformer block, with concrete shapes on each GPU at every step. Apply one cut to every matmul first — comm explodes (four gathers per block). Then pair the two cuts as duals and watch them snap into the architecture's widen-narrow rhythm, landing at two all-reduces per block."
description: "How to split a full transformer block across two GPUs, with concrete shapes traced through every step. Start with column-parallel everywhere, see why it costs four gathers per block, then pair it with row-parallel to land at the Megatron pattern of two all-reduces per block."
tags: ["tensor-parallelism", "transformers", "llm-serving", "megatron", "mental-model"]
series: ["llm-stories"]
showToc: true
TocOpen: false
weight: 3
---

[Article 01](/llm_stories/posts/01-tensor-parallelism-mental-model/) left you with two ways to split **one** matmul across two GPUs. They're easier to keep straight by what they *do* than by what they're called in the literature, so let's lay them out side by side:

|                          | **Strategy A** — *split the `fx`es*                  | **Strategy B** — *split the rows*                            |
|--------------------------|--------------------------------------------------------|----------------------------------------------------------------|
| What you slice           | the matrix's columns (each column is one `fx`)         | the matrix's rows (each row is a basis vector)                 |
| Each GPU's **input**     | each token's **full** input vector                    | **half** of each token's input features                        |
| Each GPU's **output**    | **half** of each token's output features              | a **partial sum** of each token's *full* output                |
| How outputs combine      | **concatenate** (free)                                 | **all-reduce** (one comm step)                                 |
| Also known as            | column-parallel                                        | row-parallel                                                   |

The compact way to read each column: **A = "full in, half out."** **B = "half in, sum out."** That's enough mental model for everything below.

A real transformer block isn't one matmul — it's **four**, plus some pointwise glue. So the natural next question is: **how do we cut a *whole block* across two GPUs?**

There's an obvious first move that *almost* works. We'll build it, see exactly where it breaks, and let the fix walk us into the canonical Megatron pattern. To keep everything concrete, we'll fix small numbers and watch the shape on each GPU change at every step.

---

## 1. The setup: small numbers you can hold in your head

Two GPUs, call them **G1** and **G2**. A tiny batch and a small model:

| | value |
|---|---|
| **batch** `n` | 4 tokens |
| **model dim** `d` | 512 |
| **heads** `h` | 8 |
| **per-head dim** `d_head` | 64 |
| **attention dim** `k = h · d_head` | 512 |
| **FFN hidden** | `4d` = 2048 |

Each token is a row of 512 numbers. The batch is `[n × d] = [4 × 512]`.

A transformer block, drawn flat:

```
        ← input: [4 × 512]
    │
  LayerNorm
    │
  QKV projection      d → 3k       ← matmul   weight  [d × 3k] = [512 × 1536]
    │
  attention                          (mixes Qs with Ks; no new matmul)
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

**Four matmuls** and some glue.

> **Side note — the pointwise glue and why both GPUs do it.** LayerNorms, the activation, and the residual adds are all **pointwise**. They don't care how data is laid out across GPUs *as long as each GPU has whatever it needs locally to compute its piece*. In TP we make the simple choice: **when data is sitting full on both GPUs, both GPUs just run the pointwise op on their own copy.** Same input, same output, redundant compute. Why not have one GPU compute it and broadcast the result? Because **comm is the bottleneck, not compute.** A pointwise op over a few thousand numbers costs essentially nothing on a GPU; sending data across GPUs costs real latency and bandwidth. Doing the same cheap arithmetic twice is the better trade. Keep this in your back pocket — it's why you'll see "redundant" appear in the trace tables below for every LN and residual step.

So the whole TP story for this block lives at those four matmuls. Two GPUs, four cuts to make. Let's play.

---

## 2. v1 — apply Strategy A (full → half) to every matmul

What's the obvious first move? From article 01, Strategy A was:

- the **cheap** cut (concatenate, no all-reduce inside the matmul);
- on QKV it happens to **land exactly on head boundaries** — `k = 8 · 64 = 512`, split into 256 per GPU = 4 heads each;
- and "full input in, half output out" is the easier story to picture.

So apply A to all four matmuls. Walk through the block one step at a time, watching what each GPU holds — its **weight shard**, **input**, and **output** at every step.

<table class="tp-trace">
<thead>
<tr><th>Step</th><th>GPU 1</th><th>GPU 2</th></tr>
</thead>
<tbody>
<tr>
  <td class="step-label">input</td>
  <td><code>[4×512]</code> full</td>
  <td><code>[4×512]</code> full</td>
</tr>
<tr>
  <td class="step-label">LayerNorm <span class="note">(redundant)</span></td>
  <td>in <code>[4×512]</code> → out <code>[4×512]</code></td>
  <td>in <code>[4×512]</code> → out <code>[4×512]</code></td>
</tr>
<tr>
  <td class="step-label">QKV proj (A)</td>
  <td>W <code>[512×768]</code> (heads 1–4)<br>in <code>[4×512]</code> → out <code>[4×768]</code><br><span class="note">= Q+K+V for heads 1–4, each <code>[4×256]</code></span></td>
  <td>W <code>[512×768]</code> (heads 5–8)<br>in <code>[4×512]</code> → out <code>[4×768]</code><br><span class="note">= Q+K+V for heads 5–8, each <code>[4×256]</code></span></td>
</tr>
<tr>
  <td class="step-label">attention</td>
  <td>heads 1–4<br>in <code>[4×768]</code> → out <code>[4×256]</code></td>
  <td>heads 5–8<br>in <code>[4×768]</code> → out <code>[4×256]</code></td>
</tr>
<tr class="sync">
  <td colspan="3"><span class="star">★</span> GATHER #1 — output proj needs full <code>k=512</code>, each GPU only holds 256 → <code>[4×512]</code> on both</td>
</tr>
<tr>
  <td class="step-label">output proj (A)</td>
  <td>W <code>[512×256]</code><br>in <code>[4×512]</code> → out <code>[4×256]</code></td>
  <td>W <code>[512×256]</code><br>in <code>[4×512]</code> → out <code>[4×256]</code></td>
</tr>
<tr class="sync">
  <td colspan="3"><span class="star">★</span> GATHER #2 — residual needs full <code>d=512</code>, output is half <code>d=256</code> → <code>[4×512]</code> on both</td>
</tr>
<tr>
  <td class="step-label">+ residual</td>
  <td><code>[4×512]</code> → <code>[4×512]</code></td>
  <td><code>[4×512]</code> → <code>[4×512]</code></td>
</tr>
<tr>
  <td class="step-label">LayerNorm <span class="note">(redundant)</span></td>
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
  <td colspan="3"><span class="star">★</span> GATHER #3 — FFN-down needs full <code>4d=2048</code>, each GPU only holds 1024 → <code>[4×2048]</code> on both</td>
</tr>
<tr>
  <td class="step-label">FFN-down (A)</td>
  <td>W <code>[2048×256]</code><br>in <code>[4×2048]</code> → out <code>[4×256]</code></td>
  <td>W <code>[2048×256]</code><br>in <code>[4×2048]</code> → out <code>[4×256]</code></td>
</tr>
<tr class="sync">
  <td colspan="3"><span class="star">★</span> GATHER #4 — residual needs full <code>d=512</code>, output is half <code>d=256</code> → <code>[4×512]</code> on both</td>
</tr>
<tr>
  <td class="step-label">+ residual</td>
  <td><code>[4×512]</code> → <code>[4×512]</code></td>
  <td><code>[4×512]</code> → <code>[4×512]</code></td>
</tr>
</tbody>
</table>

**Four cross-GPU gathers per block.**

Two of them happen because the next A-style matmul demands a full input. The other two happen because the residual add expects a full vector and we just produced a half one. Same root cause: **Strategy A produces a half output, and almost everything downstream wants a full input.**

---

## 3. The cost of v1

Cross-GPU comm is the *slow* thing in distributed compute. The whole point of TP design is to do as few of these as possible. v1 has us paying for a gather in front of nearly every operation that needs full features.

For a 32-block model that's ~130 cross-GPU comms per forward pass. Way too many.

So the question becomes:

> **Can we avoid the gather?**

Each gather only exists because the next op needed a full vector and Strategy A had just produced a half one. What we actually need is a matmul that's *happy* consuming the half output directly.

Article 01 already handed us one.

---

## 4. v2 — pair Strategy A with Strategy B (half → sum)

Look at the two strategies through one specific lens:

- **Strategy A** *outputs* a **half**.
- **Strategy B** *inputs* a **half**.

**Same shape.** A's output is exactly what B wants as input. They snap together with no comm between them.

So replace v1's "A → gather → A" with "A → B." B eats the half output directly. The only comm cost shows up at the *end* of B — the all-reduce that turns the partial sum into the full output the residual + LN want.

Apply this to the block — pair every A matmul with a B matmul:

<table class="tp-trace">
<thead>
<tr><th>Step</th><th>GPU 1</th><th>GPU 2</th></tr>
</thead>
<tbody>
<tr>
  <td class="step-label">input</td>
  <td><code>[4×512]</code> full</td>
  <td><code>[4×512]</code> full</td>
</tr>
<tr>
  <td class="step-label">LayerNorm <span class="note">(redundant)</span></td>
  <td>in <code>[4×512]</code> → out <code>[4×512]</code></td>
  <td>in <code>[4×512]</code> → out <code>[4×512]</code></td>
</tr>
<tr>
  <td class="step-label">QKV proj (A)</td>
  <td>W <code>[512×768]</code> (heads 1–4)<br>in <code>[4×512]</code> → out <code>[4×768]</code><br><span class="note">= Q+K+V for heads 1–4, each <code>[4×256]</code></span></td>
  <td>W <code>[512×768]</code> (heads 5–8)<br>in <code>[4×512]</code> → out <code>[4×768]</code><br><span class="note">= Q+K+V for heads 5–8, each <code>[4×256]</code></span></td>
</tr>
<tr>
  <td class="step-label">attention</td>
  <td>heads 1–4<br>in <code>[4×768]</code> → out <code>[4×256]</code></td>
  <td>heads 5–8<br>in <code>[4×768]</code> → out <code>[4×256]</code></td>
</tr>
<tr>
  <td class="step-label">output proj (B)</td>
  <td>W <code>[256×512]</code><br>in <code>[4×256]</code> → out <code>[4×512]</code> <span class="note">(partial sum)</span></td>
  <td>W <code>[256×512]</code><br>in <code>[4×256]</code> → out <code>[4×512]</code> <span class="note">(partial sum)</span></td>
</tr>
<tr class="sync">
  <td colspan="3"><span class="star">★</span> ALL-REDUCE #1 — sum the two partial <code>[4×512]</code> halves into the full <code>[4×512]</code> on both GPUs (residual + LN need it)</td>
</tr>
<tr>
  <td class="step-label">+ residual</td>
  <td><code>[4×512]</code> → <code>[4×512]</code></td>
  <td><code>[4×512]</code> → <code>[4×512]</code></td>
</tr>
<tr>
  <td class="step-label">LayerNorm <span class="note">(redundant)</span></td>
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
  <td>W <code>[1024×512]</code><br>in <code>[4×1024]</code> → out <code>[4×512]</code> <span class="note">(partial sum)</span></td>
  <td>W <code>[1024×512]</code><br>in <code>[4×1024]</code> → out <code>[4×512]</code> <span class="note">(partial sum)</span></td>
</tr>
<tr class="sync">
  <td colspan="3"><span class="star">★</span> ALL-REDUCE #2 — sum the two partial <code>[4×512]</code> halves into the full <code>[4×512]</code> on both GPUs (residual + LN need it)</td>
</tr>
<tr>
  <td class="step-label">+ residual</td>
  <td><code>[4×512]</code> → <code>[4×512]</code></td>
  <td><code>[4×512]</code> → <code>[4×512]</code></td>
</tr>
</tbody>
</table>

**Two all-reduces per block.**

That's the Megatron pattern. We didn't have to be told it — we walked into it.

---

## 5. The duality you didn't see coming

Article 01 introduced A and B as if they were two separate strategies — two ways to read one matrix. Put them side by side and look at what flows in and out of each:

- **A** takes a **full input** and produces a **half output**.
- **B** takes a **half input** and produces a **full sum** as output.

They're not two strategies. They're **two halves of one round-trip.** A's output shape *is* B's input shape. B's output shape (after the all-reduce) *is* A's input shape. You couldn't have invented A without secretly inventing B as its return half.

Now look at what the block actually does:

- **Attention** has a *widen* (QKV projection: `d → k`) followed by a *narrow* (output projection: `k → d`).
- **FFN** has a *widen* (`d → 4d`) followed by a *narrow* (`4d → d`).

A widening matmul is exactly where A makes sense — there are lots of output features to spread across GPUs. A narrowing matmul is exactly where B makes sense — there are lots of input features to spread across GPUs, and the small output is something you sum back up.

The block isn't accidentally A→B-friendly. It's **structurally** A→B-friendly: two widen-narrow pairs glued together by pointwise things. The "Megatron pattern" isn't really an algorithm someone designed. It's the only comm pattern that respects what the architecture was already doing. The duality of A and B and the widen-narrow rhythm of the block are the same fact told twice.

A quick word on cost: a gather and an all-reduce move similar amounts of data per GPU (an all-reduce is roughly a reduce-scatter followed by an all-gather under the hood). v1 had **4 gathers** per block; v2 has **2 all-reduces** — half the comm, with no change to the model itself.

---

## 6. Why the cut has to land on a head boundary

The v2 trace quietly assumed something: that QKV's column cut splits `k = 512` into two slabs of 256 along the **head boundary**, so each GPU owns 4 whole heads. That assumption is doing more work than it looks like. Try the counterfactual.

Imagine **single-head attention** — same `k = 512`, but one head, no head structure. Apply Strategy A on QKV exactly as before: each GPU gets `Q, K, V` each of shape `[4 × 256]`. Now run attention.

The first step is `Q Kᵀ`. Each GPU computes `Q_half @ K_halfᵀ`, producing a `[4 × 4]` matrix — but that matrix is a **partial sum** over the 256 features each GPU happens to hold. The true scores are the sum of both GPUs' partials.

Here's the problem: the next step is **softmax**. Softmax is non-linear, so you can't apply it locally and reconcile after — `softmax(a) + softmax(b) ≠ softmax(a + b)`. The reduction has to happen *before* softmax. Which means an extra sync sitting right in the middle of attention:

> ★ ALL-REDUCE on the `[n × n]` scores, before softmax.

That's a third all-reduce per block, on top of v2's two. The Megatron pattern collapses to *three* sync points, and the new one is on a tensor that scales with sequence length squared — exactly the comm you most want to avoid.

The fix is structural, not algorithmic: don't let the cut cross a head. **Each head's `Q Kᵀ` must live entirely on one GPU**, so the partial-sum problem never arises. Multi-head attention gives that to us for free — heads are independent by construction, head boundaries are natural cut points, and the column split on `k = h · d_head` lands exactly between them whenever `h` divides evenly across GPUs.

So multi-head isn't a happy coincidence the systems people exploited. It's the **structural prerequisite** for v2 to exist at all. Pick any cut that lands inside a head, and softmax forces a sync that ruins everything. Pick a cut that lands between heads, and the non-linearity stays local. The Megatron pattern doesn't just *happen* to work on multi-head architectures — it requires them.

---

## 7. What this opens

You now have **one block** running on two GPUs with two all-reduces per pass. That earns the next round of "wait, but what about..." questions:

- **What if I have many blocks and many GPUs?** TP cuts *within* a block. The cut *across* blocks — staging entire blocks on different GPUs and pipelining microbatches through them — is a different beast. **Pipeline parallelism**, next article.
- **What if FFN is replaced with experts?** The column-then-row pattern still applies to each expert's matmuls, but routing tokens to the right expert introduces a new kind of comm. **MoE**, soon.
- **What if the batch's sequence lengths are wildly different?** The comm pattern is unchanged, but the attention math has to deal with variable-length sequences — and that's where continuous batching enters.

Same grammar. Each one is its own walk-through.
