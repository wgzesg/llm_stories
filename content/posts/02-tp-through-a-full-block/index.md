---
title: "Walking Tensor Parallelism Through a Full Block"
date: 2026-04-29T00:00:00+00:00
draft: false
summary: "Take article 01's two cuts and walk them through a full transformer block. Try column-parallel everywhere first, watch the comm explode, then let row-parallel catch column's output for free — and land at two all-reduces per block."
description: "How to split a full transformer block across two GPUs. Start with column-parallel everywhere, see why it costs a gather in front of every matmul, then pair it with row-parallel to land at the Megatron pattern of two all-reduces per block."
tags: ["tensor-parallelism", "transformers", "llm-serving", "megatron", "mental-model"]
series: ["llm-stories"]
showToc: true
TocOpen: false
weight: 2
---

[Article 01](/llm_stories/posts/01-tensor-parallelism-mental-model/) left you with two ways to split **one** matmul across two GPUs:

- **Strategy A (column-parallel)** — each GPU runs its half of the `fx`es on the **full** input. Outputs **concatenate**. Cheap.
- **Strategy B (row-parallel)** — each GPU runs its rows on **half** the input. Outputs are **partial sums** that need an **all-reduce**. Expensive (one comm step per layer).

But a real transformer block isn't one matmul — it's four, plus some pointwise glue. The natural next question is: **how do we cut a *whole block* across two GPUs?**

There's an obvious first move that *almost* works. Let's start there, see exactly where it breaks, and let the fix walk us into the canonical Megatron pattern.

---

## 1. One block, drawn flat

A transformer is just a stack of `N` "blocks." Open one up and look inside:

```
       │
   LayerNorm
       │
   QKV projection      d → 3k     ← matmul
       │
   attention            (no new matmul; mixes Qs with Ks)
       │
   output projection   k → d      ← matmul
       │
       + residual
       │
   LayerNorm
       │
   FFN up-projection   d → 4d     ← matmul
       │
   activation (GeLU)    (pointwise)
       │
   FFN down-projection 4d → d     ← matmul
       │
       + residual
       │
```

**Four matmuls** and some glue. The glue — LayerNorms, the activation, the residual adds — is all **pointwise**. It doesn't care how the data is split across GPUs, as long as each GPU has whatever it needs locally. So the *only* interesting decisions in this whole block are at those four matmuls.

We have two GPUs. Let's play.

---

## 2. v1 — just split everything column-wise

What's the obvious first move? From article 01:

- column-parallel was the **cheap** cut (concatenate, no all-reduce in the middle of a single matmul);
- column-parallel applied to the QKV projection happens to **land exactly on head boundaries** — each GPU naturally owns "its heads" of Q, K, V;
- and intuitively, "each GPU computes its half of the output features" is just easier to picture than the row-parallel story.

So: take all four matmuls and apply Strategy A to each. Walk through the block one step at a time, asking *what does each GPU hold after this step?*

```
QKV proj (col-parallel)
   GPU 1 → its half of QKV   (its heads)
   GPU 2 → its half of QKV   (its heads)
```

So far so good. Each GPU has half the columns of the QKV output.

```
attention (each GPU on its own heads)
   GPU 1 → attention output for its heads      (still split)
   GPU 2 → attention output for its heads      (still split)
```

Still split. Now the next matmul — the output projection — is also column-parallel.

```
output proj (col-parallel)
   each GPU needs the FULL input vector  ← uh oh
   ...but each GPU only has its half.
```

Stuck. To run the next matmul, both GPUs need the *full* input vector. So before the output projection we have to do a cross-GPU **gather** — each GPU sends its half to the other so both end up holding the whole thing.

The same thing happens at the FFN up-projection (column-parallel, needs full input → gather), and at the FFN down-projection (gather).

Count them up: **three gathers per block, every forward pass.** (And another three in the backward pass.)

---

## 3. The cost of v1

Cross-GPU communication is the *slow* thing in distributed compute. The whole point of caring about TP comm patterns is to do as few of these as possible. v1 has us paying for a gather in front of nearly every matmul.

For a 32-block model that's ~100 cross-GPU comms per forward pass on a single training step. Way too many.

So the question becomes:

> **Can we avoid the gather?**

The gather only exists because we picked column-parallel for the matmul *after* attention, and column-parallel demands a full input. What we actually need is a matmul that's *happy* to consume a split input.

Article 01 already gave us one.

---

## 4. v2 — let the next matmul consume the split directly

Look at the two strategies through one specific lens:

- column-parallel **outputs** something split into halves (the kind you'd concatenate).
- row-parallel **inputs** something split into halves (one feature half on each GPU).

**Same shape.** Strategy A's output is exactly what Strategy B wants as input. They snap together with no comm between them.

So replace the v1 pattern of "column → gather → column" with "column → row." No gather. Strategy B just eats the split directly.

What does this look like across the whole block? Pair every column-parallel matmul with a row-parallel one:

```
QKV proj (col)
   GPU 1, GPU 2 → split QKV   (no comm)
attention (each GPU on its heads)
   GPU 1, GPU 2 → split attention output
output proj (row)
   GPU 1, GPU 2 → partial sum, length d
   → all-reduce
   GPU 1, GPU 2 → full output, length d, identical on both

+ residual
LayerNorm                     (full vector on each GPU, runs locally)

FFN up (col)
   GPU 1, GPU 2 → split FFN-hidden, length 2d each   (no comm)
activation (pointwise)
   GPU 1, GPU 2 → still split
FFN down (row)
   GPU 1, GPU 2 → partial sum, length d
   → all-reduce
   GPU 1, GPU 2 → full output, length d, identical on both

+ residual
LayerNorm
```

Two A→B pairs. **Two all-reduces per block.**

That's the Megatron pattern. We didn't have to be told it — we walked into it.

---

## 5. Why this is the natural rhythm

Look at what the data looks like at each "rest point" in the block:

- **Before any matmul:** full vector, identical on both GPUs.
- **Between A and B:** split across GPUs. *No comm needed* — that's exactly the shape B wants.
- **After B + all-reduce:** full vector, identical on both GPUs.
- **Before the next A:** full vector again. ✓

So the block **enters** in the "full vector replicated" state and **leaves** in the same state. In between, the data is allowed to be split — but only across the A→B span, which is the *one place* where split is the right thing.

The pattern isn't a clever construction. It's just **the only chain where A's output shape matches B's input shape**. Anything else would need a comm step in the middle to fix the mismatch.

The pointwise glue (residual adds, LayerNorms, the activation) all run on the post-all-reduce, full-vector state — they need full features to do their job, and the all-reduce just delivered that. The pieces fit because the rhythm makes them fit.

---

## 6. One more check — multihead attention changes nothing

The QKV projection has output dimension `k = h × d_head` (number of heads × per-head dimension). When you split column-parallel on `k`, the cut **lands exactly between heads**. Each GPU ends up owning some heads' worth of Q, K, V.

Attention itself then runs **locally** on each GPU's heads — Head 1 only mixes Head 1's queries and keys, Head 2 does its own thing, and they never need to peek at each other. So even though attention introduces a non-linearity (per-head softmax over the relative positions), that non-linearity stays inside the GPU that owns the head. No inter-GPU comm.

From the **comm perspective**, the multihead version is byte-for-byte identical to the single-head version. Same column cut on QKV, same row cut on output projection, same one all-reduce.

Multihead was a *modeling* decision — different heads learn to attend to different relational patterns. It just happens to make the column cut feel even more natural, because the cut respects head boundaries by construction. From the systems side, nothing about the comm pattern depends on whether you have one head or 32.

---

## 7. What this opens

You now have **one block** running on two GPUs with two all-reduces per forward pass. That earns the next round of "wait, but what about..." questions:

- **What if I have many blocks and many GPUs?** TP cuts *within* a block. The cut *across* blocks — staging entire blocks on different GPUs and pipelining microbatches through them — is a different beast. **Pipeline parallelism**, next article.
- **What if FFN is replaced with experts?** The column-then-row pattern still applies to each expert's matmuls, but now there's a routing step (which token goes to which expert) that introduces a new kind of comm. **MoE**, soon.
- **What if the batch's sequence lengths are wildly different?** The comm pattern is unchanged, but the attention math has to deal with variable-length sequences — which is where continuous batching enters.

Same grammar. Each one is its own walk-through.
