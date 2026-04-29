---
title: "Walking Tensor Parallelism Through a Full Block"
date: 2026-04-29T00:00:00+00:00
draft: false
summary: "Take article 01's two cuts and walk them through a full transformer block, with concrete shapes on each GPU at every step. Try column-parallel everywhere first, watch the comm explode (four gathers per block), then let row-parallel catch column's output for free — and land at two all-reduces per block."
description: "How to split a full transformer block across two GPUs, with concrete shapes traced through every step. Start with column-parallel everywhere, see why it costs four gathers per block, then pair it with row-parallel to land at the Megatron pattern of two all-reduces per block."
tags: ["tensor-parallelism", "transformers", "llm-serving", "megatron", "mental-model"]
series: ["llm-stories"]
showToc: true
TocOpen: false
weight: 2
---

[Article 01](/llm_stories/posts/01-tensor-parallelism-mental-model/) left you with two ways to split **one** matmul across two GPUs:

- **Strategy A (column-parallel)** — each GPU runs its half of the `fx`es on the **full** input. Outputs **concatenate**. Cheap.
- **Strategy B (row-parallel)** — each GPU runs its rows on **half** the input. Outputs are **partial sums** that need an **all-reduce**. One comm step per layer.

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

## 2. v1 — just split everything column-wise

What's the obvious first move? From article 01:

- column-parallel was the **cheap** cut (concatenate, no all-reduce in the middle of one matmul);
- on QKV it happens to **land exactly on head boundaries** — `k = 8 · 64 = 512`, split column-wise into 256 per GPU = 4 heads per GPU;
- and "each GPU computes its own half of the output features" is just easier to picture.

So apply column-parallel to all four matmuls. Walk through the block one step at a time, watching the shape that **each** GPU holds:

```
Step                                Each GPU holds              Comm
─────────────────────────────────────────────────────────────────────────
input                               [4 × 512]   full            —
LayerNorm                           [4 × 512]   full            — (redundant)
QKV proj    (col)                   [4 × 768]   its 4 heads of QKV   —
attention                           [4 × 256]   its 4 heads' out     —

                                    ↓ next is output proj (col), it needs the FULL
                                    ↓ k=512 input, but each GPU only has 256.

GATHER                              [4 × 512]   full            ★ gather #1
output proj (col)                   [4 × 256]   half of d output     —

                                    ↓ next is + residual, residual is full d=512,
                                    ↓ output is split d=256.

GATHER                              [4 × 512]   full            ★ gather #2
+ residual                          [4 × 512]   full            —
LayerNorm                           [4 × 512]   full            — (redundant)

FFN-up      (col)                   [4 × 1024]  half FFN-hidden      —
activation                          [4 × 1024]  half (pointwise)     —

                                    ↓ next is FFN-down (col), needs FULL 4d=2048
                                    ↓ input, each GPU only has 1024.

GATHER                              [4 × 2048]  full            ★ gather #3
FFN-down    (col)                   [4 × 256]   half of d output     —

                                    ↓ next is + residual, full d=512 needed.

GATHER                              [4 × 512]   full            ★ gather #4
+ residual                          [4 × 512]   full            —
```

**Four cross-GPU gathers per block.** Per forward pass. (And another four in backward.)

Two of them happen because the next col-parallel matmul demands a full input. The other two happen because the residual add expects a full vector and we just produced a split one. Same root cause either way: **column-parallel produces a split output, and almost everything downstream wants a full input.**

---

## 3. The cost of v1

Cross-GPU comm is the *slow* thing in distributed compute. The whole point of TP design is to do as few of these as possible. v1 has us paying for a gather in front of nearly every operation that needs full features.

For a 32-block model that's ~130 cross-GPU comms per forward pass — and we doubled it for backward. Way too many.

So the question becomes:

> **Can we avoid the gather?**

Each gather only exists because the next op needed a full vector that we'd just split. What we actually need is for the next matmul to be *happy* with the split input.

Article 01 already handed us one.

---

## 4. v2 — let the next matmul consume the split directly

Look at the two strategies through one specific lens:

- column-parallel **outputs** something split into halves.
- row-parallel **inputs** something split into halves.

**Same shape.** Strategy A's output is exactly what Strategy B wants as input. They snap together with no comm between them.

So replace v1's "column → gather → column" with "column → row." Strategy B eats the split directly. The only comm cost shows up at the *end* of B (the all-reduce that turns partial sums into the full output the residual + LN want).

Apply this to the block — pair every column-parallel matmul with a row-parallel one:

```
Step                                Each GPU holds              Comm
─────────────────────────────────────────────────────────────────────────
input                               [4 × 512]   full            —
LayerNorm                           [4 × 512]   full            — (redundant)
QKV proj    (col)                   [4 × 768]   its 4 heads of QKV   —
attention                           [4 × 256]   its 4 heads' out     —
output proj (row)                   [4 × 512]   partial sum     —

                                    ↓ need full for + residual / LN

ALL-REDUCE                          [4 × 512]   full            ★ all-reduce #1
+ residual                          [4 × 512]   full            —
LayerNorm                           [4 × 512]   full            — (redundant)

FFN-up      (col)                   [4 × 1024]  half FFN-hidden      —
activation                          [4 × 1024]  half (pointwise)     —
FFN-down    (row)                   [4 × 512]   partial sum     —

                                    ↓ need full for + residual / LN

ALL-REDUCE                          [4 × 512]   full            ★ all-reduce #2
+ residual                          [4 × 512]   full            —
```

**Two all-reduces per block.**

That's the Megatron pattern. We didn't have to be told it — we walked into it.

---

## 5. Why this is the natural rhythm

Look at what the data looks like at each "rest point":

- **Before any matmul:** `[4 × 512]` full, identical on both GPUs.
- **Between A and B (inside attention, inside FFN):** split across GPUs. *No comm needed* — that's exactly what B wants.
- **After B + all-reduce:** `[4 × 512]` full, identical on both GPUs.
- **Before the next A:** `[4 × 512]` full again. ✓

The block **enters** in the "full vector replicated" state and **leaves** in the same state. In between, the data is allowed to be split — but only across the A→B span, which is the *one place* where split is the right shape.

The pattern isn't a clever construction. It's just **the only chain where A's output shape matches B's input shape**, *and* the "full vector replicated" state is preserved at the boundaries where the pointwise glue (residual, LN) lives. Everything snaps where it needs to snap.

A quick word on cost: a gather and an all-reduce move similar amounts of data per GPU (an all-reduce is essentially a reduce-scatter followed by an all-gather under the hood). v1 had **4 gathers** per block; v2 has **2 all-reduces**. Roughly half the comm, with no change to the model itself.

---

## 6. One more check — multihead attention changes nothing

The QKV projection has output dim `k = h · d_head = 8 · 64 = 512`. When you split column-parallel on `k` with 2 GPUs, the cut **lands exactly between heads** — each GPU ends up owning 4 heads' worth of Q, K, V, sized `[4 × 256]` per GPU.

Attention itself then runs **locally** on each GPU's 4 heads. Head 1 only mixes Head 1's queries and keys, Head 2 does its own thing, and they never need to peek at each other. So even though attention introduces a non-linearity (the per-head softmax over relative positions), that non-linearity stays *inside the GPU that owns the head*. **No inter-GPU comm, ever, inside attention.**

From the **comm perspective**, the multihead version is byte-for-byte identical to a single-head one. Same column cut on QKV, same row cut on output projection, same one all-reduce.

Multihead was a *modeling* choice — different heads learn to attend to different relational patterns. It just happens to make the column cut feel even more natural, because the cut respects head boundaries by construction. From the systems side, nothing about the comm pattern depends on whether you have one head or 32.

---

## 7. What this opens

You now have **one block** running on two GPUs with two all-reduces per forward pass. That earns the next round of "wait, but what about..." questions:

- **What if I have many blocks and many GPUs?** TP cuts *within* a block. The cut *across* blocks — staging entire blocks on different GPUs and pipelining microbatches through them — is a different beast. **Pipeline parallelism**, next article.
- **What if FFN is replaced with experts?** The column-then-row pattern still applies to each expert's matmuls, but routing tokens to the right expert introduces a new kind of comm. **MoE**, soon.
- **What if the batch's sequence lengths are wildly different?** The comm pattern is unchanged, but the attention math has to deal with variable-length sequences — and that's where continuous batching enters.

Same grammar. Each one is its own walk-through.
