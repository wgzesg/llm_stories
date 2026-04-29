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
weight: 2
---

[Article 01](/llm_stories/posts/01-tensor-parallelism-mental-model/) left you with two ways to split **one** matmul across two GPUs. They're easier to keep straight by what they *do* than by what they're called in the literature, so let's lay them out side by side:

|                          | **Strategy A** — *split the `fx`es*                  | **Strategy B** — *split the rows*                            |
|--------------------------|--------------------------------------------------------|----------------------------------------------------------------|
| What you slice           | the matrix's columns (each column is one `fx`)         | the matrix's rows (each row is a basis vector)                 |
| Each GPU's **input**     | the **full** input vector                              | **half** of the input features                                 |
| Each GPU's **output**    | **half** of the output features                        | a **partial sum** of the *full* output                         |
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

So apply A to all four matmuls. Walk through the block one step at a time, watching the shape that **each** GPU holds. In the trace below, `(A)` next to a matmul means "Strategy A: full input → half output."

```
Step                                Each GPU holds              Comm
─────────────────────────────────────────────────────────────────────────
input                               [4 × 512]   full            —
LayerNorm                           [4 × 512]   full            — (redundant)
QKV proj      (A: full→half)        [4 × 768]   its 4 heads of QKV   —
attention                           [4 × 256]   its 4 heads' out     —

                                    ↓ next matmul (output proj) is also A,
                                    ↓ which needs the FULL k=512 input,
                                    ↓ but each GPU only holds 256.

GATHER                              [4 × 512]   full            ★ gather #1
output proj   (A: full→half)        [4 × 256]   half of d output     —

                                    ↓ next is + residual; residual is full d=512,
                                    ↓ output is split d=256.

GATHER                              [4 × 512]   full            ★ gather #2
+ residual                          [4 × 512]   full            —
LayerNorm                           [4 × 512]   full            — (redundant)

FFN-up        (A: full→half)        [4 × 1024]  half FFN-hidden      —
activation                          [4 × 1024]  half (pointwise)     —

                                    ↓ next matmul (FFN-down) is A; needs FULL
                                    ↓ 4d=2048 input, each GPU only has 1024.

GATHER                              [4 × 2048]  full            ★ gather #3
FFN-down      (A: full→half)        [4 × 256]   half of d output     —

                                    ↓ next is + residual; full d=512 needed.

GATHER                              [4 × 512]   full            ★ gather #4
+ residual                          [4 × 512]   full            —
```

**Four cross-GPU gathers per block.** Per forward pass. (And another four in backward.)

Two of them happen because the next A-style matmul demands a full input. The other two happen because the residual add expects a full vector and we just produced a half one. Same root cause: **Strategy A produces a half output, and almost everything downstream wants a full input.**

---

## 3. The cost of v1

Cross-GPU comm is the *slow* thing in distributed compute. The whole point of TP design is to do as few of these as possible. v1 has us paying for a gather in front of nearly every operation that needs full features.

For a 32-block model that's ~130 cross-GPU comms per forward pass — and we doubled it for backward. Way too many.

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

```
Step                                Each GPU holds              Comm
─────────────────────────────────────────────────────────────────────────
input                               [4 × 512]   full            —
LayerNorm                           [4 × 512]   full            — (redundant)
QKV proj      (A: full→half)        [4 × 768]   its 4 heads of QKV   —
attention                           [4 × 256]   its 4 heads' out     —
output proj   (B: half→sum)         [4 × 512]   partial sum     —

                                    ↓ need full for + residual / LN

ALL-REDUCE                          [4 × 512]   full            ★ all-reduce #1
+ residual                          [4 × 512]   full            —
LayerNorm                           [4 × 512]   full            — (redundant)

FFN-up        (A: full→half)        [4 × 1024]  half FFN-hidden      —
activation                          [4 × 1024]  half (pointwise)     —
FFN-down      (B: half→sum)         [4 × 512]   partial sum     —

                                    ↓ need full for + residual / LN

ALL-REDUCE                          [4 × 512]   full            ★ all-reduce #2
+ residual                          [4 × 512]   full            —
```

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

## 6. Multi-head attention was pre-cut for this

Multi-head attention was invented years before tensor parallelism. The motivation was purely **modeling**: different heads should learn to attend to different relational patterns. So someone sliced the `k`-dimensional attention space into `h = 8` chunks of `d_head = 64` each, ran a separate attention on each chunk, concatenated the results, and carried on. A choice about *what the model can learn* — not about *how to run it on multiple GPUs*.

Years later, TP came along and Strategy A needed something specific from the QKV cut: it needed **independent slabs** that could run a non-linear, sequence-mixing operation (attention) without ever syncing across GPUs.

That's a tall order. Most cuts of a neural network leave dependencies between the pieces — and the moment you have dependencies, you need comm.

But multi-head attention had already done the hard part. **Heads are independent by construction.** Head 1 only mixes Head 1's queries and keys, Head 2 does its own thing, and they never need to peek at each other. The independence is an architectural guarantee, written into the very definition of multi-head attention years before anyone was thinking about TP.

Look at our setup: `h = 8` heads, 2 GPUs, 4 heads each. Strategy A's column cut on QKV (`k = 512` split into 256 per GPU) **lands exactly on a head boundary** — each GPU owns 4 whole heads. The per-head softmax — usually a sync point that ruins this kind of trick — runs entirely inside the GPU that owns the head. Even the non-linearity is local.

That's the part the literature usually skims past. Multi-head attention was a gift the modelers accidentally left for the systems people. The cuts were already drawn, the independence guarantee was already established, the comm-free attention computation was already there. **TP just walked in and used them.** It didn't have to invent anything.

---

## 7. What this opens

You now have **one block** running on two GPUs with two all-reduces per forward pass. That earns the next round of "wait, but what about..." questions:

- **What if I have many blocks and many GPUs?** TP cuts *within* a block. The cut *across* blocks — staging entire blocks on different GPUs and pipelining microbatches through them — is a different beast. **Pipeline parallelism**, next article.
- **What if FFN is replaced with experts?** The column-then-row pattern still applies to each expert's matmuls, but routing tokens to the right expert introduces a new kind of comm. **MoE**, soon.
- **What if the batch's sequence lengths are wildly different?** The comm pattern is unchanged, but the attention math has to deal with variable-length sequences — and that's where continuous batching enters.

Same grammar. Each one is its own walk-through.
