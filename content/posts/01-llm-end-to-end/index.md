---
title: "An LLM, End to End"
date: 2026-05-06T00:00:00+00:00
draft: false
summary: "Three zoom levels — the model end-to-end, one transformer block opened up, and the loop that turns a prompt into output. Just enough to ask the right questions about everything that comes after."
description: "Fundamentals of a modern decoder-only LLM at three zoom levels: the bird's-eye stack, one transformer block, and the generation loop. The on-ramp for the rest of the LLM Stories series."
tags: ["fundamentals", "transformer", "llm-serving", "mental-model"]
series: ["llm-stories"]
showToc: true
TocOpen: false
weight: 2
---

An LLM, at its plainest, is a function that reads some text and produces more text. You give it a prompt; it gives you a continuation. That's the whole user-facing contract.

But "produces more text" is hiding a lot. The model doesn't compose a whole reply in one shot — it produces *one token at a time*, looping on its own output. And the thing that produces each next token is a fixed-size mathematical machine called a **transformer**.

This article opens that machine up at three zoom levels:

1. The whole machine, end to end — what comes in, what comes out, what's in between.
2. One layer up close — what's actually inside the part labelled "transformer block".
3. The full loop — how the machine is used to produce a long reply, one token at a time.

We'll keep things abstract — symbols like `d`, `L`, `h` rather than specific numbers — because the *structure* is what's portable. Different models pick different sizes, but they all sit in this same shape. Concrete numbers earn their place in later articles, where they're actually load-bearing.

Along the way, real questions surface naturally — things like *"wait, does the model really redo all of that every time it produces one token?"* or *"what if the model is too big to fit on one GPU?"* Those questions are exactly what the rest of the series picks apart. Each one becomes its own article down the line.

---

## Part I — The model end to end

## 1. Tokens in, next token out

Hand the model a string — say, `"the quick brown fox jumps over"` — and ask it to keep going. What does it actually do, mechanically? Six steps, top to bottom.

**1. Tokenize.** First, the string is chopped into pieces called **tokens**. Each token is a small integer ID — because under the hood, the model can only do arithmetic on numbers. Roughly: common short words become one token each, rarer words get broken into a few pieces. We'll call the count of tokens `N`.

**2. Embed.** Each integer ID gets looked up in a giant table called the **embedding table**. The table has one row per possible token in the vocabulary, and each row is a vector of `d` numbers. (`d` is one of the model's design choices — its **hidden dimension**. Real models put `d` in the thousands.) Looking up `N` tokens turns a list of `N` IDs into a tensor of shape `[N × d]`: `N` rows, each `d` numbers wide.

Why a vector and not just keep the integer ID? Because the model only knows how to do linear algebra, and an integer ID has no useful geometry — token 5 isn't "closer to" token 6 than to token 100, even though they're consecutive integers. The embedding table gives every token a *learned point* in a `d`-dimensional space, where tokens with similar meanings end up nearby and unrelated ones end up far apart. Each row is the model's initial, context-free *feeling* about what that token means.

(The vocabulary holds some `vocab` distinct tokens — typically tens of thousands. So the embedding table itself is `[vocab × d]`.)

**3. A stack of transformer blocks.** This `[N × d]` tensor now flows through `L` **transformer blocks**, stacked one on top of the next. Each block reads the whole sequence, mixes information across positions, and writes back a refined version. Crucially, every block's input and output have the *same* `[N × d]` shape — only the *contents* of the rows change. After all `L` of these passes, the rows are no longer raw, context-free meanings — they're rich, position-aware representations of what each token means *in this particular sequence*. We'll dig into why blocks stack so well in §2, and open one up in Part II.

**4. Final norm.** A small normalization step right at the top of the stack — a clean-up pass. Same shape going in, same shape coming out.

**5. LM head.** A linear layer projects each row from `d` features back out to one number per token in the vocabulary — `vocab` numbers per row. Output shape: `[N × vocab]`. Each row is a long vector of "scores" over the entire vocabulary. These scores are called **logits**. The logit for token `t` at position `i` is the model's raw, unsquashed answer to "how plausible is `t` as the next token at position `i`?"

**6. Softmax → sample.** The row we actually care about is the *last* one — the position right after the last input token, where the model's prediction for "what comes next" sits. Run that `vocab`-long row through a **softmax**, which turns the raw logits into a clean probability distribution (all positive, all summing to 1). Sample one token from that distribution. That's the model's guess for the next token.

A picture of the whole stack:

<svg viewBox="0 0 520 720" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;font-family:system-ui,sans-serif;display:block;margin:1.5rem auto">
  <defs>
    <marker id="arr-stack" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
      <path d="M 0 0 L 8 4.5 L 0 9 Z" fill="currentColor"/>
    </marker>
  </defs>
  <text x="260" y="24" text-anchor="middle" font-size="14" fill="currentColor" font-weight="600">the model, end to end</text>

  <rect x="140" y="660" width="240" height="40" fill="rgba(74,144,226,0.18)" stroke="#4a90e2" stroke-width="1.5"/>
  <text x="260" y="680" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">token IDs (integers)</text>
  <text x="260" y="694" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.65" font-family="ui-monospace,monospace">shape: [N]</text>

  <line x1="260" y1="660" x2="260" y2="624" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-stack)"/>

  <rect x="140" y="572" width="240" height="50" fill="rgba(245,166,35,0.20)" stroke="#f5a623" stroke-width="1.5"/>
  <text x="260" y="593" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">embedding lookup</text>
  <text x="260" y="610" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.65" font-family="ui-monospace,monospace">[vocab × d]</text>

  <line x1="260" y1="572" x2="260" y2="540" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-stack)"/>
  <text x="395" y="558" font-size="11" fill="currentColor" opacity="0.65" font-family="ui-monospace,monospace">[N × d]</text>

  <rect x="140" y="248" width="240" height="288" fill="rgba(150,150,150,0.08)" stroke="rgba(150,150,150,0.55)" stroke-width="1.5" stroke-dasharray="5 4"/>
  <text x="260" y="270" text-anchor="middle" font-size="13" fill="currentColor" font-weight="600">L transformer blocks</text>
  <text x="260" y="286" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.6">[N × d] in, [N × d] out, repeated</text>
  <rect x="180" y="300" width="160" height="26" fill="rgba(120,180,140,0.20)" stroke="#78b48c" stroke-width="1"/>
  <text x="260" y="318" text-anchor="middle" font-size="11" fill="currentColor">block 1</text>
  <rect x="180" y="332" width="160" height="26" fill="rgba(120,180,140,0.20)" stroke="#78b48c" stroke-width="1"/>
  <text x="260" y="350" text-anchor="middle" font-size="11" fill="currentColor">block 2</text>
  <text x="260" y="380" text-anchor="middle" font-size="14" fill="currentColor" opacity="0.55">⋮</text>
  <text x="260" y="398" text-anchor="middle" font-size="14" fill="currentColor" opacity="0.55">⋮</text>
  <rect x="180" y="412" width="160" height="26" fill="rgba(120,180,140,0.20)" stroke="#78b48c" stroke-width="1"/>
  <text x="260" y="430" text-anchor="middle" font-size="11" fill="currentColor">block L−1</text>
  <rect x="180" y="444" width="160" height="26" fill="rgba(120,180,140,0.20)" stroke="#78b48c" stroke-width="1"/>
  <text x="260" y="462" text-anchor="middle" font-size="11" fill="currentColor">block L</text>

  <line x1="260" y1="248" x2="260" y2="216" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-stack)"/>
  <text x="395" y="234" font-size="11" fill="currentColor" opacity="0.65" font-family="ui-monospace,monospace">[N × d]</text>

  <rect x="140" y="172" width="240" height="40" fill="rgba(245,166,35,0.20)" stroke="#f5a623" stroke-width="1.5"/>
  <text x="260" y="196" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">final LayerNorm</text>

  <line x1="260" y1="172" x2="260" y2="140" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-stack)"/>

  <rect x="140" y="92" width="240" height="48" fill="rgba(245,166,35,0.20)" stroke="#f5a623" stroke-width="1.5"/>
  <text x="260" y="113" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">LM head</text>
  <text x="260" y="130" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.65" font-family="ui-monospace,monospace">[d × vocab]</text>

  <line x1="260" y1="92" x2="260" y2="60" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-stack)"/>
  <text x="395" y="78" font-size="11" fill="currentColor" opacity="0.65" font-family="ui-monospace,monospace">[N × vocab] logits</text>

  <rect x="140" y="36" width="240" height="24" fill="rgba(74,144,226,0.18)" stroke="#4a90e2" stroke-width="1.5"/>
  <text x="260" y="53" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">softmax(last row) → next-token distribution</text>
</svg>

So the entire model is a function: it reads `N` tokens and returns a probability distribution over what the `(N+1)`-th token should be. Everything else — the chatty replies, the long answers, the streaming output you see in a chat UI — comes from running this function in a loop. We'll get to that loop in Part III.

---

## 2. Why blocks stack: the stream-processor pattern

Before opening up one of those blocks, there's one structural property worth naming, because it shapes everything that comes later in the series.

Look back at the six steps in §1. Once we're past tokenization, every step in the middle of the pipeline reads and writes tensors of the **same shape**:

- The embedding turns `N` token IDs into an `[N × d]` tensor.
- Each transformer block reads `[N × d]` and returns `[N × d]`.
- The final norm reads `[N × d]` and returns `[N × d]`.
- Only the LM head, at the very top, changes the width — back out to `vocab`.

The shape **never changes** in the middle of the pipeline. The contents do — each block refines the rows, building up a richer, more context-aware representation — but the geometry is fixed at `[N × d]` from the bottom of the stack to the top.

This isn't an accident. It's what makes the design work.

When the input and output of a unit have the same shape, you can drop the unit into the pipeline anywhere, in any quantity. You can stack 8 of them, or 32, or 80 — the tensor flowing between any two consecutive blocks is always the right shape for the next block to consume. Each block is a kind of **stream processor**: it reads a fixed-shape stream of tokens, refines it, and hands it on, oblivious to whether it's the first block in the stack or the last. (You may have seen this idea elsewhere — Unix pipes, audio plugins, image-processing pipelines. Same shape in, same shape out, stack as many as you want.)

That property — same shape in, same shape out — is why "make the model bigger" mostly means "stack more blocks." A small open-source model and a massive flagship model often look almost identical at this level of zoom: the same six-step pipeline, the same internal block structure, just a different `L` (and a slightly wider `d`). Same recipe, scaled.

Some names worth holding on to, since later sections and articles will use them:

- `N` — the length of the current sequence. Varies per request — it's a property of the input, not the model.
- `d` — the **hidden dimension**. The width of every row in the stream.
- `L` — how many transformer blocks are stacked.
- `vocab` — how many distinct tokens the model knows about. Sets the width of the embedding table and the LM head.

We'll meet two more in Part II: `h` (the number of attention heads inside a block) and `d_head` (each head's width).

So at this zoom level, the model is just a stack of `L` identical-shape stream processors operating on `[N × d]` tensors, capped by an embedding lookup at the bottom and an LM head at the top. The rest of the series will spend a lot of time on how to make those stream processors fast — but the stream-processor structure itself doesn't change.

---

## Part II — Inside one block

## 3. The block, drawn flat

Now let's open up one of those `L` transformer blocks. The good news: they all have the *same internal structure* — different blocks have different *learned numbers*, but the wiring is identical. So understanding one block is understanding all of them.

A block has two halves, each wrapped in a residual connection (the little `+` at the bottom of each half — we'll explain that in a sec):

```
input  [N × d]
   │
   ├──────────────────────────┐
   │                          │
 LayerNorm 1                  │   residual
   │                          │
 QKV projection               │       d → 3d, then split into Q, K, V
   │                          │
 multi-head attention         │       mixes across positions
   │                          │
 output projection            │       d → d
   │                          │
   + ────────────────────────┘
   │
   ├──────────────────────────┐
   │                          │
 LayerNorm 2                  │   residual
   │                          │
 FFN-up                       │       d → 4d
   │                          │
 activation (GeLU)            │       pointwise nonlinearity
   │                          │
 FFN-down                     │       4d → d
   │                          │
   + ────────────────────────┘
   │
output [N × d]
```

The two halves are the two main events: an **attention** sub-layer and an **FFN** (feed-forward network) sub-layer. The other parts (LayerNorm, the activation, the `+`) are smaller pieces of glue.

A quick word on what each piece is doing:

- **LayerNorm** is a normalization step — for each row of the tensor, it rescales the numbers so they have a clean mean and variance. Cheap, pointwise, and mostly there to keep the numbers from drifting into bad ranges as they pass through many layers. Think of it as a "tidy-up" pass.
- **The residual `+`** means: take what came *into* this half, and add it onto what came *out*. So each half is computing a **delta** — a refinement to the existing representation, not a replacement. That's what lets us stack many blocks without the signal getting hopelessly mangled along the way.
- **The QKV projection** is just three linear layers fused into one big matmul. It produces three tensors — Q (queries), K (keys), V (values) — each of shape `[N × d]`, by applying three different weight matrices to the input.
- **Multi-head attention** is the only step that lets information flow between tokens. It's the main event — §4 walks through what it actually computes, and §5 explains why it's "multi-head."
- **The output projection** is a final linear layer that mixes the attention output back into something the residual `+` can absorb.
- **FFN-up** and **FFN-down** are two linear layers with a nonlinearity in between. Together they widen each token's `d`-dim representation to `4d`, run a per-element nonlinearity, and pull it back to `d`. No mixing across tokens — every token is processed on its own.

Same shape in, same shape out — the §2 mantra. Stack many of these and you have the body of the model.

---

## 4. What attention actually does

We've said "attention mixes across positions" several times now without saying *how*. Let's fix that.

At every position, the model produces three vectors from that position's `[d]`-wide row:

- a **query** `Q` — *"what am I looking for?"*
- a **key** `K` — *"here's what I have to offer"*
- a **value** `V` — *"if you decide you care about me, here's the actual content I want to pass along"*

(That's exactly what the QKV projection does — three linear layers, one each for Q, K, V, fused into one matmul.)

To update position `i`'s row, the model does three things:

1. **Compute scores.** It compares `i`'s query against *every* position's key, with a dot product. A bigger dot product = "those two vectors point in similar directions" = "this position is interesting to position `i`." A smaller (or negative) dot product = "not interesting." So we end up with a list of `N` scores — one per position.
2. **Turn scores into weights.** Run those scores through a softmax to get **attention weights** — all positive, all summing to 1. High weight on position `j` means *"i cares a lot about j;"* low weight means *"i basically ignores j."*
3. **Take a weighted average of values.** Compute a weighted sum of every position's value vector, using those weights. That sum is what gets written back as `i`'s updated representation.

In one sentence: position `i`'s new row is a weighted average of every position's value vector, where the weights are decided by how well `i`'s query matched each one's key.

That's it — that's the entire mechanical content of attention. Everything else in the block (the LayerNorms, the FFN, the residuals) is supporting infrastructure for *this one operation*. It's also the **only step in the entire model** that lets information flow between tokens. Take attention away and the model can't tell that "fox" and "the" are part of the same sentence.

We'll add two more details to this picture:

- **§5 — Heads.** Attention isn't run once on the full `d`-wide features — it's run multiple times in parallel on different slices of the features.
- **§6 — Causal mask.** Position `i` isn't actually allowed to attend to *every* position. It can only look at positions `j ≤ i`. We'll explain why.

The FFN sub-layer, by comparison, is much simpler: every row gets passed through the same two linear layers and a nonlinearity, independently of every other row. No mixing across positions there — that's the attention sub-layer's job.

So the rhythm of every transformer block is: **positions mix (attention), then features mix (FFN).** Repeated `L` times.

---

## 5. Heads

Here's a small thing about attention that turns out to matter a lot: it's not run *once* on the full `d`-dim features, it's run `h` times in parallel on different slices of the features.

After the QKV projection produces Q, K, V each of shape `[N × d]`, we *reshape* each one along the feature dimension into `h` groups of width `d_head = d / h`. Each group is one **head**. Each head runs the §4 attention computation on its own slice — its own queries, its own keys, its own values. Their outputs are concatenated back into `[N × d]` and fed into the output projection.

```
                                    reshape                  per head                      concat
    [N × d]   ───────────────────▶  [N × h × d_head]   ─────────────▶   [N × h × d_head]   ─────▶   [N × d]
       Q, K, V                       h heads of d_head                  h attention                  to output
                                                                         outputs                     projection
```

Real models pick `h` and `d_head` to multiply back to `d` — typically a few dozen heads, each one a hundred-something wide.

The model-design intuition: different heads can learn to pay attention to different *kinds* of things. Some end up tracking short-range syntactic relationships ("which word does this pronoun refer to?"). Others track longer-range patterns. Multiple heads = multiple "perspectives" on what to attend to.

The systems intuition we'll need later is more brutal: **heads are independent.** Head 0 doesn't talk to head 1 during attention. Each one runs its own little attention computation on its own slice of the features and produces its own output. That independence is just a property of how the model is built — but it'll turn out to be a lifesaver every time we want to split work across hardware.

---

## 6. The causal mask

Inside attention, there's one more rule we haven't mentioned, and it's essential: when a token at position `i` attends, it can only see positions `j ≤ i`. Positions `j > i` are masked out — their attention scores are forced to `−∞` before softmax, which makes their post-softmax weights exactly zero, which means they contribute nothing to position `i`'s output.

Why this rule exists comes from training. The model is trained one *next token* at a time: feed in a sequence, ask the model to predict each next token from everything that came before it. If position `i` were allowed to peek at position `i+1` during attention, it would be allowed to *cheat* by reading the answer. The mask is what enforces "no peeking ahead."

The mask has a second job, too — and it's the one we'll lean on the most. It's what makes the generation loop in Part III well-defined. The token at position `N+1` only depends on tokens 1..N, never the other way around. So we can compute new tokens in order, one at a time, without ever having to revise an earlier one. That property is what makes "generate a long answer one token at a time" work at all.

---

## 7. The whole block in one picture

We've now opened up every piece of a transformer block — the two halves (§3), attention's Q/K/V mechanism (§4), the split into heads (§5), the causal mask (§6). Here's the full picture in one trace, with the tensor shape labelled at every step.

Skim it once for the overall flow, then come back to it whenever something later in the series references *"the `[h × N × N]` score matrix"* or *"the reshape into heads"* — this diagram is the shape you're being asked to picture.

<svg viewBox="0 0 720 1240" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;font-family:system-ui,sans-serif;display:block;margin:1.5rem auto">
  <defs>
    <marker id="arr-block" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
      <path d="M 0 0 L 8 4.5 L 0 9 Z" fill="currentColor"/>
    </marker>
  </defs>
  <text x="360" y="24" text-anchor="middle" font-size="14" fill="currentColor" font-weight="600">inside one block — every operation, every shape</text>

  <rect x="280" y="50" width="160" height="36" fill="rgba(74,144,226,0.20)" stroke="#4a90e2" stroke-width="1.5"/>
  <text x="360" y="73" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">input</text>
  <text x="455" y="72" font-size="11" fill="currentColor" opacity="0.7" font-family="ui-monospace,monospace">[N × d]</text>

  <line x1="360" y1="86" x2="360" y2="102" stroke="currentColor" stroke-width="1.5"/>
  <circle cx="360" cy="106" r="3.5" fill="currentColor"/>
  <line x1="360" y1="110" x2="360" y2="130" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-block)"/>
  <path d="M 360 106 L 220 106 L 220 755 L 349 755" fill="none" stroke="currentColor" stroke-width="1.5" stroke-dasharray="4 4" marker-end="url(#arr-block)"/>
  <text x="170" y="400" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.65" font-style="italic">residual</text>

  <rect x="280" y="135" width="160" height="32" fill="rgba(245,166,35,0.18)" stroke="#f5a623" stroke-width="1.5"/>
  <text x="360" y="156" text-anchor="middle" font-size="12" fill="currentColor">LayerNorm 1</text>
  <line x1="360" y1="167" x2="360" y2="190" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-block)"/>
  <text x="455" y="183" font-size="11" fill="currentColor" opacity="0.7" font-family="ui-monospace,monospace">[N × d]</text>

  <rect x="280" y="195" width="160" height="32" fill="rgba(245,166,35,0.18)" stroke="#f5a623" stroke-width="1.5"/>
  <text x="360" y="216" text-anchor="middle" font-size="12" fill="currentColor">QKV projection</text>
  <line x1="360" y1="227" x2="360" y2="250" stroke="currentColor" stroke-width="1.5"/>
  <line x1="280" y1="250" x2="440" y2="250" stroke="currentColor" stroke-width="1.5"/>
  <line x1="280" y1="250" x2="280" y2="276" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-block)"/>
  <line x1="360" y1="250" x2="360" y2="276" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-block)"/>
  <line x1="440" y1="250" x2="440" y2="276" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-block)"/>

  <rect x="252" y="280" width="56" height="28" fill="rgba(120,180,140,0.20)" stroke="#78b48c" stroke-width="1"/>
  <text x="280" y="298" text-anchor="middle" font-size="11" fill="currentColor" font-weight="600">Q</text>
  <rect x="332" y="280" width="56" height="28" fill="rgba(120,180,140,0.20)" stroke="#78b48c" stroke-width="1"/>
  <text x="360" y="298" text-anchor="middle" font-size="11" fill="currentColor" font-weight="600">K</text>
  <rect x="412" y="280" width="56" height="28" fill="rgba(120,180,140,0.20)" stroke="#78b48c" stroke-width="1"/>
  <text x="440" y="298" text-anchor="middle" font-size="11" fill="currentColor" font-weight="600">V</text>
  <text x="495" y="298" font-size="11" fill="currentColor" opacity="0.7" font-family="ui-monospace,monospace">each [N × d]</text>

  <line x1="280" y1="308" x2="280" y2="328" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-block)"/>
  <line x1="360" y1="308" x2="360" y2="328" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-block)"/>
  <line x1="440" y1="308" x2="440" y2="328" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-block)"/>

  <rect x="240" y="332" width="240" height="28" fill="rgba(150,150,150,0.10)" stroke="rgba(150,150,150,0.5)" stroke-width="1" stroke-dasharray="3 3"/>
  <text x="360" y="350" text-anchor="middle" font-size="11" fill="currentColor" font-style="italic">reshape Q, K, V along feature dim into h heads</text>
  <line x1="360" y1="360" x2="360" y2="380" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-block)"/>
  <text x="495" y="378" font-size="11" fill="currentColor" opacity="0.7" font-family="ui-monospace,monospace">each [N × h × d_head]</text>

  <rect x="200" y="385" width="320" height="280" fill="rgba(74,144,226,0.05)" stroke="#4a90e2" stroke-width="1.5" stroke-dasharray="6 4"/>
  <text x="360" y="403" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">multi-head attention (per head, in parallel)</text>

  <rect x="240" y="415" width="240" height="32" fill="rgba(245,166,35,0.18)" stroke="#f5a623" stroke-width="1.5"/>
  <text x="360" y="436" text-anchor="middle" font-size="12" fill="currentColor">Q · Kᵀ / √d_head</text>
  <line x1="360" y1="447" x2="360" y2="467" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-block)"/>
  <text x="495" y="463" font-size="11" fill="currentColor" opacity="0.7" font-family="ui-monospace,monospace">[h × N × N] scores</text>

  <rect x="240" y="471" width="240" height="32" fill="rgba(245,166,35,0.18)" stroke="#f5a623" stroke-width="1.5"/>
  <text x="360" y="492" text-anchor="middle" font-size="12" fill="currentColor">+ causal mask (future → −∞)</text>
  <line x1="360" y1="503" x2="360" y2="523" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-block)"/>
  <text x="495" y="519" font-size="11" fill="currentColor" opacity="0.7" font-family="ui-monospace,monospace">[h × N × N]</text>

  <rect x="240" y="527" width="240" height="32" fill="rgba(245,166,35,0.18)" stroke="#f5a623" stroke-width="1.5"/>
  <text x="360" y="548" text-anchor="middle" font-size="12" fill="currentColor">softmax (along last dim)</text>
  <line x1="360" y1="559" x2="360" y2="579" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-block)"/>
  <text x="495" y="575" font-size="11" fill="currentColor" opacity="0.7" font-family="ui-monospace,monospace">[h × N × N] weights</text>

  <rect x="240" y="583" width="240" height="32" fill="rgba(245,166,35,0.18)" stroke="#f5a623" stroke-width="1.5"/>
  <text x="360" y="604" text-anchor="middle" font-size="12" fill="currentColor">weights · V</text>
  <line x1="360" y1="615" x2="360" y2="635" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-block)"/>
  <text x="495" y="631" font-size="11" fill="currentColor" opacity="0.7" font-family="ui-monospace,monospace">[N × h × d_head]</text>

  <rect x="240" y="639" width="240" height="22" fill="rgba(150,150,150,0.10)" stroke="rgba(150,150,150,0.5)" stroke-width="1" stroke-dasharray="3 3"/>
  <text x="360" y="654" text-anchor="middle" font-size="11" fill="currentColor" font-style="italic">concat heads back into [N × d]</text>

  <line x1="360" y1="665" x2="360" y2="690" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-block)"/>
  <text x="455" y="683" font-size="11" fill="currentColor" opacity="0.7" font-family="ui-monospace,monospace">[N × d]</text>

  <rect x="280" y="694" width="160" height="32" fill="rgba(245,166,35,0.18)" stroke="#f5a623" stroke-width="1.5"/>
  <text x="360" y="715" text-anchor="middle" font-size="12" fill="currentColor">output projection</text>
  <line x1="360" y1="726" x2="360" y2="744" stroke="currentColor" stroke-width="1.5"/>

  <circle cx="360" cy="755" r="11" fill="rgba(150,150,150,0.18)" stroke="currentColor" stroke-width="1.5"/>
  <text x="360" y="760" text-anchor="middle" font-size="14" fill="currentColor" font-weight="600">+</text>

  <line x1="360" y1="766" x2="360" y2="790" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-block)"/>
  <text x="455" y="785" font-size="11" fill="currentColor" opacity="0.7" font-family="ui-monospace,monospace">[N × d]</text>

  <circle cx="360" cy="795" r="3.5" fill="currentColor"/>
  <path d="M 360 795 L 220 795 L 220 1095 L 349 1095" fill="none" stroke="currentColor" stroke-width="1.5" stroke-dasharray="4 4" marker-end="url(#arr-block)"/>
  <text x="170" y="940" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.65" font-style="italic">residual</text>

  <line x1="360" y1="799" x2="360" y2="815" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-block)"/>

  <rect x="280" y="819" width="160" height="32" fill="rgba(245,166,35,0.18)" stroke="#f5a623" stroke-width="1.5"/>
  <text x="360" y="840" text-anchor="middle" font-size="12" fill="currentColor">LayerNorm 2</text>
  <line x1="360" y1="851" x2="360" y2="870" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-block)"/>
  <text x="455" y="867" font-size="11" fill="currentColor" opacity="0.7" font-family="ui-monospace,monospace">[N × d]</text>

  <rect x="280" y="874" width="160" height="32" fill="rgba(245,166,35,0.18)" stroke="#f5a623" stroke-width="1.5"/>
  <text x="360" y="895" text-anchor="middle" font-size="12" fill="currentColor">FFN-up (d → 4d)</text>
  <line x1="360" y1="906" x2="360" y2="925" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-block)"/>
  <text x="455" y="922" font-size="11" fill="currentColor" opacity="0.7" font-family="ui-monospace,monospace">[N × 4d]</text>

  <rect x="280" y="929" width="160" height="32" fill="rgba(245,166,35,0.18)" stroke="#f5a623" stroke-width="1.5"/>
  <text x="360" y="950" text-anchor="middle" font-size="12" fill="currentColor">activation (GeLU)</text>
  <line x1="360" y1="961" x2="360" y2="980" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-block)"/>
  <text x="455" y="977" font-size="11" fill="currentColor" opacity="0.7" font-family="ui-monospace,monospace">[N × 4d]</text>

  <rect x="280" y="984" width="160" height="32" fill="rgba(245,166,35,0.18)" stroke="#f5a623" stroke-width="1.5"/>
  <text x="360" y="1005" text-anchor="middle" font-size="12" fill="currentColor">FFN-down (4d → d)</text>
  <line x1="360" y1="1016" x2="360" y2="1084" stroke="currentColor" stroke-width="1.5"/>

  <circle cx="360" cy="1095" r="11" fill="rgba(150,150,150,0.18)" stroke="currentColor" stroke-width="1.5"/>
  <text x="360" y="1100" text-anchor="middle" font-size="14" fill="currentColor" font-weight="600">+</text>

  <line x1="360" y1="1106" x2="360" y2="1130" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-block)"/>

  <rect x="280" y="1135" width="160" height="36" fill="rgba(74,144,226,0.20)" stroke="#4a90e2" stroke-width="1.5"/>
  <text x="360" y="1158" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">output</text>
  <text x="455" y="1157" font-size="11" fill="currentColor" opacity="0.7" font-family="ui-monospace,monospace">[N × d]</text>
</svg>

Three things worth pausing on:

- **Shapes start and end at `[N × d]`** — the §2 mantra. Inside one block, the tensor briefly takes other shapes (`[N × 4d]` in the middle of the FFN, `[h × N × N]` for the attention scores) — but those are *transient*. The block always returns to `[N × d]` so the next block can consume it.
- **The `[h × N × N]` score matrix is the one that surprises.** Its size scales with the **square of the sequence length**. Harmless when `N` is small, awkward when `N` is large — that's where the cost of long sequences will eventually bite. Worth noticing now; future articles will come back to it.
- **Each residual `+` re-injects the input** of that half back onto the output. So each half is computing a *delta*, not a replacement. That's why we can stack many blocks without the signal collapsing.

---

## Part III — Using the model to generate

## 8. One forward gives you one token

The model in §1 takes a sequence of length `N` and returns a probability distribution over what the next token should be. **One** token. Not a whole sentence, not even a phrase — a single next-token guess.

But we're used to LLMs producing long replies. How does a one-token-at-a-time machine produce a paragraph? Exactly how you'd guess: by running over and over and feeding its own output back in.

Concretely:

1. Start with the prompt — a sequence of length `N`.
2. Run a forward pass on it. You get a distribution over what token `N+1` should be.
3. Sample from that distribution (or just take the most likely token, "argmax"). You now have a token at position `N+1`.
4. Append it to the sequence. The sequence is now length `N+1`.
5. Run another forward pass on the *full* `(N+1)`-long sequence. You get a distribution over token `N+2`.
6. Sample. Append. Sequence is now length `N+2`.
7. Repeat until either the model samples a special **end-of-sequence** token (it has been trained to emit one when it thinks the response is complete) or you hit a length cap you've imposed.

A picture of the loop:

<svg viewBox="0 0 760 360" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;font-family:system-ui,sans-serif;display:block;margin:1.5rem auto">
  <defs>
    <marker id="arr-loop" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
      <path d="M 0 0 L 8 4.5 L 0 9 Z" fill="currentColor"/>
    </marker>
  </defs>
  <text x="380" y="26" text-anchor="middle" font-size="14" fill="currentColor" font-weight="600">generation loop: sample, append, repeat</text>

  <rect x="40" y="70" width="120" height="44" fill="rgba(74,144,226,0.20)" stroke="#4a90e2" stroke-width="1.5"/>
  <text x="100" y="90" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">prompt</text>
  <text x="100" y="106" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.7" font-family="ui-monospace,monospace">length N</text>

  <line x1="160" y1="92" x2="200" y2="92" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-loop)"/>

  <rect x="200" y="70" width="100" height="44" fill="rgba(120,180,140,0.20)" stroke="#78b48c" stroke-width="1.5"/>
  <text x="250" y="90" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">forward</text>
  <text x="250" y="106" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.7">on length N</text>

  <line x1="300" y1="92" x2="340" y2="92" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-loop)"/>

  <rect x="340" y="70" width="120" height="44" fill="rgba(245,166,35,0.20)" stroke="#f5a623" stroke-width="1.5"/>
  <text x="400" y="90" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">sample token N+1</text>
  <text x="400" y="106" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.7">from last-row softmax</text>

  <line x1="460" y1="92" x2="500" y2="92" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-loop)"/>

  <rect x="500" y="70" width="120" height="44" fill="rgba(74,144,226,0.20)" stroke="#4a90e2" stroke-width="1.5"/>
  <text x="560" y="90" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">sequence</text>
  <text x="560" y="106" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.7" font-family="ui-monospace,monospace">length N+1</text>

  <path d="M 560 114 L 560 140 L 100 140 L 100 170" fill="none" stroke="currentColor" stroke-width="1.5" stroke-dasharray="4 4" marker-end="url(#arr-loop)"/>
  <text x="330" y="155" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.7" font-style="italic">feed the appended sequence back in</text>

  <rect x="40" y="180" width="120" height="44" fill="rgba(74,144,226,0.20)" stroke="#4a90e2" stroke-width="1.5"/>
  <text x="100" y="200" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">sequence</text>
  <text x="100" y="216" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.7" font-family="ui-monospace,monospace">length N+1</text>

  <line x1="160" y1="202" x2="200" y2="202" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-loop)"/>

  <rect x="200" y="180" width="100" height="44" fill="rgba(120,180,140,0.20)" stroke="#78b48c" stroke-width="1.5"/>
  <text x="250" y="200" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">forward</text>
  <text x="250" y="216" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.7">on length N+1</text>

  <line x1="300" y1="202" x2="340" y2="202" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-loop)"/>

  <rect x="340" y="180" width="120" height="44" fill="rgba(245,166,35,0.20)" stroke="#f5a623" stroke-width="1.5"/>
  <text x="400" y="200" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">sample token N+2</text>

  <line x1="460" y1="202" x2="500" y2="202" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-loop)"/>

  <rect x="500" y="180" width="120" height="44" fill="rgba(74,144,226,0.20)" stroke="#4a90e2" stroke-width="1.5"/>
  <text x="560" y="200" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">sequence</text>
  <text x="560" y="216" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.7" font-family="ui-monospace,monospace">length N+2</text>

  <text x="380" y="260" text-anchor="middle" font-size="14" fill="currentColor" opacity="0.55">⋮</text>

  <rect x="220" y="288" width="320" height="44" fill="rgba(150,150,150,0.10)" stroke="rgba(150,150,150,0.6)" stroke-width="1.5" stroke-dasharray="4 4"/>
  <text x="380" y="308" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">until model emits end-of-sequence</text>
  <text x="380" y="324" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.7">or until a length cap is hit</text>
</svg>

That is the whole generation procedure, mathematically. Every output token from any LLM-based system you've ever used was produced by a loop that looks like this.

---

## 9. The first uncomfortable observation

Walk through the cost of generating `K` new tokens from a prompt of length `N`.

- Forward 1 runs on the prompt: length `N`.
- Forward 2 runs on prompt + 1 new token: length `N+1`.
- Forward 3: length `N+2`.
- …
- Forward `K`: length `N+K−1`.

Every forward repeats almost everything the previous forward already did. The first `N` tokens of forward 2's input are *identical* to forward 1's input — the model nevertheless runs every block on every position from scratch, as if it had never seen them before.

If you total it up, the work scales like roughly `(N + K)² / 2` — quadratic in the eventual sequence length. And most of that work is *recomputing things that haven't changed*. A new token added at the end of the sequence doesn't change any of the earlier tokens' representations. The earlier tokens are still the same prompt and the same few sampled tokens that came before this one. Nothing about them needs to be redone.

So an obvious question hangs in the air: **is all that recomputation actually necessary?** Clearly not. But avoiding it isn't free either — it means we'd have to keep some intermediate state around between forwards. Which raises its own questions: what state, exactly? Where do we put it? How big does it get? How does it grow as the conversation grows?

That kind of question is exactly what this series picks at later on.

---

## 10. The map of questions

With §1's stack, §2's stream-processor pattern, §4's attention mechanism, §7's full-block picture, and §8's loop in hand, a lot of practical questions about *running* an LLM follow naturally — and most of them don't have obvious answers. Each of these is the seed of an article down the line.

- **The model itself can be huge.** Once you stack enough blocks (large `L`) at a wide enough `d`, the weights alone are too big to fit on a single GPU. How do we split one forward pass across multiple GPUs? *(Articles 02 and 03.)*
- **Many users at once.** A real serving system has many concurrent prompts of different lengths. How do they share one forward pass without padding waste? *(Article 04.)*
- **Generation is repetitive.** As §9 hinted, the naive loop redoes most of its work. What state could we keep around to avoid that, and what does keeping it cost us?
- **Different users finish at different times.** When some users are still generating their thousandth output token and others are just starting their first, how does the engine keep everyone moving without leaving the GPU idle?
- **Some prompts are huge.** What if the prompt itself is so long that a single forward pass on it takes forever, or runs out of memory? Can we process it in pieces?
- **The "remembered state" has to live somewhere.** Whatever we keep around to avoid §9's recomputation needs to be allocated, stored, freed, and shared across many concurrent requests. How do we manage that, and what does "out of memory" mean in this setting?
- **Attention's `[h × N × N]` matrix gets expensive.** Look back at §7: that score matrix scales with the **square of the sequence length**. For long sequences it becomes the bottleneck — and naively it has to be materialized in memory. Can we be smarter?
- **Generating one new token feels nothing like processing a long prompt.** §9's per-call cost is *very* different depending on whether you're processing a long input or appending one extra output token. Maybe the engine should treat those two cases differently — or even split them across different machines.

The model in §1–§7 is what all of these questions are *about*. The loop in §8 is what they're trying to make fast and efficient. The rest of the series picks them off one at a time.
