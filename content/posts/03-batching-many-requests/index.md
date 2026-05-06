---
title: "How to Batch Many Requests Through One Forward Pass"
date: 2026-05-03T00:00:00+00:00
draft: false
summary: "Many users hit the model at once with different-length prompts. Walk through one transformer block on a flat multi-request tensor and see which layers batch for free and which need a real fix — and whether TP has to change."
description: "How to batch many concurrent prefill requests through a TP-parallelized transformer. Walk a full block on a flattened multi-request tensor and watch where batching is free vs. where it isn't."
tags: ["batching", "varlen-attention", "selective-batching", "llm-serving", "mental-model"]
series: ["llm-stories"]
showToc: true
TocOpen: false
weight: 4
---

[Article 02](/llm_stories/posts/02-tp-through-a-full-block/) left us with one transformer block running on two GPUs in two all-reduces per layer. But a real serving system has many users hitting the model concurrently — and their prompts are all different lengths. A 50-token *"what time is it"* sits next to a 5,000-token essay draft.

Two questions to chase through this article:

1. **How do we batch variable-length requests through one forward pass efficiently?** The naive answer — pad everything to the longest prompt and run it as a fixed batch — wastes a lot of compute on the short ones. There has to be a smarter way.
2. **Does TP have to know any of this is happening?** Or can the batching trick and the model-splitting story stay independent of each other?

We'll answer both by carrying Article 02's setup forward and watching what each layer does when more than one request flows through it.

---

## 1. Setup

Same numbers as Article 02:

| | value |
|---|---|
| GPUs | 2 (TP=2) |
| layers | 8 |
| `d` (model dim) | 512 |
| `h` (heads) | 8, four per GPU |
| `d_head` | 64 |
| `k = h · d_head` | 512 |
| FFN hidden | `4d` = 2048 |

Two example requests for the running discussion: **request A** of length 10, **request B** of length 30.

Three explicit assumptions we'll hold this article to:

- **Prefill only.** We're computing the forward pass over each request's prompt. No token-by-token decoding yet — that's Article 04.
- **Each request fits in one batch.** A batch holds ≥1 *whole* requests, never a fraction. Article 05 will relax this with chunked prefill.
- **No KV cache yet.** The KV cache is what lets a *later* token attend back to earlier ones during decode. In a prefill-only world we just compute outputs and ship them; there's nothing to cache for later. KV cache enters with Article 04.

These keep the spatial story clean. The *temporal* story (continuous batching across iterations) is its own article.

---

## 2. One request first: `N` is just a tensor dimension

Before two requests, recall what one request looks like under the v2 pattern from Article 02. A single prefill of length `N` flows through the block as `[N × 512]`. From the trace there:

- 8 layers × 2 all-reduces per layer = **16 all-reduces** per forward pass.
- Every all-reduce moves a `[N × 512]` tensor across GPUs.

The thing worth pausing on: **`N` only appears in tensor shapes, never in comm step counts.** Whether `N=10` or `N=10,000`, you do exactly 16 all-reduces. They just carry more or fewer bytes per step.

So adding more tokens to one request is "free" comm-wise — the per-byte cost scales linearly with token count, but you're not paying for *additional* sync events.

That's a nice property. The next question is whether it survives when the extra tokens come from *different requests*.

---

## 3. The naive answer and the smarter idea

**Naive: pad to max length.** Stack A and B as a `[2 × 30 × 512]` batch. Request A gets 20 padding tokens that the model still computes against. Linear-layer waste is mild (the matmul is bigger by 2×). Attention waste is severe — each request's attention is `O(L²)` work, so A's attention does `30² = 900` operations per head per layer instead of the `10² = 100` it actually needs. **9× too much work for A alone**, and the padded tokens contribute nothing to the output you care about.

**Smarter: flatten.** Concatenate A and B's tokens into one tensor of shape `[(10+30) × 512] = [40 × 512]`. No padding, no batch dimension — just a flat stream of tokens.

The question this raises: **does every step of the forward pass do the right thing on a flattened tensor of mixed-request tokens?** Some steps clearly will. Some will need thought. Let's walk the whole block and see.

---

## 4. The whole block, step by step

Start at the input `[40 × 512]` and trace every step of one block. For each one, ask: does it compute the right answer when its input contains tokens from multiple requests?

| Step | What it does | On `[40 × 512]`? |
|---|---|---|
| LayerNorm | normalizes each row independently | ✓ trivially fine |
| QKV proj (linear) | matmul against shared `W` | needs analysis |
| Attention | sequence-mixing per request | needs analysis |
| Output proj (linear) | matmul against shared `W` | needs analysis |
| Residual add | per-row sum | ✓ trivially fine |
| LayerNorm | normalizes each row independently | ✓ trivially fine |
| FFN-up (linear) | matmul against shared `W` | needs analysis |
| Activation (GeLU) | per-element non-linearity | ✓ trivially fine |
| FFN-down (linear) | matmul against shared `W` | needs analysis |
| Residual add | per-row sum | ✓ trivially fine |

Half the steps check off immediately. **Pointwise operations** — LayerNorm, GeLU, residual adds — process each row independently. Whether row `i` belongs to request A or request B is invisible to them. They're per-token and they don't mix. So they batch for free.

That leaves five steps that need closer examination: four linear matmuls (QKV proj, output proj, FFN-up, FFN-down) and one attention block.

But the convenience is: **all four linear matmuls have the same structure** — `Y = X @ W` where `W` is shared across all rows of `X`. So once we understand how *one* linear behaves under flattened batching, all four follow. And there's only one attention block per layer.

The whole batching problem reduces to two questions:

1. **Does a linear layer compute the right answer on `[40 × 512]`?**
2. **Does attention compute the right answer on `[40 × 512]`?**

§5 takes the linear question. §6 takes the attention question. Once those two are settled, the whole block is settled.

---

## 5. Linear layers: the easy half

Go back to how [Article 01](/llm_stories/posts/01-tensor-parallelism-mental-model/) framed a linear layer. The weight matrix is a row of little **feature extractors** — each `fx` is its own opaque function that takes one token's `d`-wide feature vector and returns one number. A linear layer with `k` outputs is just `k` of those extractors running side by side on the same token.

```
token  ⇒  [ fx1   fx2   fx3   ...   fxk ]   ⇒   [ fx1(token), fx2(token), ..., fxk(token) ]
```

The thing worth pausing on: every `fx` looks at **one token** and returns **one number**. It doesn't peek at the next token. It doesn't peek at the previous token. It has no concept of what conversation the token came from. There is **no place in the math where request boundaries could enter**, because the math only ever sees one token at a time.

So when we hand the layer a flat tensor `[40 × 512]` — 40 tokens stacked — it just runs every `fx` on every token. 40 tokens, `k` extractors each, fills out a `[40 × k]` output. The fact that the first 10 rows are request A and the last 30 are request B is **invisible** to the operation; we never even had a chance to mix them.

That's the entire reason linear layers batch trivially. They're not "magically batchable" — they were already per-token. We're just running more of them.

**Under TP=2:** unchanged from Article 02. The `fx`es are still split across GPUs, with each GPU owning half:

- G1 runs heads 1–4's `fx`es on `[40 × 512]` → `[40 × 768]`
- G2 runs heads 5–8's `fx`es on `[40 × 512]` → `[40 × 768]`

The all-reduce shape grew from `[N × 512]` to `[40 × 512]`, but the **count of all-reduces is unchanged**. Same comm pattern, more bytes per step.

And since the same argument applies to all four linear matmuls in the block — QKV, output, FFN-up, FFN-down — **all the linears are now solved.** One step left.

---

## 6. Attention: the hard half

Why is attention different? Because attention is **sequence-mixing**. Each token's output depends on *all* tokens in its sequence, not just its own row:

```
out[i, :] = softmax( Q[i, :] @ K.T / √d_head ) @ V
```

That `K.T` and `V` reach across the whole sequence. If `K` and `V` come from a tensor that contains tokens from both A and B, then by default token `i` in A would attend to tokens of B — and vice versa. The math would *technically* run, but the answer would be wrong: A's output would be mixed with B's keys and values, which is not what the model was trained to produce.

So we need a way to keep request A's attention strictly within A's tokens, and B's strictly within B's, while still sharing the underlying flat tensor.

### 6.1 The naive approach — compute, then mask

The most direct fix: compute the full `[40 × 40]` attention matrix as if all 40 tokens were one sequence, then mask out the cross-request entries (set them to `-∞` before softmax so they contribute nothing).

The flat token buffer looks like this:

<svg viewBox="0 0 500 560" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;font-family:system-ui,sans-serif;display:block;margin:1.5rem auto">
  <text x="250" y="25" text-anchor="middle" font-size="14" fill="currentColor" font-weight="600">flat token tensor: [40 tokens × 512]</text>
  <text x="250" y="46" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.65">each row is one token of d=512 features</text>
  <rect x="180" y="70" width="140" height="110" fill="rgba(74,144,226,0.25)" stroke="#4a90e2" stroke-width="2"/>
  <text x="250" y="118" text-anchor="middle" font-size="14" fill="currentColor" font-weight="600">request A</text>
  <text x="250" y="138" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.75">10 tokens</text>
  <rect x="180" y="180" width="140" height="330" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="2"/>
  <text x="250" y="338" text-anchor="middle" font-size="14" fill="currentColor" font-weight="600">request B</text>
  <text x="250" y="358" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.75">30 tokens</text>
  <text x="335" y="74" font-size="11" fill="currentColor" opacity="0.7">row 0</text>
  <text x="335" y="184" font-size="11" fill="currentColor" opacity="0.7">row 10</text>
  <text x="335" y="514" font-size="11" fill="currentColor" opacity="0.7">row 40</text>
  <text x="250" y="540" text-anchor="middle" font-size="13" fill="currentColor" font-family="ui-monospace,monospace">cu_seqlens = [0, 10, 40]</text>
</svg>

And the full attention matrix, with cross-request blocks masked:

<svg viewBox="0 0 700 680" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;font-family:system-ui,sans-serif;display:block;margin:1.5rem auto">
  <defs>
    <pattern id="hatch-naive" patternUnits="userSpaceOnUse" width="10" height="10" patternTransform="rotate(45)">
      <rect width="10" height="10" fill="rgba(150,150,150,0.06)"/>
      <line x1="0" y1="0" x2="0" y2="10" stroke="rgba(150,150,150,0.45)" stroke-width="2"/>
    </pattern>
  </defs>
  <text x="350" y="25" text-anchor="middle" font-size="14" fill="currentColor" font-weight="600">naive: compute full 40×40, mask cross-request blocks</text>
  <text x="200" y="115" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.75">keys 0..9</text>
  <text x="440" y="115" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.75">keys 10..39</text>
  <text transform="translate(115,200) rotate(-90)" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.75">queries 0..9</text>
  <text transform="translate(115,440) rotate(-90)" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.75">queries 10..39</text>
  <rect x="140" y="140" width="120" height="120" fill="rgba(74,144,226,0.25)" stroke="#4a90e2" stroke-width="2"/>
  <text x="200" y="195" text-anchor="middle" font-size="14" fill="currentColor" font-weight="600">A → A</text>
  <text x="200" y="215" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.75">10 × 10</text>
  <rect x="260" y="140" width="360" height="120" fill="url(#hatch-naive)" stroke="rgba(150,150,150,0.5)" stroke-width="1.5" stroke-dasharray="4 4"/>
  <text x="440" y="195" text-anchor="middle" font-size="13" fill="currentColor" opacity="0.7">masked to −∞</text>
  <text x="440" y="215" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.6">10 × 30</text>
  <rect x="140" y="260" width="120" height="360" fill="url(#hatch-naive)" stroke="rgba(150,150,150,0.5)" stroke-width="1.5" stroke-dasharray="4 4"/>
  <text x="200" y="435" text-anchor="middle" font-size="13" fill="currentColor" opacity="0.7">masked</text>
  <text x="200" y="455" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.6">30 × 10</text>
  <rect x="260" y="260" width="360" height="360" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="2"/>
  <text x="440" y="435" text-anchor="middle" font-size="14" fill="currentColor" font-weight="600">B → B</text>
  <text x="440" y="455" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.75">30 × 30</text>
  <text x="350" y="660" text-anchor="middle" font-size="12" fill="currentColor" font-family="ui-monospace,monospace">computed: 1600   useful: 1000   wasted: 600</text>
</svg>

This works, but it's wasteful. The off-diagonal blocks — `10 × 30` and `30 × 10`, totaling 600 entries — are computed and immediately discarded. With more concurrent requests it gets worse: with `R` requests of equal length `L`, you compute `(RL)²` but only need `R · L²`. Cross-request work scales as `R²` while useful work scales only as `R`. Untenable for serving systems where R can easily reach into the hundreds.

### 6.2 The varlen idea — skip, don't mask

Instead of computing-then-masking, compute *only* the diagonal blocks. Loop over requests, and for each one run normal attention on its slice of the flat buffer:

<svg viewBox="0 0 700 680" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;font-family:system-ui,sans-serif;display:block;margin:1.5rem auto">
  <text x="350" y="25" text-anchor="middle" font-size="14" fill="currentColor" font-weight="600">varlen: compute only the diagonal blocks</text>
  <text x="200" y="115" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.75">keys 0..9</text>
  <text x="440" y="115" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.75">keys 10..39</text>
  <text transform="translate(115,200) rotate(-90)" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.75">queries 0..9</text>
  <text transform="translate(115,440) rotate(-90)" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.75">queries 10..39</text>
  <rect x="140" y="140" width="120" height="120" fill="rgba(74,144,226,0.25)" stroke="#4a90e2" stroke-width="2"/>
  <text x="200" y="195" text-anchor="middle" font-size="14" fill="currentColor" font-weight="600">A → A</text>
  <text x="200" y="215" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.75">10 × 10</text>
  <rect x="260" y="260" width="360" height="360" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="2"/>
  <text x="440" y="435" text-anchor="middle" font-size="14" fill="currentColor" font-weight="600">B → B</text>
  <text x="440" y="455" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.75">30 × 30</text>
  <text x="440" y="200" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.4" font-style="italic">(not computed)</text>
  <text x="200" y="440" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.4" font-style="italic">(not computed)</text>
  <text x="350" y="660" text-anchor="middle" font-size="12" fill="currentColor" font-family="ui-monospace,monospace">computed: 1000 — no waste</text>
</svg>

This is the **variable-length attention** kernel — varlen for short. It takes the flat tensor plus an array of request boundaries (`cu_seqlens`, the cumulative sequence lengths) and walks request-by-request:

```python
# cu_seqlens = [0, 10, 40]   # request A spans [0,10), B spans [10,40)
for i in range(num_requests):
    s, e = cu_seqlens[i], cu_seqlens[i+1]
    Q_i = Q[s:e]
    K_i = K[s:e]
    V_i = V[s:e]
    scores_i = (Q_i @ K_i.T) / sqrt(d_head)        # L_i × L_i
    probs_i  = softmax(scores_i + causal_mask_i)
    out[s:e] = probs_i @ V_i                       # write back into flat buffer
```

Visualizing the loop walking down the flat Q, K, V stacks:

<svg viewBox="0 0 800 580" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;font-family:system-ui,sans-serif;display:block;margin:1.5rem auto">
  <defs>
    <marker id="arrow-blue" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
      <path d="M 0 0 L 8 4.5 L 0 9 Z" fill="#4a90e2"/>
    </marker>
    <marker id="arrow-amber" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
      <path d="M 0 0 L 8 4.5 L 0 9 Z" fill="#f5a623"/>
    </marker>
  </defs>
  <text x="400" y="25" text-anchor="middle" font-size="14" fill="currentColor" font-weight="600">varlen walks the flat Q, K, V stacks request-by-request</text>
  <text x="260" y="68" text-anchor="middle" font-size="13" fill="currentColor" font-weight="600">Q</text>
  <text x="360" y="68" text-anchor="middle" font-size="13" fill="currentColor" font-weight="600">K</text>
  <text x="460" y="68" text-anchor="middle" font-size="13" fill="currentColor" font-weight="600">V</text>
  <rect x="220" y="80" width="80" height="100" fill="rgba(74,144,226,0.4)" stroke="#4a90e2" stroke-width="2.5"/>
  <rect x="320" y="80" width="80" height="100" fill="rgba(74,144,226,0.4)" stroke="#4a90e2" stroke-width="2.5"/>
  <rect x="420" y="80" width="80" height="100" fill="rgba(74,144,226,0.4)" stroke="#4a90e2" stroke-width="2.5"/>
  <text x="260" y="125" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">A</text>
  <text x="260" y="142" text-anchor="middle" font-size="9" fill="currentColor" opacity="0.75">[0:10]</text>
  <text x="360" y="125" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">A</text>
  <text x="360" y="142" text-anchor="middle" font-size="9" fill="currentColor" opacity="0.75">[0:10]</text>
  <text x="460" y="125" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">A</text>
  <text x="460" y="142" text-anchor="middle" font-size="9" fill="currentColor" opacity="0.75">[0:10]</text>
  <rect x="220" y="180" width="80" height="300" fill="rgba(245,166,35,0.4)" stroke="#f5a623" stroke-width="2.5"/>
  <rect x="320" y="180" width="80" height="300" fill="rgba(245,166,35,0.4)" stroke="#f5a623" stroke-width="2.5"/>
  <rect x="420" y="180" width="80" height="300" fill="rgba(245,166,35,0.4)" stroke="#f5a623" stroke-width="2.5"/>
  <text x="260" y="325" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">B</text>
  <text x="260" y="342" text-anchor="middle" font-size="9" fill="currentColor" opacity="0.75">[10:40]</text>
  <text x="360" y="325" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">B</text>
  <text x="360" y="342" text-anchor="middle" font-size="9" fill="currentColor" opacity="0.75">[10:40]</text>
  <text x="460" y="325" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">B</text>
  <text x="460" y="342" text-anchor="middle" font-size="9" fill="currentColor" opacity="0.75">[10:40]</text>
  <text x="40" y="115" font-size="13" fill="#4a90e2" font-weight="700">i = 0</text>
  <text x="40" y="135" font-size="11" fill="currentColor" opacity="0.85">read slice [0:10]</text>
  <line x1="160" y1="130" x2="215" y2="130" stroke="#4a90e2" stroke-width="2" marker-end="url(#arrow-blue)"/>
  <text x="40" y="320" font-size="13" fill="#f5a623" font-weight="700">i = 1</text>
  <text x="40" y="340" font-size="11" fill="currentColor" opacity="0.85">read slice [10:40]</text>
  <line x1="160" y1="335" x2="215" y2="335" stroke="#f5a623" stroke-width="2" marker-end="url(#arrow-amber)"/>
  <text x="530" y="115" font-size="13" fill="#4a90e2" font-weight="600">step 1 — compute</text>
  <text x="530" y="135" font-size="11" fill="currentColor" opacity="0.85" font-family="ui-monospace,monospace">scores = Q_A @ K_A.T</text>
  <text x="530" y="153" font-size="11" fill="currentColor" opacity="0.85" font-family="ui-monospace,monospace">probs  = softmax(scores)</text>
  <text x="530" y="171" font-size="11" fill="currentColor" opacity="0.85" font-family="ui-monospace,monospace">out[0:10] = probs @ V_A</text>
  <text x="530" y="320" font-size="13" fill="#f5a623" font-weight="600">step 2 — compute</text>
  <text x="530" y="340" font-size="11" fill="currentColor" opacity="0.85" font-family="ui-monospace,monospace">scores = Q_B @ K_B.T</text>
  <text x="530" y="358" font-size="11" fill="currentColor" opacity="0.85" font-family="ui-monospace,monospace">probs  = softmax(scores)</text>
  <text x="530" y="376" font-size="11" fill="currentColor" opacity="0.85" font-family="ui-monospace,monospace">out[10:40] = probs @ V_B</text>
  <text x="400" y="525" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.7" font-style="italic">flat Q, K, V tensors in HBM — varlen kernel slices [s:e] for each request, top to bottom</text>
</svg>

Three things to notice:

- The cross-request blocks aren't masked — they're **never computed**. The kernel skips them entirely.
- Each iteration's score matrix is the right size for *that request* — `[L_i × L_i]`, not `[40 × 40]`. So memory for scores stays small.
- The flat output buffer is filled in by writing each request's attention output into its own slice.

`cu_seqlens` is the only piece of metadata the kernel needs to know about requests. Everything else is just slicing the flat tensor.

(In practice the kernel doesn't run this loop in Python — it runs it inside the GPU in one launch, so we don't pay a kernel-launch overhead per request. The mathematical content is identical to the loop above; the optimized kernel just expresses it more efficiently. We'll come back to high-performance attention kernels in a later article.)

### 6.3 Under TP=2

Each GPU still owns its 4 heads from Article 02. The varlen kernel runs on each GPU's local Q, K, V — for *those* heads, for all requests' tokens. G1 doesn't need to know what G2 is doing during attention; G2's heads are G2's problem. The per-head independence we relied on in Article 02 still holds inside this loop. **No new comm.**

So with linears (§5) and attention (§6) both handled, every step of the block is now correctly batched.

---

## 7. Stepping back: TP didn't have to change

A "happy realization" worth pausing on. Look at what TP saw during the entire batched forward pass:

- A tensor of shape `[tokens × hidden]` flowing through the layers.
- Weights split along heads.
- All-reduces on `[tokens × hidden]` partial sums.
- 16 sync events per block, exactly as in Article 02.

**TP never saw a request boundary.** The flat tensor presented itself the same way to TP whether the 40 tokens came from one request or fifty. Request boundaries entered exactly one place — the `cu_seqlens` argument inside the varlen attention kernel — and that argument was used entirely on each GPU's local slice. No comm event involved.

So request batching and TP turn out to be **orthogonal axes that meet only inside the attention kernel**:

- **TP** answers: *how is the model split across GPUs?*
- **Request batching** answers: *how are tokens packed into one forward pass?*

Those questions don't constrain each other. We didn't design for this — it fell out of two facts that were already true:

- Linear layers are per-token (so they don't see request boundaries even on one GPU).
- Multi-head attention's heads are independent (so each GPU's per-head varlen loop never has to talk to other GPUs).

The Article 02 punchline was that multi-head attention was a gift the modelers left for the systems people building TP. Here we see the same gift extended one layer further: the same head independence that makes TP comm-free *also* makes request batching comm-free. Two unrelated tricks compose for free because they were both granted by the same architectural property.

---

## 8. Cost intuitions

A few honest words about where time actually goes once you flatten requests like this.

**Linear layers** look great. One weight matrix is read from HBM, then amortized across all `(N+M)` tokens of the flat tensor. The more tokens you pack in, the closer the GPU runs to its compute peak. This is why aggressive prefill batching is a clear throughput win.

**Attention** is more nuanced. Each request's `Q_i K_i.T` is its own matmul, which means we can't fuse one big GEMM across requests the way linears do. Modern varlen kernels run the request loop *inside* the GPU in one launch, so we don't pay a launch overhead per request. But each request still gets its own attention work proportional to `L_i²`, which means the bottleneck profile depends heavily on the *distribution* of request lengths.

Imagine two batches with the same total token count:

- **1 request × 1,000 tokens** — attention work is `1 × 1000² = 10⁶` per head per layer. The whole square is one block. **Attention dominates** the forward pass.
- **10 requests × 100 tokens each** — attention work is `10 × 100² = 10⁵`. Ten times less. **Linears dominate.**

It's the same picture as the varlen square from §6.2, just at the two extremes. Place all 1,000 tokens along one axis of the attention matrix and only the per-request diagonal blocks ever get computed — everything else is cross-request and skipped:

<svg viewBox="0 0 760 480" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;font-family:system-ui,sans-serif;display:block;margin:1.5rem auto">
  <defs>
    <pattern id="hatch-cost" patternUnits="userSpaceOnUse" width="10" height="10" patternTransform="rotate(45)">
      <rect width="10" height="10" fill="rgba(150,150,150,0.06)"/>
      <line x1="0" y1="0" x2="0" y2="10" stroke="rgba(150,150,150,0.4)" stroke-width="1.5"/>
    </pattern>
  </defs>
  <text x="380" y="25" text-anchor="middle" font-size="14" fill="currentColor" font-weight="600">same 1,000 total tokens, very different attention work</text>
  <text x="380" y="46" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.65">colored = per-request work that's actually computed; hatched = cross-request, skipped by varlen</text>
  <rect x="60" y="80" width="280" height="280" fill="url(#hatch-cost)" stroke="rgba(150,150,150,0.5)" stroke-width="1.5"/>
  <rect x="60" y="80" width="280" height="280" fill="rgba(74,144,226,0.45)" stroke="#4a90e2" stroke-width="1.5"/>
  <text x="200" y="385" text-anchor="middle" font-size="13" fill="currentColor" font-weight="600">1 request × 1,000 tokens</text>
  <text x="200" y="408" text-anchor="middle" font-size="12" fill="currentColor" font-family="ui-monospace,monospace">1 × 1000² = 10⁶</text>
  <text x="200" y="430" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.75">no cross-request waste — attention dominates</text>
  <rect x="440" y="80" width="280" height="280" fill="url(#hatch-cost)" stroke="rgba(150,150,150,0.5)" stroke-width="1.5"/>
  <g fill="rgba(245,166,35,0.5)" stroke="#f5a623" stroke-width="1">
    <rect x="440" y="80"  width="28" height="28"/>
    <rect x="468" y="108" width="28" height="28"/>
    <rect x="496" y="136" width="28" height="28"/>
    <rect x="524" y="164" width="28" height="28"/>
    <rect x="552" y="192" width="28" height="28"/>
    <rect x="580" y="220" width="28" height="28"/>
    <rect x="608" y="248" width="28" height="28"/>
    <rect x="636" y="276" width="28" height="28"/>
    <rect x="664" y="304" width="28" height="28"/>
    <rect x="692" y="332" width="28" height="28"/>
  </g>
  <text x="580" y="385" text-anchor="middle" font-size="13" fill="currentColor" font-weight="600">10 requests × 100 tokens</text>
  <text x="580" y="408" text-anchor="middle" font-size="12" fill="currentColor" font-family="ui-monospace,monospace">10 × 100² = 10⁵</text>
  <text x="580" y="430" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.75">most of the square is skipped; linears dominate</text>
</svg>

Same outer square. Same total tokens. The colored fraction — what actually gets computed — drops by 10× as you split one long request into ten short ones. The L²-scaling of attention means long-context batches are attention-compute-bound while many-short batches are linear-bandwidth-bound. The flat-tensor trick is the same in both regimes; the bottleneck shifts.

This is also the foreshadowing for decode: when each "request" generates one token at a time, per-request `Q_i K_i.T` becomes a `1 × L_kv` vector times an `L_kv × d_head` matrix. Arithmetic intensity drops to ~1, the per-request matmul stops being meaty, and the entire forward pass becomes bandwidth-bound on weight reads. That's a fundamentally different optimization target — and it's why decode lives in its own article.

---

## 9. What this opens

We now have a scheme for running many concurrent prefill requests through a TP-parallelized model: flatten tokens into one tensor, do one big matmul through every linear layer, do varlen attention through every attention block. The model's TP comm pattern doesn't change. The waste from naive padding is gone. Each request gets exactly the work it needs — no more, no less.

Three follow-up questions earn the next round of articles:

- **What if a request needs to generate many output tokens?** Prefill is one shot per prompt. Decode adds a token-by-token phase with a very different bottleneck profile and a new structure (the KV cache) to remember earlier tokens. **Article 04 — decode and continuous batching across iterations.**
- **What if one request is so long it doesn't fit in a batch?** Sometimes the assumption "every request fits" breaks. The fix is *chunked prefill* — process the prompt in slices, building up the KV cache as you go. **Article 05.**
- **How does the varlen attention kernel actually run fast on a GPU?** We used naive attention math throughout this article. The high-performance version (FlashAttention) avoids materializing the score matrix at all, using a tiled online-softmax recurrence. That's a kernel-level deep-dive worth its own article, later in the series.

Same grammar each time: pick one assumption from the current article, relax it, see what falls out.
