# LLM Stories

A series of essays building up *mental models* for how modern LLMs are actually served — written in plain language, no math notation, lots of pictures made of ASCII.

The goal isn't to teach you the equations. It's to build the **intuitions** that make every later equation feel inevitable. Each article picks one slice of the LLM serving pipeline and walks through it as a discovery journey — the kind where each section ends with *"oh, that's all it is?"*

Most articles come in two languages: English and 中文. The Chinese versions keep technical terms in English (中英混排) — they're written for Chinese tech developers and learners, not as literal translations.

---

## Where we are now

The very first article covers a single, narrow slice:

> **One layer.** **Prefill node.** **The QKV projection step.**
>
> What does that one matmul mean, and how do we split it across GPUs?

That's it. One matmul. From this we get the full mental model for **tensor parallelism** — column-parallel, row-parallel, and why multi-head attention falls naturally into column-parallel TP.

It's a tiny corner of the LLM serving picture. Everything else builds outward from here.

---

## Planned articles

The series will expand outward layer by layer, axis by axis. Roughly in this order:

1. **Tensor parallelism, built from scratch in your head** ✅ *(this one)*
   - One layer's QKV projection. Two ways to read a matrix. Multi-head attention as already-pre-cut.
2. **Attention itself** — what happens after the QKV projection. Q · K, the softmax, the V mix. Why it's quadratic in sequence length.
3. **Norms, residuals, and the rest of the block** — LayerNorm/RMSNorm, residual streams, what they're really doing.
4. **FFN and MoE** — the other half of the transformer block. Why FFN is `d → 4d → d`. How MoE replaces dense FFN with sparse routing.
5. **Stacking layers** — what changes when you have N layers, not just 1. Pipeline parallelism enters the chat.
6. **Batching** — `n` tokens at once, dynamic batching, continuous batching.
7. **Decode node and KV cache** — the *other* phase of serving. Why KV cache exists. What's different about generating one token at a time.
8. **Putting it all together** — how a full serving stack composes prefill nodes, decode nodes, KV cache transfer, and parallelism axes.

These will arrive as I write them. Order may shift; topics may merge or split.

---

## Articles

| # | Title | English | 中文 |
|---|---|---|---|
| 01 | Tensor parallelism, built from scratch in your head | [en.md](01-tensor-parallelism-mental-model/en.md) | [zh.md](01-tensor-parallelism-mental-model/zh.md) |

---

## Style

- **No matrix-math notation.** Just shapes (`[n × d]`) and stories.
- **ASCII diagrams over LaTeX.** Anything that needs a picture should be drawable in a code block.
- **Discovery journey, not lecture.** The reader should feel they *derived* the answer with us, not had it handed down.
- **Pick one mental model and stick with it.** When metaphors compete, kill the weaker one.
- **Chinese versions = native voice, not translations.** Tech terms stay English; the prose is rewritten for Chinese readers, not transliterated.
