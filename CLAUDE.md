# CLAUDE.md — Writing Style and Project Conventions

This file is the editorial brief for the **LLM Stories** series. Read it before writing or editing any article. It captures the voice, structural conventions, and taste decisions that have crystallized across articles 01–06.

## What this series is

Learning notes about how modern LLMs are actually served — written in plain language, no math notation, lots of diagrams. The reader should feel they *derived* the answer with us, not had it handed down.

The goal is **building intuitions that make every later equation feel inevitable** — not teaching equations.

---

## Voice and tone

- **Confident "discovery journey," not lecture.** Each section ends with *"oh, that's all it is?"* — not *"as we have shown..."*
- **Short, direct sentences.** No fluff, no hedging, no academic throat-clearing. If a sentence isn't earning its place, cut it.
- **Use contractions** in English (`isn't`, `can't`, `don't`). They're warmer.
- **Em dashes for asides** — short, sharp ones. Not parentheses unless the aside is genuinely peripheral.
- **Bold for the key claim** of a paragraph. **Italic for emphasis** on a single word. Don't overuse either.
- **"We" sometimes, sparingly.** Not every sentence. Use it when the next move is genuinely shared with the reader.
- **Name things explicitly when introducing them.** "We'll call this `K_tok`." Then use the symbol consistently.
- **Use the reader's mental model, not the textbook's.** When metaphors compete, kill the weaker one.

### Things to avoid

- "It's important to note that..." → just state the thing.
- "From now on..." / "Going forward..." → unnecessary scaffolding.
- "Let's dive into..." / "In this section we will..." → just dive in.
- Sentences that summarize what was just said in the previous paragraph.
- Repeating the same observation twice from different angles (the "two parallel observations" trap — collapse them into one walk-through).
- Long bullet lists when prose would do. Bullets are for genuinely parallel items.
- Pre-empting future articles' content. Hint, don't lecture.
- Out-of-nowhere academic asides ("the papers worth knowing" sections, citation lists). Cite inline where load-bearing; otherwise drop.
- Meta-commentary about notation choices.

---

## Article structure

### Standard skeleton

1. **Opening** picks up a thread from a previous article (when applicable). One paragraph. End with one sentence stating what this article will do.
2. **Numbered sections** (`## 1.`, `## 2.`, …). One section, one job. Aim for 6–8 top-level sections; never more than 10.
3. **Build intuition before formalism.** First section is usually the conceptual setup; numbers come 1–2 sections later.
4. **One concrete anchor model + GPU** throughout the article. We use **Llama-2-7B on H100** as the running example across the series. Don't switch to a different model mid-article unless absolutely necessary — it dilutes the through-line.
5. **Closing section: "What this opens"** — names the threads this article leaves dangling, points at the next article without lecturing about it.

### Section discipline

- A section should have **one charter**, statable in one sentence. If it has two, it's two sections.
- The charter should be in the section's title where possible (e.g., "Why one engine can't serve both well" — title states the thesis).
- **Subheadings inside a section** (`### `) are fine when the section's argument has named beats. Don't subdivide for the sake of it.
- **End-of-section transitions** should point forward concretely: "§3 confirms both with concrete numbers." Not "let's see what's next."

### Pedagogical scaffolding

- **Name what you're going to show before showing it.** "Two facts fall out of one formula:" — then list them. Don't make the reader do detective work.
- **Walk formulas end-to-end in one pass.** If the analysis has two parts (e.g., short-L vs long-L behavior), structure them as Part 1 / Part 2 of the *same* walk-through, not as two parallel sections that each restate the same insight on different grounds.
- **Show, don't tell** — when you derive something, do the derivation in plain text and let the reader watch.

---

## Math, formulas, notation

- **Code blocks for formulas**, not LaTeX. Hugo doesn't render LaTeX in this theme.
- **Define notation before use, in a table.** One symbol, one meaning, with units. Keep the table compact — drop symbols that aren't actually used.
- **Be careful with units**. `Π` is a *count* (dimensionless); `2Π` is *FLOPs* in one context and *bytes* in another. Flag the collision once, then use cleanly.
- **Assume fp16 throughout** the series unless an article specifically needs to talk about quantization. Skip `b_dt` and write `2Π` directly.
- **Concrete numbers next to symbols.** `Π = 7B` (not `Π = 7e9`). `K_tok ≈ 512 KB/token` (not `524288 bytes/token`).
- **Tables for parameter sweeps.** Right-align numerical columns. Include a "regime" column when relevant (`compute-bound` / `bandwidth-bound`).

---

## Diagrams

### SVG over ASCII

After bugs with Markdown mangling box-drawing characters, **always use inline SVG** for diagrams. ASCII diagrams are fine for tiny inline shape sketches inside prose but not for standalone figures.

### SVG conventions

- `viewBox` for responsive sizing; `style="max-width:100%;height:auto"`.
- `font-family:system-ui,sans-serif` for consistency.
- `display:block;margin:1.5rem auto` to center.
- `fill="currentColor"` and `stroke="currentColor"` for text and lines — works in both light and dark themes.
- Use `opacity` (e.g., `0.85`, `0.65`) to soften secondary text rather than choosing a different color.

### Color palette (consistent across the series)

- **Compute / SM / prefill**: `rgba(74,144,226,...)` (blue) with stroke `#4a90e2`.
- **Cache / intermediate / decode**: `rgba(245,166,35,...)` (amber) with stroke `#f5a623`.
- **Storage / HBM**: `rgba(126,211,33,...)` (green) with stroke `#7ed321`.
- **Mask / hatch**: gray (`rgba(150,150,150,...)`) with hatch pattern.

### The blank-line bug

**Never put blank lines inside an SVG block.** Hugo's Markdown parser will treat them as paragraph breaks and wrap the orphaned `<text>` / `<line>` / `<rect>` elements in `<p>` tags, which is invalid inside `<svg>` and breaks rendering. Always write SVGs as one contiguous block.

### What renders as what

- **Inline SVG** for: shape traces, attention masks, memory hierarchies, pipelines, splits, anything that needs structure.
- **Tables** for: parameter sweeps, comparisons across configurations.
- **Code blocks** for: formulas, pseudocode, scheduler bookkeeping.

---

## Chinese versions (`index.zh.md`)

### Style standard: 中英混排, native voice

- **Technical terms stay English**: `prefill`, `decode`, `iteration`, `scheduler`, `batch`, `KV cache`, `ridge point`, `intensity`, `compute-bound`, `bandwidth-bound`, `tensor core`, `SRAM`, `HBM`, `NVLink`, `InfiniBand`, `PagedAttention`, `FlashDecoding`, `GQA`, `MLA`, `MoE`, `EP`, `TP`, `PP`, `RDMA`, etc.
- **Human-language frame in Chinese.** The story, the metaphors, the connective tissue — all native Chinese.
- **No literal translation.** Rebuild the prose flow in Chinese rhythm, not sentence-by-sentence equivalents. If a phrase feels translated, it is — rewrite.
- **Chinese punctuation**: 。，：、—— and `""` for emphasis quotes. Not `,` or `.` or `--` or `""`.
- **No Oxford-comma-style listing.** Chinese uses 、between list items.

### Title format

```
[English-term] 和/与 [English-term]：[一句话核心命题]
```

Examples:
- `Tensor parallelism 心智模型：从零搭起`
- `ORCA 和 chunked prefill：把每次 iteration 的开销摆平`
- `Prefill/Decode 拆机：两个阶段坐在 roofline 的两边`

### Roadmap status conventions

- `[done]` — both English and Chinese shipped, both linked.
- `[done] (EN)` — English shipped, Chinese still pending.
- `[next]` — actively drafting (in either language).
- `[planned]` — on deck.
- `[speculative]` — a hole worth digging, may or may not get filled.

### Article cross-reference style

- English: `[Article 04](/llm_stories/posts/04-batching-many-requests/)`
- Chinese: `[Article 04](/llm_stories/posts/04-batching-many-requests/)` — link target stays English (since we link to the *English* version unless the Chinese exists). The `Article` token can stay English in Chinese prose.

### Quote style for callouts

In both English and Chinese, blockquotes (`>`) are used for the article's central thesis statement, usually one per article, near the end of the analysis section.

---

## Code and edits

- **Don't add comments** unless the *why* is non-obvious. Well-named identifiers and clean structure already explain *what*.
- **No filler scaffolding** — no `// TODO`, no placeholder functions, no "this section will explain..."
- **Don't pre-build for hypothetical future requirements.** Three similar lines is better than a premature abstraction.

---

## Iteration loop with the editor (Zesong)

When working on an article, expect this rhythm:

- **The editor pushes hard on tightness.** "This part is extra." "I feel this is a bit unintuitive." "Make the logic clearer." Take these seriously — they almost always point at real bloat or unclear pedagogy.
- **Convert ASCII → SVG when asked.** Don't argue.
- **Drop redundant terms aggressively.** Past examples: `token-row` collapsed to `token`; `the linears` collapsed to `matmul layers`; `b_dt` dropped because we always assume fp16.
- **When two sections feel similar, they probably are.** Look for a way to merge into one walk-through, or sharpen the distinction so each is doing distinct work.
- **Section count drift downward, not upward.** If the article has grown from 7 to 10 sections, something probably wants to merge.

---

## Memory and persistence

- **Cross-session project memory** lives in `/Users/zesong/.claude/projects/-Users-zesong-Document-llm-stories/memory/` (auto-managed). Article ordering decisions, style decisions worth remembering across sessions go there.
- **In-session work scaffolding** (plans, todos) is ephemeral.
- **This file (`CLAUDE.md`)** is the durable editorial brief — update it when a new convention crystallizes.

---

## Article-by-article register

Quick anchor of what's shipped and what each article is about, so future sessions can orient fast:

| # | English title | Status |
|---|---|---|
| 01 | An LLM, end to end | shipped (EN + ZH) |
| 02 | Tensor parallelism, built from scratch in your head | shipped (EN + ZH) |
| 03 | Walking TP through a full block | shipped (EN + ZH) |
| 04 | How to batch many requests through one forward pass | shipped (EN + ZH) |
| 05 | ORCA and chunked prefill | shipped (EN + ZH) |
| 06 | Prefill and decode disaggregation | shipped (EN + ZH) |
| 07 | The engineering of disaggregation (KV transfer fabrics, memory tiers) | next |
| 08 | Pipeline parallelism | planned |
| 09 | MoE and expert parallelism | planned |
| 10 | PagedAttention | planned |
| 11+ | Sequence/context parallelism, FlashAttention/FlashDecoding, GQA/MLA, speculative decoding, KV compression | speculative |

---

## Tech stack reminder

- Hugo (extended) + PaperMod theme.
- Site lives at https://wgzesg.github.io/llm_stories/.
- Hugo dev server: `hugo server -D --bind 127.0.0.1 --port 1313` then http://localhost:1313/llm_stories/.
- GitHub Actions auto-deploys on push to `main`.
- Markdown is the source of truth.
