# LLM Stories

A series of essays building up *mental models* for how modern LLMs are actually served — written in plain language, no math notation, lots of diagrams.

📖 **Live site**: https://wgzesg.github.io/llm_stories/

The goal isn't to teach you the equations. It's to build the **intuitions** that make every later equation feel inevitable. Each article picks one slice of the LLM serving pipeline and walks through it as a discovery journey — the kind where each section ends with *"oh, that's all it is?"*

Most articles come in two languages: English and 中文. The Chinese versions keep technical terms in English (中英混排) — they're written for Chinese tech developers and learners, not as literal translations.

---

## Articles

| # | Title | English | 中文 |
|---|---|---|---|
| 01 | An LLM, end to end | [index.md](content/posts/01-llm-end-to-end/index.md) | — |
| 02 | Tensor parallelism, built from scratch in your head | [index.md](content/posts/02-tensor-parallelism-mental-model/index.md) | [index.zh.md](content/posts/02-tensor-parallelism-mental-model/index.zh.md) |
| 03 | Walking tensor parallelism through a full block | [index.md](content/posts/03-tp-through-a-full-block/index.md) | [index.zh.md](content/posts/03-tp-through-a-full-block/index.zh.md) |
| 04 | How to batch many requests through one forward pass | [index.md](content/posts/04-batching-many-requests/index.md) | [index.zh.md](content/posts/04-batching-many-requests/index.zh.md) |
| 05 | ORCA and chunked prefill: evening out the iteration | [index.md](content/posts/05-orca-and-chunked-prefill/index.md) | — |

The full list — shipped, in progress, and the holes still left to dig — lives in the [roadmap](content/posts/00-roadmap/index.md).

---

## Tech stack

- **Static site generator**: [Hugo](https://gohugo.io/) (extended)
- **Theme**: [PaperMod](https://github.com/adityatelange/hugo-PaperMod) (added as a git submodule under `themes/PaperMod`)
- **Hosting**: GitHub Pages, deployed automatically by `.github/workflows/hugo.yml` on every push to `main`

### Local preview

```bash
# Clone with the theme submodule
git clone --recurse-submodules <repo-url>
cd llm_stories

# If you cloned without --recurse-submodules:
git submodule update --init --recursive

# Run the dev server
hugo server -D --buildDrafts

# Open http://localhost:1313/llm_stories/
```

### Adding a new article

```bash
hugo new content posts/02-some-article/index.md       # English (default)
hugo new content posts/02-some-article/index.zh.md    # Chinese
```

Then edit the frontmatter (`draft: false` when ready) and the body.

### Editing published articles

Just edit the markdown in `content/posts/<slug>/` and `git push`. The GitHub Action rebuilds the site automatically. Markdown is the source of truth; nothing is ever "locked."

---

## Style

- **No matrix-math notation.** Just shapes (`[N × d]`) and stories.
- **Pictures over LaTeX.** ASCII diagrams for high-level layout sketches; inline SVG for shape traces, attention masks, and anything that needs to be richer.
- **Discovery journey, not lecture.** The reader should feel they *derived* the answer with us, not had it handed down.
- **Pick one mental model and stick with it.** When metaphors compete, kill the weaker one.
- **Chinese versions = native voice, not translations.** Tech terms stay English; the prose is rewritten for Chinese readers, not transliterated.
