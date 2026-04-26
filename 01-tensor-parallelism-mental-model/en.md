# Tensor Parallelism, Built From Scratch in Your Head

This isn't a tutorial. It's a walk through the mental model — the kind where each section makes you go *"oh, that's all it is?"* By the end, tensor parallelism shouldn't feel like an engineering trick. It should feel like the only two reasonable things you could possibly do.

No matrix-math notation. Just shapes and stories.

---

## 1. The only picture you need of an input

Forget tokens-as-words for a second. To a model, a token is just a row of numbers — `d` of them. A "feature vector" if you want to be fancy.

A whole sentence (or batch) is just a stack of those rows:

```
Token 1 → [ f1  f2  f3  ...  fd ]
Token 2 → [ f1  f2  f3  ...  fd ]
Token 3 → [ f1  f2  f3  ...  fd ]
   ...
Token n → [ f1  f2  f3  ...  fd ]
```

That's it. `n` tokens, each living in `d`-dimensional space. Hold that picture — everything else builds on it.

---

## 2. Where this matrix actually shows up

Before we play with weight matrices abstractly, let's anchor on a concrete moment in real LLM serving — so the shapes feel like something, not nothing.

When an LLM handles your prompt, the first big phase is **prefill**: shove all `n` prompt tokens through the network in one shot. (The token-by-token decoding comes after.) And the very first computation inside prefill is the **QKV projection** in attention — each token (length `d`) gets turned into a query, key, and value vector (each length `k`).

Stack the tokens as the `n × d` table from section 1, and the whole QKV step (just the Q part shown here) is **one matrix multiply**:

```
[ n × d ]   @   [ d × k ]   =   [ n × k ]
  tokens         weight          per-token
                 matrix          query vectors
```

That's the shape. Now sit with the question for a beat:

> **What is this matmul actually *doing*?**
> What does it mean to multiply an `n × d` table of tokens by a `d × k` weight matrix?

"We computed the queries" is the boring answer. The interesting question is what's happening **inside** that `d × k` matrix — and there are two very different stories you can tell. Each one quietly hands you a different way to split the work across GPUs.

---

## 3. The same weight matrix, told two different ways

A linear layer takes a token (length `d`) and produces something of length `k`. The "thing" doing this is a weight matrix of shape `d × k`.

> **Quick aside that matters a lot.** When I say "linear layer," I don't mean one specific block in the network. I mean *every* matmul in a transformer:
> - the **Q, K, V** projections in attention — each is a `d × k` matrix turning a token into a query/key/value vector
> - the **attention output** projection
> - the **FFN up-projection** (`d → 4d`) and **FFN down-projection** (`4d → d`)
> - even the **unembedding** at the end
>
> They're all the same shape of operation: token in, matrix multiply, token out. So the two-views story below — and the two parallelism strategies that fall out of it — apply to **all of them**. Once you see it for one, you've seen it for the whole transformer.

Here's the fun part: a `d × k` matrix can be **read two ways** — **column by column** or **row by row**. Same numbers, same multiplication, but two completely different mental scenes. We'll walk through both.

### Story A — read it **column by column** (a row of `fx`es)

Stop seeing the weight matrix as a grid of numbers. Zoom out. Each column is a self-contained little function — give it a token, it returns one number. We'll call each one **`fx`** (short for **feature extractor**) and just draw the whole weight matrix as a row of `k` of them:

```
weight  =  [ fx1   fx2   fx3   ...   fxk ]
```

That's the whole matrix. Not numbers — *fxes*. Each one is its own opaque thing.

> *How* does each `fx` turn a token into a number? It happens to be an inner product with that column's `d` weights. But honestly — for building intuition, you don't care. It's just "`fxi` looks at the token and reports a score."

Now applying this layer to a token is just: send the token down the row, collect what comes out the bottom.

```
token  ⇒  [ fx1   fx2   ...   fxk ]
              ↓     ↓           ↓
          [ fx1(token), fx2(token), ..., fxk(token) ]
```

A token walks past `k` little extractors, each shouts a number, you collect the numbers. Output has length `k`. Done.

### Story B — read it **row by row** (a stack of basis vectors)

Now lay the same matrix flat. There are `d` rows, each of length `k`:

```
Row 1 → [ r1  r2  r3  ...  rk ]
Row 2 → [ r1  r2  r3  ...  rk ]
   ...
Row d → [ r1  r2  r3  ...  rk ]
```

Each row is a **basis vector living in the output space** (length `k`). And the token's `d` features are the **coefficients** that say how much of each row to mix in.

```
output  =  f1 · Row1  +  f2 · Row2  +  ...  +  fd · Rowd
```

The layer's job, told this way: take the `d` features of a token, use them as a recipe, and **linearly combine** the `d` row-vectors into one output vector.

### The "wait, what?" moment

Both stories describe the **exact same multiplication**. Same numbers in, same numbers out. But your brain holds two different scenes:

| Story A (columns) | Story B (rows) |
|---|---|
| many independent `fx`es | one big linear combination |
| "extract `k` features from the token" | "mix `d` row-vectors into the output" |
| output is *collected* | output is *summed* |

This duality isn't a curiosity — it's the seed of tensor parallelism. The two ways you can read a matrix are the two ways you can split it across GPUs.

---

## 4. Now there are two GPUs. What's the obvious thing to do?

You have one matrix and two GPUs. You stare at the matrix. There are really only two natural lines you could draw on it: a vertical one or a horizontal one.

Section 3 just told you what each of those lines *means*.

---

## 5. Strategy A — Split the `fx`es (Column Parallel)

Take Story A seriously. The weight matrix is just a row of `k` black-box `fx`es. Cutting it across two GPUs is — literally — drawing one vertical line through that row:

```
weight =  [ fx1  ...  fx(k/2)  ‖  fx(k/2+1)  ...  fxk ]
                    ↑                                ↑
                    └──── GPU 1 ───┘  └──── GPU 2 ───┘
```

**Each GPU sees the full token.** It just runs *its* half of the `fx`es.

```
GPU 1 →  [ fx1(token), ..., fx(k/2)(token) ]
GPU 2 →  [ fx(k/2+1)(token), ..., fxk(token) ]
```

To assemble the final output, **glue them side by side**:

```
output  =  [  GPU1's half  |  GPU2's half  ]
```

That's it. No summing, no synchronizing in the middle. Each GPU runs *different* `fx`es on the *same* input, and the answers just live next to each other.

**Cost**: cheap. Concatenation is basically free.

---

## 6. Strategy B — Split the rows (Row Parallel)

Take Story B seriously. The weight matrix is a stack of `d` basis-vector rows. Cutting it across two GPUs is — literally — drawing one horizontal line through that stack:

```
weight  =  [ Row 1       ]  ┐
           [ Row 2       ]  │  GPU 1  (paired with features 1..d/2)
           [   ...       ]  │
           [ Row(d/2)    ]  ┘
           ─────────────────────────
           [ Row(d/2+1)  ]  ┐
           [   ...       ]  │  GPU 2  (paired with features d/2+1..d)
           [ Row d       ]  ┘
```

And here's the catch: each row is multiplied by its matching token feature (Row `i` pairs with `f_i`). So splitting the rows **automatically** splits the input too — GPU 1 only ever needs `f_1..f_(d/2)`, GPU 2 only ever needs the rest.

**Each GPU sees only half the token.** It produces a partial output — a length-`k` vector that's only part of the sum.

```
GPU 1 →  partial output (its rows, weighted by its features)
GPU 2 →  partial output (its rows, weighted by its features)
```

To assemble the final output, **add them up**:

```
output  =  GPU1's partial  +  GPU2's partial
```

This time you can't just concatenate — both GPUs produced length-`k` vectors that need to be *summed element-wise*. That sum has to happen across the network. (This is the "all-reduce" you'll see in TP papers.)

**Cost**: more expensive. Every forward pass through this layer pays a cross-GPU sum.

---

## 7. The two strategies, side by side

|  | Split columns (A) | Split rows (B) |
|---|---|---|
| Story it lives in | "row of `fx`es" | "weighted combination of rows" |
| What each GPU holds | some of the `fx`es | some of the rows + matching features |
| Each GPU sees... | the **full** input | **part** of the input |
| How outputs combine | **concatenate** | **sum** (all-reduce) |
| Communication | cheap | expensive |

Same matrix. Two stories. Two ways to cut it. That's the whole game.

---

## 8. Multi-head attention: the cuts were already there

Time to apply this to a real piece of a transformer — the QKV projection in attention — and watch column-parallel TP fall out for free.

### The setup

The Q (or K, or V) projection turns each token (length `d`) into a query vector of length `k`. But `k` isn't arbitrary — it's structured:

> `k  =  h × d_head`

where `h` is the **number of heads** and `d_head` is the **per-head dimension**.

So our row of `fx`es is *organized*. Group every `d_head` consecutive `fx`es together and call each group a **head**:

```
W_Q  =  [ fx1 ... fx(dh) │ fx(dh+1) ... fx(2·dh) │ ... │ fx((h-1)·dh+1) ... fxk ]
         └── Head 1 ────┘  └──── Head 2 ─────────┘       └──── Head h ──────────┘
```

Head 1's `fx`es produce Head 1's query vector. Head 2's `fx`es produce Head 2's. Same matrix, same row of `fx`es — just organized into groups.

> **Implementation note.** In practice this is one big matmul of shape `[d, h × d_head]`, not `h` separate ones — one large matrix multiply is dramatically faster on a GPU than many small ones. The "h heads" structure lives in *what each column means*, not in how many matrices there are. (Production codebases often go further and fuse Q, K, and V into a single `[d, 3 × h × d_head]` matmul.) Mathematically — and for training — it makes no difference. The head structure is purely a logical grouping.

### Why heads make column-parallel feel inevitable

Here's the punchline.

Heads are **independent** during the attention computation itself. Head 1's attention only mixes Head 1's queries with Head 1's keys; Head 2 does its own thing with its own stuff; they never peek at each other. The heads only get combined back together at the very end, by a separate matrix (the output projection).

So if you're going to split the `fx`es column-wise across GPUs anyway (Strategy A from section 5)... split **between heads**:

```
W_Q  =  [ Head 1  │  Head 2  │  Head 3  │  Head 4 ]
            ↑         ↑           ↑          ↑
            └── GPU 1 ─┘           └── GPU 2 ──┘
```

Each GPU owns some heads. It computes *its* heads' Q, K, V *and* runs their attention end-to-end on its own. **Zero communication during attention itself.** Each GPU is doing its own private mini-attention.

### The aha

Multi-head attention wasn't designed for tensor parallelism. It was designed because different heads learn to attend to different relational patterns in the input — that's a modeling choice, not a systems one.

But when TP came along, it walked in and noticed: *attention was already pre-sliced into independent slabs called "heads."* It didn't have to invent anything. It just respected the cuts that were already there.

This is the cleanest column-parallel TP case in a real transformer — and it falls directly out of Story A. The matrix is a row of `fx`es; the `fx`es are grouped into heads; heads are independent; therefore split on head boundaries. One mental model the whole way down.
