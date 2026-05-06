---
title: "Tensor Parallelism 心智模型：从零搭起"
date: 2026-04-26T00:00:00+00:00
draft: false
summary: "把 weight matrix 用两种方式读，就有两种把它切到多 GPU 上的方法。从 transformer prefill 里的一次 matmul，推出 tensor parallelism 的整套心智模型。"
description: "Tensor parallelism 心智模型 —— 从 transformer prefill 阶段的一次 matmul 推出。顺带讲清楚 multi-head attention 为什么早就为 column-parallel TP 切好了刀口。"
tags: ["tensor-parallelism", "transformers", "llm-serving", "attention", "mental-model"]
series: ["llm-stories"]
showToc: true
TocOpen: false
weight: 3
---

这不是一份教程。这是一段在你脑子里搭 mental model 的旅程 —— 每读完一节，你会忍不住想 *"哦，原来就这么回事？"*。读到最后，tensor parallelism 不再是一个工程上的奇技淫巧。在那个场景下，它会变成 —— **你能想到的最自然不过的两个选择**。

不写矩阵公式。只讲 shape 和故事。

---

## 1. 关于"输入"，你只需要这一张图

先别把 token 当成"词"。在模型眼里，一个 token 就是一行数字 —— `d` 个数。要装一点的话，叫 "feature vector"。

一整句话就是这堆行的堆叠：

```
Token 1 → [ f1  f2  f3  ...  fd ]
Token 2 → [ f1  f2  f3  ...  fd ]
Token 3 → [ f1  f2  f3  ...  fd ]
   ...
Token n → [ f1  f2  f3  ...  fd ]
```

就这。`n` 个 token，每个都活在 `d` 维空间里。把这张图记牢 —— 后面所有东西都建立在它之上。

---

## 2. 这个矩阵到底从哪冒出来的

在我们抽象地玩 weight matrix 之前，先在真实的 LLM 推理场景里找一个具体的落脚点 —— 这样 shape 才有实感，而不是一串空气。

LLM 处理你的 prompt 时，第一个大阶段叫 **prefill**：把 `n` 个 prompt token 一次性全部塞进网络。（一个一个吐回答 token 是后面的 decode 阶段。）而 prefill 内部第一个计算，就是 attention 里的 **QKV projection** —— 每个 token（长度 `d`）要变成一个 query、一个 key、一个 value（每个长度 `k`）。

把 token 堆成 section 1 那张 `n × d` 表，整个 QKV 这一步（这里只画 Q）就是**一次矩阵乘法**：

```
[ n × d ]   @   [ d × k ]   =   [ n × k ]
  tokens         weight          每个 token
                 matrix          的 query
```

shape 就长这样。先停一下，把这个问题嚼一会儿：

> **这个 matmul 到底在*做*什么？**
> 一张 `n × d` 的 token 表去乘一个 `d × k` 的 weight matrix，到底是个什么意思？

"算出 query 啊"是无聊的回答。**有意思的问题是 —— 这个 `d × k` 矩阵*内部*到底发生了什么。** 这里有两个完全不同的故事可讲。每个故事都会悄悄给你一种把工作切到多 GPU 上的方式。

---

## 3. 同一个 weight matrix，两种讲法

linear layer 把一个 token（长度 `d`）变成长度 `k` 的东西。干这活的是一个 shape 为 `d × k` 的 weight matrix。

> **一个超级重要的题外话。** 我说 "linear layer" 的时候，不是指网络里某一个特定的 block。我是指 transformer 里**每一个** matmul：
>
> - attention 里的 **Q, K, V** projection —— 每个都是把 token 变成 query / key / value 的 `d × k` 矩阵
> - **attention output** projection
> - **FFN up-projection** (`d → 4d`) 和 **FFN down-projection** (`4d → d`)
> - 甚至最末尾的 **unembedding**
>
> 它们都是同一种 shape 的运算：token 进，矩阵乘，token 出。所以下面要讲的"两种视角"，以及由此引出的两种并行策略，对**所有这些**都适用。一个看明白了，整个 transformer 的 matmul 都看明白了。

接下来就是有意思的部分了：一个 `d × k` 矩阵可以**用两种方式读** —— **一列一列地读**或**一行一行地读**。同样的数字，同样的乘法，但脑子里是两幅完全不同的画面。两个我们都会走一遍。

### Story A —— 一列一列地读（一排 `fx`）

别再把 weight matrix 当成一堆数字组成的格子。镜头拉远。每一列都是一个独立的小函数 —— 给它一个 token，它返回一个数字。我们就把每一列叫做 **`fx`**（**feature extractor** 的缩写），然后把整个 weight matrix 画成一排 `k` 个 `fx`：

```
weight  =  [ fx1   fx2   fx3   ...   fxk ]
```

整个矩阵就这样。不再是数字 —— 是 **`fx`**。每一个都是它自己独立的小黑盒。

> 每个 `fx` *怎么*把 token 变成一个数字？其实就是和它那一列的 `d` 个权重做一个内积。但说实话 —— 为了建立 intuition，你**不用关心**。它就是 "`fxi` 看一眼 token，给出一个分数"。

那这一层 layer 作用在 token 上，就只是：让 token 从这一排 `fx` 走过一遍，把每个 `fx` 吐出来的数字接住。

```
token  ⇒  [ fx1   fx2   ...   fxk ]
              ↓     ↓           ↓
          [ fx1(token), fx2(token), ..., fxk(token) ]
```

一个 token 走过 `k` 个小评委（feature extractor），每个吼出一个数字，你按顺序收集起来。output 长度就是 `k`。完事。

### Story B —— 一行一行地读（一摞 basis vector）

现在把同一个矩阵平铺。它有 `d` 行，每行长度 `k`：

```
Row 1 → [ r1  r2  r3  ...  rk ]
Row 2 → [ r1  r2  r3  ...  rk ]
   ...
Row d → [ r1  r2  r3  ...  rk ]
```

每一行都是 output space（长度 `k`）里的一个 **basis vector**。而 token 的 `d` 个 feature 就是 **coefficient** —— 告诉你每一行该掺多少进去。

```
output  =  f1 · Row1  +  f2 · Row2  +  ...  +  fd · Rowd
```

这一层 layer 用这种讲法就是：拿 token 的 `d` 个 feature 当配方，把 `d` 个 row vector **线性组合**成一个 output vector。

### "等等，怎么……" 的瞬间

两种讲法描述的是**完全相同的乘法**。同样的数字进去，同样的数字出来。但你脑子里的画面截然不同：


| Story A (column)                | Story B (row)                     |
| --------------------------------- | ----------------------------------- |
| 多个独立的`fx`                  | 一次大的线性组合                  |
| "从 token 里提取`k` 个 feature" | "把`d` 个 row vector 混成 output" |
| output 是*被收集起来*的         | output 是*被加起来*的             |

这个二元性不是个 trivia —— **它就是 tensor parallelism 的种子**。矩阵能怎么读，就能怎么切到 GPU 上。

---

## 4. 现在你有两块 GPU。最自然的事是什么？

一个矩阵，两块 GPU。盯着这个矩阵看。其实你能在它上面画的"自然的"切线只有两条：要么竖着切，要么横着切。

Section 3 刚刚告诉你，每条线*意味着*什么。

---

## 5. Strategy A —— 切 `fx` (Column Parallel)

把 Story A 当真。weight matrix 就是一排 `k` 个黑盒 `fx`。把它切到两块 GPU 上 —— 字面意义上 —— 就是在这排上画一条竖线：

```
weight =  [ fx1  ...  fx(k/2)  ‖  fx(k/2+1)  ...  fxk ]
                    ↑                                ↑
               └──── GPU 1 ───┘  └──── GPU 2 ───┘
```

**每块 GPU 都看到完整的 token。** 它只是跑*它自己*那一半 `fx`。

```
GPU 1 →  [ fx1(token), ..., fx(k/2)(token) ]
GPU 2 →  [ fx(k/2+1)(token), ..., fxk(token) ]
```

要拼出最终 output，**直接拼起来就行**：

```
output  =  [  GPU1 那半  |  GPU2 那半  ]
```

完事。中间不需要求和，不需要同步。每块 GPU 在跑*不同*的 `fx`，输入是*同一个*，结果就这么紧挨着放。

**通信成本**：便宜。拼接基本不要钱。

---

## 6. Strategy B —— 切 row (Row Parallel)

把 Story B 当真。weight matrix 是一摞 `d` 个 basis vector row。把它切到两块 GPU 上 —— 字面意义上 —— 就是横着画一条线：

```
weight  =  [ Row 1       ]  ┐
           [ Row 2       ]  │  GPU 1  (配 feature 1..d/2)
           [   ...       ]  │
           [ Row(d/2)    ]  ┘
           ─────────────────────────
           [ Row(d/2+1)  ]  ┐
           [   ...       ]  │  GPU 2  (配 feature d/2+1..d)
           [ Row d       ]  ┘
```

但这里有个微妙的事：每一行都要乘上对应的 token feature（Row `i` 配 `f_i`）。所以**切了 row，自动就切了输入** —— GPU 1 永远只需要 `f_1..f_(d/2)`，GPU 2 只需要剩下那一半。

**每块 GPU 只看到 token 的一半。** 它产出的 output 是个长度为 `k` 的 vector —— 但只是总和的*一部分*。

```
GPU 1 →  partial output (它的 row 乘它的 feature)
GPU 2 →  partial output (它的 row 乘它的 feature)
```

要拼出最终 output，**得加起来**：

```
output  =  GPU1 的 partial  +  GPU2 的 partial
```

这次不能直接拼接 —— 两块 GPU 各自产出长度为 `k` 的 vector，需要*逐元素相加*。这个加法必须跨 GPU 完成。（这就是 TP 论文里那个 "all-reduce"。）

**通信成本**：贵。这一层每次 forward 都要跨 GPU 做一次求和。

---

## 7. 两种策略，并排对比


|                     | 切 column (A) | 切 row (B)                |
| --------------------- | --------------- | --------------------------- |
| 基于哪个故事        | "一排`fx`"    | "row 的加权组合"          |
| 每块 GPU 拥有什么   | 一部分`fx`    | 一部分 row + 配套 feature |
| 每块 GPU 看到的输入 | **完整**的    | 只有**一部分**            |
| 输出怎么合          | **拼接**      | **求和** (all-reduce)     |
| 通信开销            | 便宜          | 贵                        |

同一个矩阵。两个故事。两种切法。这就是全部游戏规则。

---

## 8. Multi-Head Attention：切口本来就在那

把这套东西用到 transformer 里一个真实的部件上 —— attention 里的 QKV projection —— 看着 column-parallel TP 怎么*免费*掉出来。

### 场景

Q（K, V 同理）projection 把每个 token（长度 `d`）变成一个长度 `k` 的 query vector。但 `k` 不是一个随便的数字 —— 它是有结构的：

> `k  =  h × d_head`

其中 `h` 是 **head 数量**，`d_head` 是 **每个 head 的维度**。

所以我们那一排 `fx` 是有*组织*的。每 `d_head` 个相邻的 `fx` 分一组，每一组就叫一个 **head**：

```
W_Q  =  [ fx1 ... fx(dh) │ fx(dh+1) ... fx(2·dh) │ ... │ fx((h-1)·dh+1) ... fxk ]
         └── Head 1 ────┘  └──── Head 2 ─────────┘       └──── Head h ──────────┘
```

Head 1 的 `fx` 产出 Head 1 的 query vector，Head 2 的 `fx` 产出 Head 2 的，以此类推。同一个矩阵，同一排 `fx` —— 只是分了组。

> **实现层面的小注。** 实际代码里这是**一个**大 matmul，shape `[d, h × d_head]`，**不是** `h` 个小 matmul —— 一个大的矩阵乘法在 GPU 上比一堆小的快得多得多。"h 个 head" 这个结构活在*每一列代表什么*这件事里，不在矩阵的数量里。（很多生产环境的代码会更激进 —— 把 Q, K, V 三个 projection fuse 成一个 `[d, 3 × h × d_head]` 的大 matmul，一次算完。）数学上 —— 包括训练时 —— 完全没区别。head 这个结构纯粹是个**逻辑上的分组**。

### 为什么 head 让 column-parallel 显得理所当然

下面是高潮部分。

attention 内部的计算里，head 之间是**互不相干**的。Head 1 的 attention 只让 Head 1 的 query 和 Head 1 的 key 玩，Head 2 自己玩自己的，永远不互相看。所有 head 真正混在一起，要等到最后一个独立的矩阵 —— output projection。

所以如果你反正都要把 `fx` 按 column 切到 GPU 上（也就是 section 5 里的 Strategy A）……就**沿着 head 的边界切**：

```
W_Q  =  [ Head 1  │  Head 2  │  Head 3  │  Head 4 ]
            ↑         ↑           ↑          ↑
            └── GPU 1 ─┘           └── GPU 2 ──┘
```

每块 GPU 拥有几个 head。它独立完成*自己*那几个 head 的 Q, K, V projection *和* attention。**整个 attention 期间零通信。** 每块 GPU 都在跑自己的私有 mini-attention。

### 真正的 aha

multi-head attention 不是为 tensor parallelism 设计的。它的初衷是让不同的 head 学到关注输入里不同的关系模式 —— 这是个建模层面的选择，不是系统层面的。

但 TP 出现的时候，它走过来一看：*哎，attention 早就被预先切成一块块独立的 "head" 了。* 它根本不需要发明任何东西。它只是顺着*已经存在的*切口把工作分了。

这就是真实 transformer 里最干净的 column-parallel TP 案例 —— 它直接从 Story A 掉出来。matrix 是一排 `fx`，`fx` 按 head 分组，head 之间互相独立 —— 所以沿着 head 边界切。**一个 mental model 用到底。**
