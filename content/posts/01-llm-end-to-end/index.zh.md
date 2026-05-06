---
title: "LLM 从头到尾走一遍"
date: 2026-05-06T00:00:00+00:00
draft: false
summary: "三个 zoom level —— 整张网络、一个 transformer block 的内部、生成一段文字的完整循环。够你在脑子里搭起一个 LLM 的样貌，也够你开始问对的问题。"
description: "现代 decoder-only LLM 的基础知识，从三个 zoom level 看：鸟瞰整个 stack、打开一个 transformer block、跑完一次完整生成。LLM Stories 系列的入口。"
tags: ["fundamentals", "transformer", "llm-serving", "mental-model"]
series: ["llm-stories"]
showToc: true
TocOpen: false
weight: 2
---

你看到 LLM 写出来的每一段回复，都是**一个 token 一个 token 地生成出来的** —— 同一个固定大小的模型，反反复复地在自己刚吐出来的 output 上跑了一遍又一遍。不是只有那段最精彩的，而是*每一段*都这么来。这个模型本身叫 **transformer**，外面那个反复调用它的循环则基本上是机械式的记账工作。把这个模型搞懂、把那个循环搞懂，这件事就吃透了。系列后面的所有事 —— 把模型切到多 GPU 上、让很多用户共享一次 forward、让超长 prompt 装得下、让生成更快 —— 都建立在这两块之上。

这一篇从三个 zoom level 把模型打开：

1. 整个模型，从头到尾 —— 进什么、出什么、中间发生了什么。
2. 拉近一层 —— 那个标着 "transformer block" 的东西，里面究竟是什么。
3. 完整循环 —— 这个一次只生成一个 token 的模型，怎么被用来吐出一长段回复的。

为了讨论尽量通用，我们一直用符号（`d`、`L`、`h` 之类）而不是具体数字 —— 因为各家模型的*结构*相通，但具体数字千差万别。不同模型尺寸不一样，但骨架都长这样。具体数字留给后面的文章 —— 等到它们真的承重的时候再说。

边读边会冒出一些自然的问题 —— 比如*"等等，每生成一个 token 模型就要把前面那么多事重新做一遍？"*或者*"那如果模型大到一张 GPU 装不下怎么办？"*这些问题，正是后面整个系列要逐一拆开来回答的。每个问题最后都会有自己的一篇。

---

## Part I —— 整个模型，从头到尾

## 1. token 进来，下一个 token 出去

塞给模型一段话 —— 比如 `"the quick brown fox jumps over"` —— 让它接着往下写。它实际上到底做了哪些事？从头到尾六步。

**1. Tokenize。** 第一步把字符串切成一小段一小段，每一段叫一个 **token**。每个 token 都是一个小整数 ID —— 因为模型底层只会做算术，处理不了"字"本身。粗略地说：常见的短词通常一个 token，生僻词或长词会被拆成几段。token 的数量我们记作 `N`。

**2. Embed。** 每个整数 ID 拿去查一张巨大的表 —— **embedding table**。表里每一行对应词表里的一个 token，每一行是 `d` 个数字组成的向量（`d` 是模型自己挑的一个超参，叫 **hidden dimension**。真实模型里 `d` 一般在几千这个量级）。`N` 个 token 查完之后，原本一串 `N` 个整数 ID 变成了一个形状 `[N × d]` 的 tensor：`N` 行，每行 `d` 个数。

为什么要换成向量、不直接用整数 ID？因为模型底层只会做线性代数，整数 ID 之间没有任何有用的几何关系 —— token 5 不会因为是连续整数就比 token 100 离 token 6 "更近"。embedding table 给每个 token 在 `d` 维空间里安排了一个*学到的位置*：意思相近的 token 落在附近，没什么关系的 token 离得远。每一行可以理解成模型对那个 token 的"第一印象" —— 还没看到它在句子里的上下文之前，凭空对它的*感觉*。

（词表大小记作 `vocab`，一般几万这个量级。所以 embedding table 自己就是一个 `[vocab × d]` 的矩阵 —— 这本身就是不小一坨参数，§2 里再聊。）

**3. 一摞 transformer block。** 这个 `[N × d]` 的 tensor 接着穿过 `L` 个 **transformer block**，一个摞一个。每个 block 都会读一遍完整序列，把不同 position 之间的信息混合一下，再写回一个更精炼的版本。关键是：每个 block 的 input 和 output 形状*完全一样*，都是 `[N × d]` —— 只是行里的*内容*被改了。

`L` 个 block 都过完之后，每一行已经离最初那个起点很远了。它代表的不再是这个 token 的脱离上下文的、通用含义，而是它*在这个具体序列里*的含义。block 为什么能这样一直摞下去，§2 专门聊；§Part II 会把一个 block 拆开来看。

**4. Final norm。** 一摞 block 顶上还有一个小小的归一化步骤 —— 算是个收尾的整理。形状不变，进什么形状出什么形状。

**5. LM head。** 一个 linear layer 把每一行从 `d` 维投回去，每一行变成 `vocab` 个数 —— 词表里每个 token 一个数。output 形状 `[N × vocab]`。每一行是一长条对整个词表的"打分"。这种原始分数叫 **logits**。位置 `i` 上 token `t` 的 logit，是模型对*"在 position `i` 上下一个 token 是 `t` 有多合理"*那种原始的、没归一化过的回答。

**6. Softmax → sample。** 我们真正要的是*最后一行* —— 也就是最后一个输入 token 之后那个位置，那里放着模型对"下一个该是什么"的预测。**Softmax** 把那一行的 logits 拧成一个干净的概率分布 —— 全是正数，加起来等于 1。从这个分布里采样一个 token，这就是模型对下一个 token 的猜测。

整个 stack 拍下来：

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

所以整个模型本质上就是一个函数：吃 `N` 个 token，吐回一个对第 `N+1` 个 token 该是什么的概率分布。其它所有东西 —— 那些聊天式的回复、长篇回答、聊天 UI 里一个字一个字蹦出来的流式输出 —— 都是把这个函数反复调用得来的。这个循环，Part III 来讲。

---

## 2. block 为什么能一直摞下去：stream-processor 模式

一句话讲清楚 transformer：**一摞 `L` 个完全同款的 "stream processor"**，吃一个固定形状的 token 流，加工一下，往下传。这个形状就是 `[N × d]`。同样的形状进、同样的形状出，重复 `L` 次。

这条性质为什么重要？两个理由，后面整个系列都在反复用：

1. **它让模型可以靠"摞"来变大。** 想要一个更大的模型？多摞几个 block 就行。一个小型开源模型和一个巨大的旗舰模型，从这个 zoom level 看几乎是一模一样的 —— 同样的六步 pipeline、同样的 block 结构，只是 `L` 不同（`d` 也稍微宽一点）。同一份菜谱，放大版本。
2. **它让所有下游工具都不用关心"在第几层"。** 一个 block 根本不知道自己是第 1 个还是第 32 个，所以任何接触 block 的工具（切到多 GPU 的 splitter、做 batching 的 batcher、做 scheduling 的 scheduler）也都不用关心。整摞 block 是一片整齐的 substrate，工具直接在上面操作。

（这个想法在别的地方你可能也见过 —— Unix pipe、音频插件、图像处理流水线。同样的形状进、同样的形状出，想摞多少摞多少。）

把形状钉得更死一点：回看 §1 的六步，过了 tokenization 之后，中间每一步进出的都是同一个 `[N × d]` tensor。

- embedding 把 `N` 个 token ID 变成一个 `[N × d]` tensor。
- 每个 transformer block 读 `[N × d]`、返回 `[N × d]`。
- final norm 读 `[N × d]`、返回 `[N × d]`。
- 只有最顶上的 LM head 改了宽度 —— 把它换回 `vocab` 那么宽。

管道中段，形状**从来不变**。变的是内容 —— 每个 block 都在精炼这些行，慢慢搭出越来越丰富、越来越能感知上下文的表示 —— 但从最底层到最顶层，几何结构始终是 `[N × d]`。

后面会反复用到的几个符号：

- `N` —— 当前这段序列的长度。每次请求都不一样 —— 它是 input 的属性，不是模型的属性。
- `d` —— **hidden dimension**。流过 stack 的 tensor 每一行的宽度。
- `L` —— 摞了多少个 transformer block。
- `vocab` —— 模型认识多少种不同的 token。决定了 embedding table 和 LM head 的宽度。

Part II 还会再见到两个：`h`（一个 block 里的 attention head 数量）和 `d_head`（每个 head 多宽）。

---

## Part II —— 把一个 block 打开

## 3. 一个 block 拍平来看

现在我们打开 `L` 个 transformer block 里的一个。好消息是：它们*内部结构都一样* —— 不同 block 学到的*参数*不同，但接线方式一模一样。看懂一个，就看懂了所有 `L` 个。

一个 block 分成两半，每一半都被一个 residual connection 包起来（每一半底下那个小小的 `+` —— 一会儿就解释）：

<svg viewBox="0 0 720 720" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;font-family:system-ui,sans-serif;display:block;margin:1.5rem auto">
  <defs>
    <marker id="arr-block-overview" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
      <path d="M 0 0 L 8 4.5 L 0 9 Z" fill="currentColor"/>
    </marker>
  </defs>
  <text x="360" y="24" text-anchor="middle" font-size="14" fill="currentColor" font-weight="600">one block — two halves, each wrapped in a residual</text>

  <rect x="270" y="44" width="180" height="36" fill="rgba(74,144,226,0.20)" stroke="#4a90e2" stroke-width="1.5"/>
  <text x="360" y="67" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">input</text>
  <text x="465" y="66" font-size="11" fill="currentColor" opacity="0.7" font-family="ui-monospace,monospace">[N × d]</text>

  <line x1="360" y1="80" x2="360" y2="98" stroke="currentColor" stroke-width="1.5"/>
  <circle cx="360" cy="100" r="3.5" fill="currentColor"/>
  <line x1="360" y1="103" x2="360" y2="118" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-block-overview)"/>
  <path d="M 360 100 L 190 100 L 190 342 L 349 342" fill="none" stroke="currentColor" stroke-width="1.5" stroke-dasharray="4 4" marker-end="url(#arr-block-overview)"/>
  <text x="135" y="220" font-size="11" fill="currentColor" opacity="0.65" font-style="italic">residual</text>

  <path d="M 615 122 L 635 122 L 635 322 L 615 322" fill="none" stroke="currentColor" stroke-width="1" opacity="0.45"/>
  <text x="645" y="220" font-size="11" fill="currentColor" font-weight="600" opacity="0.7">attention</text>
  <text x="645" y="234" font-size="11" fill="currentColor" font-weight="600" opacity="0.7">sub-layer</text>

  <rect x="270" y="122" width="180" height="32" fill="rgba(245,166,35,0.18)" stroke="#f5a623" stroke-width="1.5"/>
  <text x="360" y="143" text-anchor="middle" font-size="12" fill="currentColor">LayerNorm 1</text>
  <text x="465" y="142" font-size="11" fill="currentColor" opacity="0.65" font-style="italic">tidy-up</text>
  <line x1="360" y1="154" x2="360" y2="172" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-block-overview)"/>

  <rect x="270" y="176" width="180" height="32" fill="rgba(245,166,35,0.18)" stroke="#f5a623" stroke-width="1.5"/>
  <text x="360" y="197" text-anchor="middle" font-size="12" fill="currentColor">QKV projection</text>
  <text x="465" y="196" font-size="11" fill="currentColor" opacity="0.65" font-style="italic">d → 3d, split into Q, K, V</text>
  <line x1="360" y1="208" x2="360" y2="226" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-block-overview)"/>

  <rect x="270" y="230" width="180" height="32" fill="rgba(245,166,35,0.18)" stroke="#f5a623" stroke-width="1.5"/>
  <text x="360" y="251" text-anchor="middle" font-size="12" fill="currentColor">multi-head attention</text>
  <text x="465" y="250" font-size="11" fill="currentColor" opacity="0.65" font-style="italic">mixes across positions</text>
  <line x1="360" y1="262" x2="360" y2="280" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-block-overview)"/>

  <rect x="270" y="284" width="180" height="32" fill="rgba(245,166,35,0.18)" stroke="#f5a623" stroke-width="1.5"/>
  <text x="360" y="305" text-anchor="middle" font-size="12" fill="currentColor">output projection</text>
  <text x="465" y="304" font-size="11" fill="currentColor" opacity="0.65" font-style="italic">d → d</text>
  <line x1="360" y1="316" x2="360" y2="332" stroke="currentColor" stroke-width="1.5"/>

  <circle cx="360" cy="342" r="11" fill="rgba(150,150,150,0.18)" stroke="currentColor" stroke-width="1.5"/>
  <text x="360" y="347" text-anchor="middle" font-size="14" fill="currentColor" font-weight="600">+</text>

  <line x1="360" y1="353" x2="360" y2="372" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-block-overview)"/>
  <text x="465" y="368" font-size="11" fill="currentColor" opacity="0.65" font-family="ui-monospace,monospace">[N × d]</text>

  <circle cx="360" cy="378" r="3.5" fill="currentColor"/>
  <line x1="360" y1="381" x2="360" y2="396" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-block-overview)"/>
  <path d="M 360 378 L 190 378 L 190 620 L 349 620" fill="none" stroke="currentColor" stroke-width="1.5" stroke-dasharray="4 4" marker-end="url(#arr-block-overview)"/>
  <text x="135" y="498" font-size="11" fill="currentColor" opacity="0.65" font-style="italic">residual</text>

  <path d="M 615 400 L 635 400 L 635 600 L 615 600" fill="none" stroke="currentColor" stroke-width="1" opacity="0.45"/>
  <text x="645" y="498" font-size="11" fill="currentColor" font-weight="600" opacity="0.7">FFN</text>
  <text x="645" y="512" font-size="11" fill="currentColor" font-weight="600" opacity="0.7">sub-layer</text>

  <rect x="270" y="400" width="180" height="32" fill="rgba(245,166,35,0.18)" stroke="#f5a623" stroke-width="1.5"/>
  <text x="360" y="421" text-anchor="middle" font-size="12" fill="currentColor">LayerNorm 2</text>
  <text x="465" y="420" font-size="11" fill="currentColor" opacity="0.65" font-style="italic">tidy-up</text>
  <line x1="360" y1="432" x2="360" y2="450" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-block-overview)"/>

  <rect x="270" y="454" width="180" height="32" fill="rgba(245,166,35,0.18)" stroke="#f5a623" stroke-width="1.5"/>
  <text x="360" y="475" text-anchor="middle" font-size="12" fill="currentColor">FFN-up</text>
  <text x="465" y="474" font-size="11" fill="currentColor" opacity="0.65" font-style="italic">d → 4d</text>
  <line x1="360" y1="486" x2="360" y2="504" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-block-overview)"/>

  <rect x="270" y="508" width="180" height="32" fill="rgba(245,166,35,0.18)" stroke="#f5a623" stroke-width="1.5"/>
  <text x="360" y="529" text-anchor="middle" font-size="12" fill="currentColor">activation (GeLU)</text>
  <text x="465" y="528" font-size="11" fill="currentColor" opacity="0.65" font-style="italic">pointwise nonlinearity</text>
  <line x1="360" y1="540" x2="360" y2="558" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-block-overview)"/>

  <rect x="270" y="562" width="180" height="32" fill="rgba(245,166,35,0.18)" stroke="#f5a623" stroke-width="1.5"/>
  <text x="360" y="583" text-anchor="middle" font-size="12" fill="currentColor">FFN-down</text>
  <text x="465" y="582" font-size="11" fill="currentColor" opacity="0.65" font-style="italic">4d → d</text>
  <line x1="360" y1="594" x2="360" y2="610" stroke="currentColor" stroke-width="1.5"/>

  <circle cx="360" cy="620" r="11" fill="rgba(150,150,150,0.18)" stroke="currentColor" stroke-width="1.5"/>
  <text x="360" y="625" text-anchor="middle" font-size="14" fill="currentColor" font-weight="600">+</text>

  <line x1="360" y1="631" x2="360" y2="648" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-block-overview)"/>

  <rect x="270" y="652" width="180" height="36" fill="rgba(74,144,226,0.20)" stroke="#4a90e2" stroke-width="1.5"/>
  <text x="360" y="675" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600">output</text>
  <text x="465" y="674" font-size="11" fill="currentColor" opacity="0.7" font-family="ui-monospace,monospace">[N × d]</text>
</svg>

这两半就是 block 干的两件主要的事：一个 **attention** sub-layer，一个 **FFN**（feed-forward network）sub-layer。其它那些零件（LayerNorm、activation、`+`）是小一些的胶水。

每个零件大致在干什么：

- **LayerNorm** 是一个归一化步骤 —— 对 tensor 的每一行，把里面的数重新缩放，让它们的均值和方差落在一个干净的范围里。便宜、按行做、纯粹是为了在数字穿过很多层之后不让它们漂到奇怪的数量级去。可以当成一个"整理"步骤。
- **residual `+`** 的意思是：把进入这一半之前的东西和这一半算出来的东西加在一起。所以每一半算出来的其实是一个 **delta** —— 在已有表示上做一次精炼，而不是整个换掉。这就是我们能摞很多个 block 而信号一路不糊的原因。
- **QKV projection** 就是三个 linear layer 合并成一次大 matmul。它给 input 套三个不同的 weight matrix，产出三个 tensor —— Q（queries）、K（keys）、V（values）—— 每个形状都是 `[N × d]`。
- **Multi-head attention** 是整个模型里唯一让信息在 token 之间流动的步骤。它是 block 的主角 —— §4 讲它真正在算什么，§5 讲为什么前面要加 "multi-head"。
- **output projection** 是最后一个 linear layer，把 attention 的输出整合成能被 residual `+` 直接吸收的形态。
- **FFN-up** 和 **FFN-down** 是中间夹一道非线性的两个 linear layer。它俩合作把每个 token 的 `d` 维表示先扩到 `4d`、过一道按元素的非线性、再压回 `d`。不在 token 之间混 —— 每个 token 各自处理自己。

同样的形状进、同样的形状出 —— §2 那条口诀。摞很多个，就是模型的主体。

---

## 4. attention 到底在算什么

"attention 在 position 之间混合"这句话我们说过好几遍了，但一直没讲*怎么*混合。这一节补上。

对每个 position，模型从这个位置 `[d]` 维的那一行里生出三个向量：

- 一个 **query** `Q` —— *"我在找什么？"*
- 一个 **key** `K` —— *"我能提供什么？"*
- 一个 **value** `V` —— *"如果你决定关注我，这是我想传过去的实际内容"*

（这正是 QKV projection 在做的事 —— 三个 linear layer，每个负责 Q、K、V 之一，融合成一次 matmul。）

要更新 position `i` 那一行，模型做三件事：

1. **算 score。** 拿 `i` 的 query 跟*每一个* position 的 key 做点积。点积大 = 两个向量方向接近 = "这个 position 对 `i` 来说有意思"。点积小（或者负数）= 不感兴趣。最后得到 `N` 个分数 —— 每个 position 一个。
2. **score 变成权重。** 把这些分数过一道 softmax，得到 **attention weight** —— 全是正数、加起来等于 1。位置 `j` 上权重高，意思就是*"i 很关心 j"*；权重低，就是*"i 基本上忽略 j"*。
3. **对 value 做加权平均。** 拿这些权重，对每个 position 的 value 向量做加权求和。这个和，就是 `i` 这一行更新后的表示。

一句话：position `i` 的新一行，是所有 position 的 value 向量的加权平均；权重由 `i` 的 query 跟每个 position 的 key 的匹配度决定。

就这 —— 这就是 attention 全部的机械内容。block 里其它一切（LayerNorm、FFN、residual）都是为了支撑*这一件事*存在的基础设施。它也是整个模型里**唯一**让信息在 token 之间流动的步骤。把 attention 拿掉，模型就分不清 "fox" 和 "the" 在不在同一个句子里了。

这套机制后面还要再补两个细节：

- **§5 —— Heads。** attention 不是在 `d` 维的整段 feature 上跑*一次*的，而是在不同的 feature 切片上*并行跑多次*。
- **§6 —— Causal mask。** position `i` 其实不能 attend 到*所有* position，只能看 `j ≤ i`。为什么要这样，§6 讲。

FFN sub-layer 相比之下简单很多：每一行都过同样的两个 linear layer 加一道非线性，跟其它行互不相干。FFN 不在 position 之间混 —— 那是 attention sub-layer 的活。

所以每个 transformer block 的节奏就是：**position 之间混（attention），然后 feature 之间混（FFN）。** 重复 `L` 次。

---

## 5. Heads

关于 attention 还有一件小事，但后面会很重要：它不是在 `d` 维整段 feature 上*跑一次*，而是切成 `h` 段并行跑 `h` 次。

QKV projection 生出形状都是 `[N × d]` 的 Q、K、V 之后，我们沿 feature 维把每个*reshape* 成 `h` 组，每组宽度 `d_head = d / h`。每一组就是一个 **head**。每个 head 在自己那一片 feature 上跑一遍 §4 的 attention —— 自己的 query、自己的 key、自己的 value。所有 head 的输出再 concat 回 `[N × d]`，喂给 output projection。

<svg viewBox="0 0 760 200" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;font-family:system-ui,sans-serif;display:block;margin:1.5rem auto">
  <defs>
    <marker id="arr-heads" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
      <path d="M 0 0 L 8 4.5 L 0 9 Z" fill="currentColor"/>
    </marker>
  </defs>
  <text x="380" y="22" text-anchor="middle" font-size="14" fill="currentColor" font-weight="600">multi-head attention: reshape, per-head, concat</text>

  <rect x="10" y="78" width="130" height="48" fill="rgba(74,144,226,0.20)" stroke="#4a90e2" stroke-width="1.5"/>
  <text x="75" y="98" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.75">Q, K, V</text>
  <text x="75" y="116" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600" font-family="ui-monospace,monospace">[N × d]</text>

  <line x1="140" y1="102" x2="178" y2="102" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-heads)"/>
  <text x="160" y="90" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.7" font-style="italic">reshape</text>

  <rect x="180" y="78" width="160" height="48" fill="rgba(120,180,140,0.20)" stroke="#78b48c" stroke-width="1.5"/>
  <text x="260" y="98" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.75">h heads, each d_head wide</text>
  <text x="260" y="116" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600" font-family="ui-monospace,monospace">[N × h × d_head]</text>

  <line x1="340" y1="102" x2="398" y2="102" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-heads)"/>
  <text x="370" y="90" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.7" font-style="italic">attention (§4)</text>

  <rect x="400" y="78" width="160" height="48" fill="rgba(120,180,140,0.20)" stroke="#78b48c" stroke-width="1.5"/>
  <text x="480" y="98" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.75">h attention outputs</text>
  <text x="480" y="116" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600" font-family="ui-monospace,monospace">[N × h × d_head]</text>

  <line x1="560" y1="102" x2="598" y2="102" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr-heads)"/>
  <text x="580" y="90" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.7" font-style="italic">concat</text>

  <rect x="600" y="78" width="150" height="48" fill="rgba(74,144,226,0.20)" stroke="#4a90e2" stroke-width="1.5"/>
  <text x="675" y="98" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.75">to output proj</text>
  <text x="675" y="116" text-anchor="middle" font-size="12" fill="currentColor" font-weight="600" font-family="ui-monospace,monospace">[N × d]</text>

  <text x="380" y="160" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.7" font-style="italic">each head runs §4's attention algorithm on its own slice — independently of the others</text>
</svg>

真实模型里 `h` 和 `d_head` 一定凑出来正好等于 `d` —— 一般是几十个 head，每个一百多宽。

模型设计角度看：不同的 head 可以学着去关注不同*种类*的东西。有些 head 最后在追踪短距离的句法关系（"这个代词到底指代哪个词？"），有些跟踪更远距离的模式。多个 head = 多个看 "应该看哪里" 的视角。

系统层面就更直白：**head 之间是独立的。** Head 0 在 attention 里不和 Head 1 说话。每个 head 在自己那块 feature 上各算各的，各出各的输出。

这种独立性只是模型设计的一条性质而已 —— 但它对*后面所有事情*都是承重墙。Article 02 直接利用这条性质，把整个模型一刀切到两张 GPU 上：一半的 head 在一张卡，另一半的 head 在另一张卡，attention 期间它们之间根本不需要通信。"模型大到一张 GPU 装不下怎么办"的整个故事，起点就在这里。

---

## 6. Causal mask

attention 里还有一条规则没讲，但它是必不可少的：position `i` 在 attend 的时候，*只能*看 `j ≤ i` 的位置。`j > i` 的位置会被 mask 掉 —— 它们的 attention score 在过 softmax 之前会被强行置成 `−∞`，过完 softmax 权重就变成 0，对 `i` 的 output 没有任何贡献。

为什么要有这条规则，原因来自训练。模型是按"预测下一个 token"一条一条训练的：喂一段序列进去，让模型从每个 token 之前的所有东西去预测它的下一个 token。如果 position `i` 在 attention 里能偷看 position `i+1`，那它就等于可以*作弊*，直接读答案。mask 就是用来强制"不许往后看"的。

mask 还有两个后续影响值得专门点名，后面都会用到。

第一，它让 Part III 里那个生成循环成立：position `N+1` 的 token 只依赖于 token 1..N，反过来不会。所以我们可以按顺序一个一个生成新 token，从来不需要回头修改一个已经算好的 token。这条性质让"一个一个 token 地生成长回复"这件事根本能成立。

第二 —— 也是更大的那个 —— mask 意味着旧 token 的活*永远不需要重做*。position 5 的 hidden state，无论整段序列是 5 个 token 长还是 500 个 token 长，都是同一个；后面任何新 token 都伸不回来把它改了。这种"算完就不再变"的性质，是我们能不能*想到* "把之前算过的存下来下次复用，而不是每次 forward 都从头算" 的前提。没有 mask，每来一个新 token 就要把前面所有东西重新过一遍。有了 mask，我们才能想着按顺序处理 token，把已经算过的*记住*就好 —— 这正是 §10 会落到的问题，也是后面整个系列里最承重的一类优化之一。

---

## 7. 一张图把整个 block 装进去

到这里我们已经把一个 transformer block 的所有零件都打开过了 —— 两个一半（§3）、attention 的 Q/K/V 机制（§4）、按 head 切（§5）、causal mask（§6）。下面这张图把它们按一次完整的执行画出来，每一步都标了 tensor 形状。

先扫一遍感受整体流向，之后系列里看到*"那个 `[h × N × N]` 的 score matrix"*或者*"按 head 切开的 reshape"*时，回来对一下 —— 你要在脑子里想象的就是这张图。

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

有三点值得停下来看一眼：

- **形状从 `[N × d]` 进，从 `[N × d]` 出** —— §2 那条口诀。在一个 block 内部，tensor 会短暂地变成别的形状（FFN 中间是 `[N × 4d]`，attention score 是 `[h × N × N]`） —— 但这些都是*瞬态*的。block 最后总会回到 `[N × d]`，这样下一个 block 才能接得上。
- **`[h × N × N]` 这个 score matrix 是会让人吃惊的那个。** 它的大小按**序列长度的平方**长。`N` 小的时候没事，`N` 一大就难处理 —— 长序列的代价最后就栽在这里。现在留意一下，后面的文章会回来收拾它。
- **每个 residual `+` 把那一半的 input 重新加回 output。** 所以每一半算的其实是 *delta* —— 一次微调，而不是把整段表示整个换掉。这就是为什么我们能摞很多 block 而信号不崩。

---

## Part III —— 用模型来生成

## 8. 一次 forward 给你一个 token

§1 那个模型，吃 `N` 个 token，吐回一个对下一个 token 的概率分布。**一个** token。不是一整句，连半句也不是 —— 就一个对下一个 token 的猜测。

但我们已经习惯 LLM 一段一段地回复。一次只能生成一个 token 的模型，怎么吐出一整段？跟你猜的一样：反复跑，把自己的 output 当成下一次的 input 喂回去。

具体来说：

1. 起点：prompt —— 一段长度为 `N` 的序列。
2. 跑一次 forward。得到下一个 token（也就是位置 `N+1`）应该是什么的分布。
3. 从这个分布里采样（或者直接挑最高概率那个，"argmax"）。位置 `N+1` 上就有了一个 token。
4. 把它追加到序列后面。序列长度变成 `N+1`。
5. 在*完整* 的 `N+1` 长序列上再跑一次 forward。得到位置 `N+2` 的分布。
6. 采样、追加。序列长度变成 `N+2`。
7. 重复，直到模型采样到一个特殊的 **end-of-sequence** token（训练时模型就被教会：认为回答结束时，发出这个 token），或者撞到你设定的长度上限。

循环长这样：

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

数学上讲，整个生成流程就是这样。你用过的任何基于 LLM 的系统，吐出来的每一个 token，都是从一个长这样的循环里出来的。

---

## 9. 第一个让人不舒服的观察

我们走一遍：从一个长度为 `N` 的 prompt 开始，生成 `K` 个新 token，总成本是多少。

- Forward 1 跑在 prompt 上：长度 `N`。
- Forward 2 跑在 prompt + 1 个新 token 上：长度 `N+1`。
- Forward 3：长度 `N+2`。
- …
- Forward `K`：长度 `N+K−1`。

每一次 forward 几乎都把上一次做过的事*再做一遍*。Forward 2 input 的前 `N` 个 token 和 Forward 1 input *一模一样* —— 但模型还是从头在每一个 position 上跑了一遍每个 block，就像它从没见过这些 token 一样。

总的算下来，工作量大致按 `(N + K)² / 2` 涨 —— 总序列长度的平方。其中绝大部分都是*在重新计算根本没变的东西*。在序列末尾加一个新 token，并不会改变前面任何 token 的表示。前面那些 token 还是原来那个 prompt，加上这次之前已经采样出来的那几个 token 而已。它们身上没有任何东西需要重新算。

于是一个很显然的问题就挂在那里：**这些重算真的有必要吗？** 显然没有。但是不重算也不是白来的 —— 这意味着我们要在两次 forward 之间存某种中间状态。然后这又冒出一串新问题：到底要存什么状态？放哪？它有多大？随着对话变长它怎么涨？

这种问题，正是这个系列后面要拆开来研究的。

---

## 10. 一张问题地图

后面这个系列其实就在追两条主线，关于"怎么*运行*一个 LLM"的绝大多数实际问题，都能归到其中一条。

**主线 1 —— 让一次 forward *装得下*。** 一次穿过这摞 block 的 forward，可能在好几条尺度上都"太大"了：weight 装不下一张 GPU、算一次时间太久、attention 内部太吃内存。这条主线下的文章，主题都是*把工作在空间上切开*，让一次 forward 能落在你手里这套硬件上跑下来。

- **模型本身可以很大。** 摞够多 block（`L` 大）、`d` 又够宽，光是 weight 就装不下一张 GPU。怎么把一次 forward 切到多张 GPU 上？*(Article 02 和 03 —— 用的正是我们在 §5 搭起来的 head 独立性。)*
- **prompt 本身可以很大。** §7 那个 `[h × N × N]` 的 score matrix 按**序列长度的平方**涨。prompt 一长，要么内存爆，要么把 GPU 钉太久。我们能不能把 prompt 分块处理，或者用更聪明的方式去算 attention？

**主线 2 —— 让*循环*跑得快。** 每次 forward 只产出一个 token，§9 已经把最大的成本指出来了：朴素的循环大部分时候都在重做它已经做过的事。这条主线下的文章，主题是*别再重做、把 forward 摊给多个用户、调度谁什么时候跑*。

- **别再重做。** §6 已经把那条性质摆好了：旧 token 的表示一旦算出来就不再变。所以应该可以把它存下来下次复用，而不是每次都重算。这个状态得放在*某个地方* —— 放哪？它有多大？随着对话变长它怎么涨？
- **同时来很多用户。** 真实的 serving 引擎会同时跑很多 prompt，长度都不一样，结束时间也都不一样。怎么让它们共享一次 forward 又不被 padding 拖累？当一些用户还在 token 1、另一些已经到 token 1000 的时候，scheduler 怎么让所有人都在前进？*(Article 04 就是这条线的开头。)*
- **处理一段长 prompt 跟"再多生成一个 token"完全不像。** §9 里每次调用的成本，在你是从头处理一段长输入还是只追加一个 output token 的时候，*差得非常远*。它们的瓶颈落在 GPU 的不同部位。引擎也许就该把它们当作两种不同的 workload —— 甚至放到不同的机器上去跑。

§1–§7 里那个模型，是这两条主线*都在讨论的对象*；§8 那个循环，是它们都想让它在规模化场景下跑得动的东西。系列后面的文章，就是把这些问题一道一道挑出来、逐个回答。
