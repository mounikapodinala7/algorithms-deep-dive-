# Attention as Dynamic Sparse Routing (ADSR)

> **Algorithm Deep Dive — from first principles to executable mechanics**

---

## 0. Why This Document Exists

This file is **not** a literature summary, a rephrased blog, or a high-level conceptual note.

This is a **mechanistic, end-to-end breakdown** of an algorithmic idea:

> **Treating attention as a dynamic sparse routing algorithm**

The goal is to answer, precisely and practically:

* What problem the algorithm *actually* solves
* What hidden assumption standard attention makes
* What the routing algorithm is doing under the hood
* What is learned, what is computed, and when
* Why this works *mechanistically*, not philosophically
* How to implement it without magic

---

## 1. The Real Problem Behind Attention

### 1.1 What Standard Attention Assumes (Implicitly)

Standard self-attention assumes:

> Every token is potentially useful to every other token.

Algorithmically:

For a sequence of length `N`, for each query token `q_i`, we compute:

[
\text{Attention}(q_i) = \sum_{j=1}^{N} \alpha_{ij} v_j
]

This means **N × N token interactions**, regardless of relevance.

This is not a modeling choice — it is a *computational convenience*.

---

### 1.2 The Hidden Inefficiency

In real data:

* Tokens form **semantic clusters**
* Most interactions are **low-value or redundant**
* Useful communication is **structured and sparse**

Yet dense attention:

* Computes similarities it will later down-weight
* Spends FLOPs before deciding relevance

**The inefficiency is not attention — it is late decision-making.**

---

## 2. Algorithmic Reframing

### 2.1 Core Question the Algorithm Answers

> **Before computing attention, which tokens should even be allowed to interact?**

This is a *routing* problem, not a weighting problem.

---

### 2.2 Algorithm Definition (Plain Language)

> For each query token, learn a low-cost decision function that selects a small subset of keys. Run full attention only on that subset.

This decomposes attention into two distinct stages:

1. **Routing (selection)** — discrete, sparse, cheap
2. **Attention (aggregation)** — dense, expensive, precise

---

## 3. Formal Algorithm Structure

We define two parallel representational spaces:

| Space               | Purpose                |
| ------------------- | ---------------------- |
| **Routing space**   | Decide connectivity    |
| **Attention space** | Compute content mixing |

These spaces are learned jointly but serve different roles.

---

## 4. Routing Phase (The Actual Algorithm)

### 4.1 Routing Projections

Each token is projected into a **low-dimensional routing space**:

[
r_q = W_r^q q \quad , \quad r_k = W_r^k k
]

Where:

* `dim(r_q) = dim(r_k) = d_r`
* `d_r << d_model`

This space is intentionally **coarse**.

It exists only to answer: *"Should these tokens communicate?"*

---

### 4.2 Routing Score Computation

Routing scores are computed using a cheap similarity:

[
s_{ij} = r_q^{(i)} \cdot r_k^{(j)}
]

Properties:

* Lower precision than attention
* Much cheaper
* No softmax

This score is **not** used for weighting — only for selection.

---

### 4.3 Discrete Top-K Routing

For each query token `i`, select:

[
\mathcal{R}_i = \text{TopK}*j(s*{ij})
]

This produces a **routing set**:

* Size = `K`
* Hard sparsity
* Input-dependent

Tokens outside this set:

* Are never attended to
* Never consume compute
* Never receive gradients from this query

This is the **routing decision**.

---

## 5. Attention Phase (Unmodified but Constrained)

After routing, standard attention is applied *only* to selected tokens:

[
\alpha_{ij} = \text{softmax}_{j \in \mathcal{R}_i}(q_i^T k_j)
]

[
o_i = \sum_{j \in \mathcal{R}*i} \alpha*{ij} v_j
]

No approximations here — precision is preserved **where it matters**.

---

## 6. What Is Actually Learned

### 6.1 Learnable Components

| Component             | Role              |
| --------------------- | ----------------- |
| `W_r^q`, `W_r^k`      | Routing geometry  |
| `W_q`, `W_k`, `W_v`   | Content attention |
| (Optional) Router MLP | Nonlinear routing |

There are **no hand-coded rules**.

---

### 6.2 How Routing Learns (Mechanism)

Training dynamics:

* Bad routing → missing information
* Missing information → higher loss
* Gradients reshape routing space

Result:

> Tokens that must interact collapse together in routing space.

This is **emergent structure**, not imposed sparsity.

---

## 7. Why This Is a Routing Algorithm (Not Just Sparse Attention)

Key distinction:

| Traditional Sparse Attention | ADSR                     |
| ---------------------------- | ------------------------ |
| Fixed or heuristic masks     | Learned, dynamic routing |
| Sparsity after scoring       | Sparsity before scoring  |
| Still computes dense scores  | Avoids dense computation |

ADSR changes **the order of computation**, not just the mask.

---

## 8. Computational Characteristics

Let:

* `N` = sequence length
* `d_r` = routing dimension
* `K` = routed tokens

### Costs

* Routing: `O(N^2 d_r)` (cheap, low-dim)
* Attention: `O(N K d)`

With `d_r << d` and `K << N`, this is practically sub-quadratic.

---

## 9. Failure Modes & Stability

### 9.1 Routing Collapse

If routing becomes too sharp:

* Tokens attend to themselves only

**Fixes**:

* Routing temperature
* Entropy regularization
* Random exploration tokens

---

### 9.2 Missed Long-Range Dependencies

If routing is too local:

**Fixes**:

* Multi-head routing
* Global memory tokens
* Multi-hop routing passes

---

## 10. Minimal Executable Sketch (Conceptual)

```python
rq = q @ Wrq      # [N, dr]
rk = k @ Wrk      # [N, dr]

scores = rq @ rk.T
routes = scores.topk(K, dim=-1).indices

k_sel = gather(k, routes)
v_sel = gather(v, routes)

attn = softmax((q.unsqueeze(1) * k_sel).sum(-1))
out = (attn.unsqueeze(-1) * v_sel).sum(1)
```

This is the **entire algorithm**.

---

## 10.1 Execution Diagram (Token-Level)

```
Input Tokens
T1   T2   T3   T4   T5   T6
 |    |    |    |    |    |
 v    v    v    v    v    v
Routing Projections (low-dim)
 r1   r2   r3   r4   r5   r6

Query = T3

Routing Scores:
 r3·r1  r3·r2  r3·r3  r3·r4  r3·r5  r3·r6
   |      |      |      |      |      |
   +------+------+------+------+
          Top-K Selection (K = 2)

Selected Routes:
 T2, T5

Full Attention:
 T3 ↔ T2
 T3 ↔ T5
```

Dense attention computes all interactions.

ADSR computes only the routed ones.

---

## 10.2 Computation Flow Diagram

```
┌──────────────┐
│ Input Tokens │
└──────┬───────┘
       │
       ▼
┌───────────────────┐
│ Routing Projection│
└──────┬────────────┘
       │
       ▼
┌───────────────────┐
│ Routing Scores    │
└──────┬────────────┘
       │
       ▼
┌───────────────────┐
│ Top-K Routing     │
└──────┬────────────┘
       │
       ▼
┌───────────────────┐
│ Full Attention    │
└───────────────────┘

Routing happens before attention computation.

```

## 11. What This Algorithm Ultimately Is

> **A learned, input-dependent token connectivity algorithm that turns attention into conditional computation.**

Attention becomes:
- Sparse
- Structured
- Scalable

Not by approximation — but by **learning where computation is necessary**.

---

## 12. Why This Belongs in Algorithm Deep Dive

This algorithm:
- Exposes hidden assumptions in Transformers
- Separates decision from computation
- Connects routing, MoE, and attention
- Is implementable without new primitives

It is an **algorithmic shift**, not a trick.

---

## 13. One-Line Summary (Zero Fluff)

> Attention as Dynamic Sparse Routing is an algorithm that learns which token interactions are allowed before attention is computed, enabling conditional, sparse, and scalable sequence modeling.

---

**End of Deep Dive**

```
