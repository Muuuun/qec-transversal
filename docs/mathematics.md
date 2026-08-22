# Mathematical methods

This document states the objects the package computes, the algorithms it uses,
and — for each — exactly what "complete" means. It is a specification of the
implementation, not a paper: results that are established in the literature are
attributed in [`related_work.md`](related_work.md), and nothing here is claimed
as new mathematics unless [`related_work.md`](related_work.md) says so
explicitly.

Conventions throughout: binary **row** vectors, physical coordinate order

```text
(X_0, ..., X_{n-1} | Z_0, ..., Z_{n-1}),
```

logical coordinate order

```text
(Xbar_0, ..., Xbar_{k-1} | Zbar_0, ..., Zbar_{k-1}),
```

and matrices acting on the right (`v -> v M`).

---

## 1. Stabilizer codes as symplectic subspaces

Modulo phases, a Pauli operator on `n` qubits is a vector in
$\mathbb{F}_2^{2n}$, and two Paulis commute exactly when their **symplectic
product** vanishes:

$$\langle u, v\rangle \;=\; u \, \Omega \, v^{T}, \qquad
\Omega = \begin{pmatrix} 0 & I_n \\ I_n & 0\end{pmatrix}.$$

A stabilizer code is a self-orthogonal subspace $S \subseteq \mathbb{F}_2^{2n}$
of dimension $n - k$: $\langle s, s'\rangle = 0$ for all $s, s' \in S$. Its
centraliser $S^{\perp}$ (the normaliser of the stabilizer group, modulo phases)
has dimension $n + k$, and the logical Pauli group modulo stabilizers is the
quotient

$$S^{\perp}/S \;\cong\; \mathbb{F}_2^{2k},$$

itself a non-degenerate symplectic space. The implementation stores a
symplectically **paired** basis $(\bar X_1, \dots, \bar X_k, \bar Z_1, \dots,
\bar Z_k)$ of a complement of $S$ in $S^{\perp}$, produced by a symplectic
Gram–Schmidt pass (`utils/symplectic.py`), so that
$\langle \bar X_i, \bar Z_j\rangle = \delta_{ij}$ and all other pairings vanish.

A **CSS** code is the special case
$S = \operatorname{rowspan}[H_X \mid 0] \oplus \operatorname{rowspan}[0 \mid H_Z]$
with $H_X H_Z^{T} = 0$. Keeping the two families separate is what makes the
specialised solvers of §6 cheaper than the general machinery of §3–§5; it
changes nothing mathematically.

Code objects live in `codes/`; they carry the representation and nothing else.

---

## 2. Physical gates as symplectic matrices

A Clifford unitary $U$ modulo Paulis and global phase acts on Pauli labels by a
matrix $M \in \mathrm{Sp}(2n, 2)$, i.e. $M \Omega M^{T} = \Omega$. $U$
**preserves the code** iff

$$S M \subseteq S,$$

which (for invertible $M$) is the same as $SM = S$.

A gate *ansatz* is a restriction on the shape of $M$. Fix a partition
$\mathcal{P} = \{C_1, \dots, C_r\}$ of the physical qubits. The **depth-one
$\mathcal{P}$-local** ansatz is

$$M \;=\; \bigoplus_{C \in \mathcal{P}} M_C, \qquad
M_C \in \mathrm{Sp}(2|C|, 2),$$

where $M_C$ acts on the coordinates $(x_i \mid z_i)_{i \in C}$. Special cases:

| partition | ansatz | module |
|---|---|---|
| all cells singletons | one arbitrary single-qubit Clifford per qubit ("strict", "site-dependent transversal") | `ansatz/strict.py` |
| cells of size two | fixed-matching two-local layer | `ansatz/partition.py` |
| one cell = all qubits | the whole code-preserving Clifford group | `ansatz/partition.py` |

Qubit permutations are *not* of this form (they move coordinates between
cells); they are handled separately in §7.

### 2.1 Gates across code blocks

The classical notion of "transversal" — one gate per corresponding-qubit tuple
across $\ell$ blocks, e.g. transversal CNOT — needs no new machinery. Take the
joint code $S^{\otimes \ell}$ on $\ell n$ qubits (block $b$ occupying qubits
$[bn, (b+1)n)$, parameters adding: $n \mapsto \ell n$, $k \mapsto \ell k$) and
the partition into cells

$$C_i = \{\, i,\; n+i,\; \dots,\; (\ell-1)n + i \,\}, \qquad i = 0, \dots, n-1,$$

each of size $\ell$, so the per-cell group is $\mathrm{Sp}(2\ell, 2)$.
Everything in §3–§6 then applies verbatim, including the completeness
semantics. `transversal_clifford_across_blocks` is that reduction.

Two certified examples of opposite sign, both `COMPLETE`: two Steane blocks
realise the **entire** $\mathrm{Sp}(4,2)$ logical Clifford group on their two
logical qubits (order 720) from depth-one two-block gates alone, while two
$[[5,1,3]]$ blocks reach order 18 of that same 720 and no more — a *certified
negative*, not an exhausted search.

Caveat on reading the output: the joint logical basis comes from the usual
symplectic Gram–Schmidt pass, so it need not be the "block $b$, logical qubit
$j$" basis. Group orders are basis-independent; to read an individual logical
matrix as a two-block gate, transport it into the product basis first.

---

## 3. The preservation algebra

The obstruction to solving $SM \subseteq S$ directly is that
$\prod_C \mathrm{Sp}(2|C|,2)$ is a group, not a linear space. The
linearisation is to **drop invertibility**:

$$\boxed{\;
A_{\mathcal{P}}(S) \;=\;
\Big\{\, M = \bigoplus_{C \in \mathcal{P}} M_C \;:\; S M \subseteq S \,\Big\}
\;\subseteq\; \bigoplus_{C \in \mathcal{P}} M_{2|C|}(\mathbb{F}_2). \;}$$

Two facts make this the right object.

**It is a linear subspace.** The condition $SM \subseteq S$ is linear in the
entries of $M$, so $A_{\mathcal{P}}(S)$ is the kernel of one $\mathbb{F}_2$
matrix and is computed by a single Gaussian elimination.

**It is an algebra.** If $S M \subseteq S$ and $S M' \subseteq S$ then
$S(MM') \subseteq S$; the identity is $\mathcal{P}$-block diagonal and
preserves $S$. So $A_{\mathcal{P}}(S)$ is a finite-dimensional unital
associative $\mathbb{F}_2$-algebra, of dimension at most
$\sum_C (2|C|)^2$.

The code-preserving gates of the ansatz are therefore the **symplectic units**:

$$\boxed{\;
G_{\mathcal{P}}(S) \;=\;
A_{\mathcal{P}}(S)^{\times} \;\cap\; \prod_{C \in \mathcal{P}} \mathrm{Sp}(2|C|, 2).
\;}$$

### 3.1 Singletons: the symplectic condition is free

Over $\mathbb{F}_2$ a $2\times 2$ matrix has determinant $ad + bc$, and

$$M \Omega_1 M^{T} = (\det M)\,\Omega_1 \quad\text{for } \Omega_1 = \begin{pmatrix}0&1\\1&0\end{pmatrix},$$

so $\mathrm{Sp}(2,2) = \mathrm{SL}(2,2) = \mathrm{GL}(2,2)$ — the six
single-qubit Cliffords modulo Paulis, isomorphic to $S_3$ acting on
$\{X, Y, Z\}$. Hence for the singleton partition the intersection is vacuous:

$$G_{\text{singletons}}(S) \;=\; A_{\text{singletons}}(S)^{\times}.$$

**The strict site-dependent transversal Clifford group of any stabilizer code
is exactly the unit group of its preservation algebra.** This is what turns a
search problem into an algebra problem.

For $|C| \ge 2$ the identity above fails ($\mathrm{Sp}(2m,2) \subsetneq
\mathrm{GL}(2m,2)$ for $m \ge 2$), so the form condition must be imposed
explicitly. It is a subgroup condition, so it can be applied *after* the unit
group is known.

### 3.2 The involution, and the σ-stable algebra

A symplectic $M$ satisfies $M \Omega M^{T} = \Omega$, equivalently

$$M^{-1} \;=\; \sigma(M), \qquad \sigma(M) := \Omega M^{T} \Omega .$$

$\sigma$ is an anti-automorphism with $\sigma^2 = \mathrm{id}$ — an
*involution* — so the gate group is the **unitary group of an algebra with
involution**:

$$G_{\mathcal{P}}(S) \;=\; \{\, M \in A^{\times} : \sigma(M)\,M = 1 \,\}.$$

That description is only available if $\sigma$ maps the algebra to itself, and
for $A_{\mathcal{P}}(S)$ **it does not**. The image is identified exactly.
Membership $sM \in S$ is tested against the ordinary dual $S^{\top}$, so for
$s \in S$, $w \in S^{\top}$,

$$s\,\sigma(M)\,w^{T} \;=\; (s\Omega)\, M^{T} (w\Omega)^{T}
\;=\; (w\Omega)\, M\, (s\Omega)^{T},$$

and since $S^{\top}\Omega = N$, the normalizer (both equal
$\ker(H\Omega)$, of dimension $n+k$), the condition $S\,\sigma(M) \subseteq S$
is equivalent to $N M \subseteq N$. Hence

$$\boxed{\;\sigma\big(A_{\mathcal{P}}(S)\big) \;=\; A_{\mathcal{P}}(N).\;}$$

The two coincide only when $S = N$, i.e. $k = 0$. The σ-stable object is
therefore the intersection

$$A'_{\mathcal{P}}(S) \;=\; A_{\mathcal{P}}(S) \,\cap\, A_{\mathcal{P}}(N),$$

and **it loses nothing**: an invertible symplectic $M$ with $SM = S$ also
satisfies $NM = N$ (it preserves symplectic pairings), and its inverse
$\sigma(M)$ lies in $A'$ again, so $M \in A'^{\times}$. Therefore

$$A'_{\mathcal{P}}(S)^{\times} \cap \prod_C \mathrm{Sp}(2|C|,2)
\;=\; A_{\mathcal{P}}(S)^{\times} \cap \prod_C \mathrm{Sp}(2|C|,2).$$

The refinement returns the *same* gate group from a strictly smaller algebra.
The saving is real on multi-qubit cells, where the naive algebra carries a
large σ-asymmetric part no symplectic element ever uses — measured on pair
partitions: $[[4,2,2]]$ $20 \to 16$, $[[6,2,2]]$ $28 \to 20$, $[[8,3,2]]$
$27 \to 22$, `iceberg-8` $36 \to 24$. Three of those cross the enumeration cap,
turning an honest `UNKNOWN` into an exact answer. This is the default;
`partition_algebra(..., refine=False)` recovers $A_{\mathcal{P}}(S)$ itself.

### 3.2b The symplectic cut as an index, not a sweep

Filtering an enumerated $A'^{\times}$ is wasteful — for the $[[8,3,2]]$ pair
partition, $|A'^{\times}| = 393216$ elements are enumerated to find
$|G| = 6144$ — and simply impossible once $|A^{\times}|$ reaches $10^9$, which
it does as soon as a cell holds four qubits. The involution removes the sweep.

The fibers of $\varphi: A^{\times} \to A'$, $\varphi(M) = \sigma(M)M$, are
exactly the left cosets $GM$: if $\varphi(M) = \varphi(WM)$ then
$\sigma(W)W = 1$, i.e. $W \in G$. Hence

$$|G| \;=\; |A^{\times}| \,/\, |\mathrm{im}\,\varphi|,
\qquad \mathrm{im}\,\varphi \subseteq \{a : \sigma(a) = a\}.$$

The image is computable because $\varphi$ is the orbit map of the *congruence
action* of $A^{\times}$ on $A$,

$$a \cdot u \;=\; \sigma(u)\, a\, u,$$

a right action because $\sigma$ is an anti-automorphism. The orbit of $1$ is
$\mathrm{im}\,\varphi$ and its stabiliser is $G$, so orbit-stabiliser *is* the
index formula — and the transversal it produces gives generators of $G$ by
Schreier's lemma, so the route returns a certified generating set, not only a
count. Each orbit step is one $\mathbb{F}_2$ matrix product, since
$a \mapsto \sigma(u) a u$ is linear in $a$ for fixed $u$.

Two facts make this the right trade. The orbit lives in the $\sigma$-symmetric
part of $A$, about half the dimension; and its size is $|A^{\times}|/|G|$, so a
*large* gate group makes the orbit *small* — the sweep and the index are
expensive in opposite regimes. Measured on $[[6,2,2]]$:

| partition | $\dim A$ | $|A^{\times}|$ | orbit | $|G|$ | time |
|---|---|---|---|---|---|
| $(012)(345)$ | 28 | 14 155 776 | 384 | 36 864 | 0.4 s |
| $(0123)(4)(5)$ | 36 | 3 623 878 656 | 6 144 | 589 824 | 1.5 s |
| $(01234)(5)$ | 56 | 1 168 918 299 279 360 | 1 376 256 | 849 346 560 | 7.6 min |

All three were `UNKNOWN` before. Implemented in
`algebra/unitary_group.py`, reached through
`partition_units_via_structure(..., method="phi")` and automatically by
`method="auto"` once the sweep is unaffordable; validated against the sweep on
every partition of width $\le 2$ of `c4-22`, `c6-22`, `cube-832` and `steane`
where both routes terminate (1086 partitions, exact agreement on both the
physical order and the logical image).

What still bounds the route is the *dict* of orbit points, not arithmetic: a
tiny $G$ inside a huge $A^{\times}$ gives an orbit too large to hold, and the
honest exit is `UNKNOWN`. Removing that would need $|G|$ in closed form — a
$\sigma$-adapted Wedderburn decomposition plus Wall's classification of the
unitary groups of the simple factors — which is open work here.

### 3.3 Sparse constraint construction

For a stabilizer row $s$, the image $sM$ is supported inside the closure of
$\operatorname{supp}(s)$ under $\mathcal{P}$. Membership $sM \in S$ is tested
against a basis of $S^{\perp}$ restricted to those columns. A weight-$w$ row
therefore contributes $O(w)$ independent constraints rather than $O(n)$, which
is what keeps the construction affordable for qLDPC codes.
(`algebra/preservation.py`.)

---

## 4. The finite-algebra solver

Given $A = A_{\mathcal{P}}(S)$ as a basis of flat $\mathbb{F}_2$ vectors plus
its multiplication, the package computes generators and the exact order of
$A^{\times}$ through the classical structure sequence

$$A \;\longrightarrow\; J(A) \;\longrightarrow\; A/J(A)
\;\cong\; \prod_{i} M_{d_i}(\mathbb{F}_{q_i})
\;\longrightarrow\; A^{\times},$$

with $J(A)$ the Jacobson radical. None of these algorithms is new — radical
computation in characteristic $p$ is Cohen–Ivanyos–Wales / Friedl–Rónyai, and
the Wedderburn split is textbook. What the package adds is that **every stage
carries a machine-checked certificate**, and that a failed verification returns
`status = "unknown"` rather than an unproven answer.

**Radical (`algebra/radical.py`).** The naive characteristic-zero trace form is
degenerate beyond the radical over $\mathbb{F}_2$ (already for
$\mathbb{F}_2[x]/(x^2)$). The implemented chain uses the
characteristic-polynomial-coefficient forms

$$I_0 = A, \qquad
I_{j+1} = \{\, x \in I_j : c_{2^j}(L_x L_y) = 0 \ \ \forall y \in I_j \,\},$$

with $L_x$ left multiplication and $c_m$ the coefficient of
$\lambda^{d-m}$; the chain reaches $J(A)$ once $2^j > \dim A$. Its output is
treated as a **candidate only**: it is closed into a two-sided ideal and proven
nilpotent by explicit power computation before anything is peeled.

**Nilpotent part.** For a certified-nilpotent ideal $I$ with $I^m = 0$, the
group $1 + I$ has order $2^{\dim I}$ and is generated by $\{1 + n_t\}$ for a
*filtration-adapted* basis $\{n_t\}$ of $I$ (each $n_t$ assigned a level $\ell$
with $n_t \in I^{\ell}$, the level-$\ell$ elements descending to a basis of
$I^{\ell}/I^{\ell+1}$). Adaptedness is necessary, not bookkeeping: an arbitrary
RREF basis can under-generate, and `algebra/radical.py` carries the explicit
three-dimensional counterexample.

**Semisimple part (`algebra/wedderburn.py`).** Semisimplicity of the quotient
is *proven constructively*, never assumed: the centre is split into fields by
minimal-polynomial factoring (Berlekamp), each block is exhibited as
$M_{d}(\mathbb{F}_{2^{e}})$ through an explicitly built action on an
irreducible module whose commutant is verified to be a field, and the dimension
bookkeeping $\sum_i d_i^2 e_i = \dim(A/J)$ must close exactly.

**Order.** Assembled only from certified pieces:

$$|A^{\times}| \;=\; 2^{\dim J(A)} \cdot \prod_i |\mathrm{GL}(d_i, q_i)|,
\qquad |\mathrm{GL}(d,q)| = \prod_{i=0}^{d-1}(q^{d} - q^{i}).$$

Per-block generation is itself certified: randomly drawn block units are fed to
a Schreier–Sims chain and accepted only when the chain reproduces
$|\mathrm{GL}(d,q)|$ exactly.

**Symplectic cut.** For cells of width $\ge 2$, the group $A^{\times}$ is then
enumerated *as a group* (by closure over its certified generators —
$|A^{\times}|$ elements, typically far fewer than $2^{\dim A}$) and filtered by
the per-block condition $M_C \Omega_{|C|} M_C^{T} = \Omega_{|C|}$. Blockwise
symplectic elements are closed under products, so the filtered set is a
subgroup and the count is exact.

---

## 5. From physical gate to logical action

A code-preserving $M$ maps $S^{\perp}$ to itself and fixes $S$ setwise, so it
descends to the quotient:

$$\bar M \in \mathrm{Sp}(2k, 2), \qquad
\bar M \text{ defined by } \;\; \bar v M \equiv \bar v \bar M \pmod S .$$

Concretely (`logical/action.py`), with the paired logical basis $L$ stacked as
$[\bar X; \bar Z]$, the coefficients are read off by symplectic pairing,

$$\bar M_{\text{left}} = (L M)\,\Omega\, \bar Z^{T}, \qquad
\bar M_{\text{right}} = (L M)\,\Omega\, \bar X^{T},$$

and the residue $L M + \bar M L$ must lie in $S$. **That residue check is the
certificate that the descent is well defined**; every backend runs it for every
generator it returns, and a generator that fails it is reported as
uncertified rather than silently used.

Symplecticity of $\bar M$ is a consequence of $M$ being symplectic and the
quotient form being induced, but it is re-checked numerically anyway.

**Group orders.** The subgroup of $\mathrm{Sp}(2k, 2)$ generated by the logical
images is computed by two independent engines (`logical/group.py`):

* a deterministic Schreier–Sims stabilizer chain on row vectors of
  $\mathbb{F}_2^{2k}$, exact far beyond enumeration range;
* breadth-first closure, used as a cross-check at small orders and as a
  fallback.

They are compared where both apply, and a disagreement raises rather than
picks a winner. The target for fullness is

$$|\mathrm{Sp}(2k,2)| \;=\; 2^{k^2}\prod_{i=1}^{k}\,(4^{i} - 1).$$

For $k$ too large for either engine, `logical/recognition.py` implements a
recognition certificate for the specific question "is this group all of
$\mathrm{Sp}(2k,2)$?", based on McLaughlin's classification of irreducible
groups generated by symplectic transvections. Every sub-certificate
(irreducibility, transvection-direction orbit spanning, absence of an invariant
quadratic form, exclusion of the symmetric-group cases by an exact
element-order argument) is verified exactly; the verdict is `full`, `not-full`,
or `inconclusive`, never a guess.

---

## 6. Completeness

Completeness always means **complete for the stated ansatz**. Three levels are
reported, and they are never rounded upward:

| value | meaning |
|---|---|
| `COMPLETE` | the returned set is provably the whole solution set of the ansatz |
| `INCOMPLETE_LOWER_BOUND` | everything returned is certified, but the search was capped, sampled, or scoped to a subgroup; the truth can only be larger |
| `UNKNOWN` | a verification or budget failed; nothing may be concluded in either direction |

Why the framework yields `COMPLETE` at all: the ansatz's solution set is
*exactly* $A_{\mathcal{P}}(S)^{\times} \cap \prod_C \mathrm{Sp}(2|C|,2)$, and
$A_{\mathcal{P}}(S)$ is an exact kernel — no search, no sampling. Computing the
whole unit group (or enumerating the whole algebra) therefore settles the
question; there is nothing left over to miss. A run reports `COMPLETE` when it
finished that computation and every generator passed its certificate.

`UNKNOWN` never becomes "no gate exists". This is the single invariant the code
is most careful about: a capped enumeration, an exceeded node budget, or an
uncertified algebra split all produce `UNKNOWN`, and the corresponding group
order is reported as `None` rather than as a small number.

---

## 7. Specialised solvers

These exploit extra structure. They are faster, expose additional mathematical
content, or reach ansätze the partition framework does not cover — but they do
not define the scope of the package.

### 7.1 CSS strict transversal: the shear families

For a CSS code the two diagonal parameter spaces

$$A_Z = \{a \in \mathbb{F}_2^{n} : a \odot C_X \subseteq C_Z\}, \qquad
A_X = \{b \in \mathbb{F}_2^{n} : b \odot C_Z \subseteq C_X\}$$

($\odot$ = coordinatewise product) are ordinary GF(2) nullspaces. They act as
$U_Z(a) = \prod_{a_i=1}\sqrt{Z}_i$ and $U_X(b) = \prod_{b_i=1}\sqrt{X}_i$, with
logical actions the shears

$$\lambda_Z(a) = \begin{pmatrix} I & L_X \mathrm{diag}(a) L_X^{T} \\ 0 & I\end{pmatrix},
\qquad
\lambda_X(b) = \begin{pmatrix} I & 0 \\ L_Z \mathrm{diag}(b) L_Z^{T} & I\end{pmatrix}.$$

That these two families generate the whole strict class — including layers
mixing $X$ and $Z$ per qubit — is Albert's structure theorem
([arXiv:2608.05688](https://arxiv.org/abs/2608.05688)). The package does not
take it on faith: `ansatz/strict.py` recomputes the same group from the
preservation algebra, and the test suite compares the two against each other
and against brute force over all $6^n$ single-qubit Clifford assignments for
small codes. (`ansatz/strict_css.py`, `tests/test_cross_validation.py`,
`tests/test_completeness.py`.)

### 7.2 Fixed-matching (fold-transversal) layers

Fix an involution $\tau$. A Z-type matching layer is
$U = \prod_{(i,j) \in \mathrm{pairs}(\tau)} CZ_{ij}^{c_{ij}} \prod_i
\sqrt{Z}_i^{a_i}$, whose symplectic action is the shear
$\begin{pmatrix} I & \Sigma \\ 0 & I\end{pmatrix}$ with $\Sigma$ symmetric,
carrying $a$ on the diagonal and $c$ on the matched off-diagonal positions.
Preservation is linear in $(a, c)$, so the complete family is again a GF(2)
kernel. When $\tau$ additionally maps $C_X$ onto $C_Z$ it is a ZX-duality and
the fold-Hadamard (transversal $H$ followed by the permutation) is a further
generator, certified directly from its symplectic action.
(`ansatz/matching.py`.)

Scope note: this classifies the *diagonal* layers on the given matching plus
the fold Hadamard. The Levi (CNOT-network) factor of the fixed-matching group
is computed separately in `ansatz/twofold.py`.

### 7.3 Diagonal gates in the Clifford hierarchy

A depth-one diagonal layer $U(t) = \mathrm{diag}(\omega^{\,t \cdot u})$ with
$\omega = e^{2\pi i/2^{L}}$ is **not** a symplectic object for $L \ge 3$, so the
preservation algebra does not apply. Instead, code preservation becomes a
system of congruences over $\mathbb{Z}_{2^{L}}$.

For a CSS code the layer multiplies the branch $u$ of a codeword by
$\omega^{t \cdot u}$; preservation holds exactly when that phase is constant on
every coset of $C_X$ inside $\ker H_Z$. Using
$t\cdot(u \oplus v) = t\cdot u + t\cdot v - 2\,t\cdot(u \wedge v)$ this closes
into finitely many linear congruences at descending moduli — the "coset
ladder", complete at every level. (`hierarchy/css.py`.)

The solution set is a submodule of $\mathbb{Z}_{2^{L}}^{n}$, computed by
elimination over the local ring $\mathbb{Z}_{2^{L}}$: a pivot of minimal 2-adic
valuation clears its column and is scaled by the exact power of two that keeps
it in the kernel. (`utils/modular.py`.)

For general stabilizer codes `hierarchy/general.py` solves the operator-level
condition; it is **sound** always, and complete when the code has no Z-type
stabilizers, when the CSS ladder applies after conjugation, or when the exact
support-coset enumeration fits under its cap. Anything else is reported
`complete = False`.

The logical action is a phase polynomial: $\phi(g) = t \cdot g \bmod 2^L$ on
logical basis vectors, with degree-2 coefficients $-2\,t\cdot(g_i \wedge g_j)$
and degree-3 coefficients $4\,t\cdot(g_i \wedge g_j \wedge g_l)$. A degree-$d$
monomial with coefficient $c$ sits at hierarchy level
$d + L - 1 - v_2(c)$.

**Completeness certificate.** Kernel generators only prove *soundness*. A
Smith-form certificate $(V, (a_i))$ over $\mathbb{Z}_{2^L}$ is exported so that
an independent checker can verify the returned generators span the *entire*
kernel: it recomputes $M = AV$, checks column $i$ is divisible by $2^{a_i}$ and
that the cofactors $M_i/2^{a_i}$ are independent mod 2, which forces the kernel
to be exactly $\mathrm{span}\{2^{L-a_i} V e_i\}$.
(`certificates/hierarchy.py`.)

### 7.4 Axis frames

By the Zeng–Cross–Chuang structure theorem
([arXiv:0706.1382](https://arxiv.org/abs/0706.1382)), every single-qubit
transversal gate of a stabilizer code is local-Clifford-equivalent to a
frame-diagonal one. Sweeping all $3^n$ per-qubit Pauli-axis frames, conjugating
the code into each and solving the diagonal problem there, therefore closes the
entire single-qubit transversal class at a given level — *provided* the sweep
is exhaustive **and** every reported frame is itself complete. Both conditions
are reported explicitly. (`hierarchy/frames.py`.)

### 7.5 Permutations and monomial gates

Qubit permutations leave the partition framework. Two engines:

* **Tanner-graph automorphisms** (`ansatz/permutation.py`): the automorphism
  group of the coloured Tanner graph of the checks *as given*. Exact for that
  graph, but row-**set** scoped — a symmetry that permutes the row space
  without fixing the given generating rows is invisible. Always reported as a
  lower bound on the row-space group.
* **Characteristic-codeword automorphisms** (`ansatz/codeword_permutation.py`):
  the classical invariant-set reduction underlying Leon's algorithm. Replace the
  arbitrary generating rows by all codewords of weight $\le w$, for the smallest
  $w$ at which they span; weight is permutation-invariant, so the coordinate
  projection of the incidence-graph automorphism group then *equals*
  $\{\pi : C_X \pi = C_X,\ C_Z \pi = C_Z\}$. Large ranks use the
  Brouwer–Zimmermann-style disjoint-information-set enumeration: with $h$
  disjoint systematic bases, enumerating messages of weight $\le t$ on each
  yields every codeword of weight $\le ht + h - 1$, so exactly those weight
  classes are complete.
* **Monomial / GF(4) automorphisms** (`ansatz/monomial.py`): under the
  Calderbank–Rains–Shor–Sloane correspondence a stabilizer code is an additive
  GF(4) code, and permutations combined with per-qubit Cliffords form its
  automorphism group inside $S_3 \wr S_n$. Computed through the CRSS
  three-column binary encoding ($x_i$, $z_i$, $x_i + z_i$ = the axes $X$, $Z$,
  $Y$) and BLISS. Exact when the stabilizer group is small enough to enumerate
  every nonzero element; otherwise row-set scoped, hence a lower bound.

A first-isomorphism-theorem consistency probe relates the monomial group to
the strict group: projecting $G \le S_3 \wr S_n$ onto its $S_n$ part has
kernel exactly the strict local Cliffords preserving the row set, so
$|\text{kernel}| = |G| / |\text{image}|$, which must equal the strict group
order in full-group scope.

### 7.6 One-block generated groups

`logical/generated.py` collects every certified depth-one one-block layer
(strict shears, fold layers over every certified involution, permutation gates)
and certifies the subgroup of $\mathrm{Sp}(2k,2)$ they generate. Involutions
are **sampled**, not enumerated, so the verdict is deliberately one-sided:
reaching $|\mathrm{Sp}(2k,2)|$ is a certificate of fullness, and anything short
of it is a lower bound on what the code admits — never a no-go. This matters
because the corresponding no-go statements in the literature
([arXiv:2602.13395](https://arxiv.org/abs/2602.13395)) are about gadget sets
built on *one fixed* partition or duality.

---

## 8. Sign exactness

Everything above is modulo Pauli operators and global phases. That is sound for
group orders: the sign defect of a Clifford on the stabilizer group is a linear
character, so a Pauli correction always exists and never changes the symplectic
action. It is not, however, a circuit-level claim.

`certificates/signed.py` closes the gap on demand. A code is carried as
$(H, \sigma)$ — symplectic rows plus explicit generator signs — and any
$2n \times 2n$ symplectic matrix is lifted to an exact Stim tableau, conjugated
through the signed generators, given an explicit Pauli correction, and
re-verified to fix every generator with sign $+1$. `certificates/phase.py` runs
this automatically for the strict diagonal generators, distinguishing e.g. a
logical $S$ from $S^{\dagger}$.

Since 0.2.1 this is available uniformly rather than only for diagonal layers.
Every generator record of every ansatz carries its dense $2n \times 2n$
physical symplectic matrix, and

```python
certify_signs(code, result)
```

lifts each one to an exact tableau, solves its Pauli correction, and
re-verifies that the corrected gate fixes every stabilizer generator with sign
$+1$. It returns a `SignCertificate` reporting how many generators were
checked, how many were skipped, and whether all passed.

Two scope notes that still hold and are reported rather than hidden:

* `one_block_clifford_group` collects *logical* actions, not physical layers,
  so its records have nothing to lift. `certify_signs` skips them and says so;
  a certificate with `checked == 0` is never `certified`. The discriminator is
  the matrix shape ($2k \times 2k$ versus $2n \times 2n$), chosen so that a
  logical action can never be silently verified as if it were a gate.
* The published census orders remain symplectic actions modulo Paulis and
  global phases. Sign-exactness is now a call away for every backend, but it is
  a per-result check, not something baked into the group orders — and it
  cannot change them, by the linear-character argument above.

---

## 9. Exact arithmetic

Every computation in this package is exact. There is no floating-point
arithmetic in any GF(2), modular, finite-field, or group computation. The one
place floats appear is `gf2_matmul`, which routes large GF(2) products through
float32 BLAS — exact for inner dimensions below $2^{24}$, since every value is
a small integer — and immediately reduces modulo 2. Wide matrices use a
bit-packed `uint64` elimination kernel, fuzz-tested against the dense path.
