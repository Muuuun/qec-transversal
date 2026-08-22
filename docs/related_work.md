# Related work and positioning

This document says what in `qec-transversal` is an implementation of
established theory, what is an extension, and what may be a genuinely new
computational formulation. It is deliberately conservative: where a boundary is
unclear, it is stated as unclear.

Bibliographic details below were checked against arXiv listings in August 2026.
Where a paper has a journal version the journal reference is given; where the
package's own claims depend on a paper's theorem, that dependence is spelled
out.

---

## 1. Summary of the boundary

### Established theory, implemented here

None of the following is claimed as a contribution of this package. The
package's value for these is that it provides an installable, arbitrary-input
implementation with explicit completeness reporting and exportable
certificates.

* **CSS strict-transversal Clifford structure.** The two diagonal parameter
  spaces $A_Z, A_X$, the shear logical-action formulas, and the theorem that
  these families generate the entire strict class are Albert's
  ([arXiv:2608.05688](https://arxiv.org/abs/2608.05688)).
* **CSS fixed-matching / fold-transversal structures.** Fold-transversal gates
  are due to Breuckmann and Burton
  ([arXiv:2202.06647](https://arxiv.org/abs/2202.06647), Quantum **8**, 1372
  (2024)), developed for bivariate bicycle codes by Eberhardt and Steffan
  ([arXiv:2407.03973](https://arxiv.org/abs/2407.03973)); the fixed-matching
  group decomposition ($S_M^Z$, $S_M^X$, Levi factor) is Albert's.
* **Diagonal Clifford-hierarchy searches.** Transversal diagonal logical
  operators of stabilizer codes were characterised algorithmically by Webster,
  Quintavalle and Bartlett
  ([arXiv:2303.15615](https://arxiv.org/abs/2303.15615), New J. Phys. **25**,
  103018 (2023)). A general ansatz-based search framework for diagonal logical
  gates of CSS codes and circuits is Bauer's
  ([arXiv:2607.26477](https://arxiv.org/abs/2607.26477)).
* **Automorphism-derived logical Clifford gates.** Sayginel, Koutsioumpas,
  Webster, Rajput and Browne
  ([arXiv:2409.18175](https://arxiv.org/abs/2409.18175), PRX Quantum **6**
  (2025); package `autqec`) give the rigorous formulation of stabilizer-code
  automorphisms, the generalisation of ZX-dualities to non-CSS codes, and the
  mapping of automorphism generators to physical circuits with Pauli
  corrections.
* **Monomial / GF(4) automorphisms.** The correspondence between stabilizer
  codes and additive GF(4) codes, and the reading of permutation-times-local-
  Clifford symmetries as GF(4)-code automorphisms, is Calderbank, Rains, Shor
  and Sloane (IEEE Trans. Inf. Theory **44**, 1369 (1998)); the three-column
  binary encoding used here is theirs.
* **The classification of single-qubit transversal gates.** The structure
  theorem behind the axis-frame sweep — every single-qubit transversal gate is
  local-Clifford-equivalent to a frame-diagonal one — is Zeng, Cross and Chuang
  ([arXiv:0706.1382](https://arxiv.org/abs/0706.1382)).
* **Eastin–Knill and its descendants.** That no code admits a universal
  transversal gate set is Eastin and Knill (PRL **102**, 110502 (2009)). That
  no stabilizer code admits fully transversal implementation of the logical
  Clifford group on more than one logical qubit, and that fold-transversal
  gadgets cannot realise the full Clifford group beyond two logical qubits, is
  Chakraborty and Gottesman
  ([arXiv:2602.13395](https://arxiv.org/abs/2602.13395), PRX Quantum, accepted
  2026). The package's one-block engine is written around the loophole those
  results leave open — *varying* partitions — and reports one-sided verdicts
  accordingly.
* **Physical realisation of a requested logical Clifford.** Rengaswamy,
  Calderbank, Kadhe and Pfister
  ([arXiv:1803.06987](https://arxiv.org/abs/1803.06987),
  [arXiv:1907.00310](https://arxiv.org/abs/1907.00310)) synthesise *all*
  physical Clifford circuits realising a given logical Clifford of a stabilizer
  code, by solving a partial binary symplectic system with transvections. That
  is the converse direction to this package's default question, and is the
  closest prior art to `logical/synthesis.py`.
* **The computational algebra.** Jacobson-radical computation in characteristic
  $p$ (Cohen–Ivanyos–Wales, J. Algebra **194** (1997); Friedl–Rónyai),
  Wedderburn decomposition, Berlekamp factoring, and Schreier–Sims are
  textbook or classical. No claim is made on any of them.
* **Row-space automorphism computation.** The invariant-set (bounded-weight
  codeword) reduction is Leon's 1982 algorithm and its descendants — Feulner's
  `codecan` in SageMath, MAGMA's `AutomorphismGroup`, GUAVA's `desauto`. The
  disjoint-information-set completeness bound is Brouwer–Zimmermann style.
* **Linearising a local-Clifford symmetry problem.** Dropping invertibility to
  turn a local-Clifford stabilizer condition into a linear system is the
  technique of Van den Nest, Dehaene and De Moor (PRA **70**, 034302 (2004))
  for graph states.

### Nearest prior art to the unifying framework

The single closest paper is:

* **Dasu and Burton, "A Classification of Transversal Clifford Gates for Qubit
  Stabilizer Codes"** ([arXiv:2507.10519](https://arxiv.org/abs/2507.10519)).
  They develop the theory of classifying stabilizer codes via **matrix algebras
  of endomorphisms**, a viewpoint introduced by Rains, and use it to completely
  classify the *diagonal* transversal Clifford symmetries of $\ell$ code
  blocks into six families of matrix groups.

The relationship should be stated carefully. Dasu–Burton and this package share
the central idea that transversal Clifford symmetry is governed by a finite
matrix algebra attached to the code, and the reader should treat the
endomorphism-algebra framework as *prior*, not as something this package
introduced. The differences are:

| | Dasu–Burton (2507.10519) | this package |
|---|---|---|
| goal | classification theorem: which groups can occur | computation: which group *does* occur, for a given code |
| gate class | diagonal transversal Cliffords on $\ell$ blocks | arbitrary Cliffords on a prescribed partition, of one block or of $\ell$ blocks jointly (plus specialised backends) |
| locality | transversal (one qubit per block) | arbitrary prescribed cells, singletons through the whole block |
| output | six families | explicit generators, exact order, logical image, and a certificate |

Also related in spirit:

* **Holmes, "Quantum Logic Codes"**
  ([arXiv:2606.13521](https://arxiv.org/abs/2606.13521)) studies exactly the
  regime the prescribed-partition framework is built for: lower bounds on the
  circuit depth needed for a complete logical Clifford algebra, and a code
  family with a constant-depth **2-local** transversal logical Clifford basis.
  Where Holmes constructs codes with a desired low-locality gate set, this
  package *decides*, for a code you already have, what a given locality buys
  you. The two are complementary; the package's 2-cell partitions compute the
  same kind of object Holmes constructs.
* **Tansuwannont, Chan and Takagi**
  ([arXiv:2602.09788](https://arxiv.org/abs/2602.09788)) construct a
  fold-transversal generating set of the full logical Clifford group for
  high-rate quantum Reed–Muller codes. That result is the reason the
  one-block engine in this package exists and is the benchmark its
  `rm64`/`tesseract` rows are checked against.

### Possibly distinctive here — stated as a hypothesis, not a claim

Two things, both of which need a fuller literature review before anyone should
call them new in print.

1. **Prescribed-partition code-preserving Clifford transformations of an
   arbitrary stabilizer code, computed as the symplectic unit subgroup of a
   finite preservation algebra, with exact order and explicit completeness
   certificates.** The ingredients are all prior art: the endomorphism-algebra
   viewpoint (Rains; Dasu–Burton), the linearisation trick (Van den Nest et
   al.), and the unit-group machinery (Cohen–Ivanyos–Wales; Wedderburn). What
   the package supplies is the combination — one parameterised object
   $A_{\mathcal{P}}(S)^{\times} \cap \prod_C \mathrm{Sp}(2|C|,2)$ that
   specialises to the strict transversal group at singleton cells (where
   $\mathrm{GL}(2,2) = \mathrm{Sp}(2,2)$ makes the symplectic condition free),
   to fixed-matching two-local groups at pairs, and to the whole
   code-preserving Clifford group at one cell — solved exactly for arbitrary
   stabilizer codes and arbitrary partitions.

   **This is not asserted to be the first such framework.** It is asserted to
   be the framework this package implements.

2. **Proof-carrying computation across backends.** Every headline result is
   emitted as *answer + mathematical witness + independent verifier*: the
   strict and general-stabilizer witnesses re-checked by the standalone
   `tools/check_witness.py` and `tools/check_stabilizer_witness.py` (numpy
   only, no import from this package), the Smith-form completeness certificates
   for $\mathbb{Z}_{2^L}$ kernels, the constructive Wedderburn split, the
   Schreier–Sims-certified block generation, and the sign-exact Stim
   verification of symplectic outputs. Individually each of these is standard
   practice somewhere; carrying the discipline across a whole multi-backend
   gate-search package, with a three-valued completeness field that never
   collapses a failure into a negative result, is the part worth pointing at.

---

## 2. Closest software

* [**qLDPCOrg/qLDPC**](https://github.com/qLDPCOrg/qLDPC) — the broadest
  construction-and-analysis library in the space: code families (BB, HGP,
  lifted product, quantum Tanner), a GAP-backed abstract-algebra module, and
  `get_transversal_ops`, which finds SWAP-transversal logical Cliffords via the
  code-automorphism method of arXiv:2409.18175. Complementary: its transversal
  route searches and hard-requires GAP+GUAVA or MAGMA; it does not decide
  completeness or certify nonexistence. Its SWAP-transversal class is a
  subclass of this package's monomial class, which makes it a natural
  cross-check target.
* [**hsayginel/autqec**](https://github.com/hsayginel/autqec) — an installable
  Python package finding logical Clifford gates through code automorphisms, the
  reference implementation of arXiv:2409.18175. Complementary: its primary
  search representation is a related binary-code automorphism group, optionally
  using MAGMA or BLISS. Its scope (permutation-based gates, including non-CSS
  ZX-dualities, with Pauli corrections and destabilizer bookkeeping) overlaps
  this package's `ansatz/permutation.py`, `ansatz/codeword_permutation.py` and
  `ansatz/monomial.py`, and is *disjoint* from the preservation-algebra
  framework, which does not move qubits.
* [**valbert4/two-fold-transversal**](https://github.com/valbert4/two-fold-transversal)
  — the official code-and-certificate repository for arXiv:2608.05688:
  certified survey data, fixed-matching search scripts, and special-case
  exhaustive enumerations. Its scripts are research-reproduction artifacts tied
  to the repository datasets; it does not expose an installable API or CLI
  accepting arbitrary `H_X, H_Z`. This package does not vendor any of its data.
* [**QuantumClifford.jl**, issue #434](https://github.com/QuantumSavory/QuantumClifford.jl/issues/434)
  — tracks a proposed automorphism-based implementation.
* **Bauer's implementation** accompanying
  [arXiv:2607.26477](https://arxiv.org/abs/2607.26477) is the closest thing to a
  general *ansatz-parameterised* gate finder, on the diagonal side and including
  spacetime (circuit) gates. This package's hierarchy solvers occupy the same
  problem space; its preservation-algebra solvers occupy the Clifford side that
  Bauer's diagonal framework does not target.

---

## 3. Terminology

Different papers use "transversal" for different things. This package fixes the
following usage and states the ansatz explicitly in every result.

| term used here | meaning |
|---|---|
| **strict transversal** / site-dependent transversal | one arbitrary single-qubit Clifford per qubit, independently chosen, no permutation. A depth-one layer of 1-local gates. |
| **prescribed-partition** ($\mathcal{P}$-local) | one arbitrary Clifford per cell of a fixed qubit partition. Depth one; locality is $\max_C \|C\|$. |
| **fixed-matching / fold-transversal** | the prescribed-partition case where cells are the pairs of an involution $\tau$, usually with $\tau$ a ZX-duality so a fold Hadamard exists. |
| **two-fold transversal** | the group generated by fixed-matching layers over *varying* matchings (Albert's $N_{\text{2fold}}$). Not depth one. |
| **monomial** | qubit permutation composed with per-qubit Cliffords. Not a prescribed-partition ansatz; it moves qubits. |
| **transversal across blocks** | corresponding-qubit gates on $\ell$ code blocks. Implemented as the prescribed-partition problem for the joint code $S^{\otimes \ell}$ (`transversal_clifford_across_blocks`); the *single-block* ansätze are the ones that act on one block. |

Two further distinctions the documentation never blurs:

* *complete for the specified ansatz* ≠ *complete over all physically possible
  fault-tolerant logical gates*. The package only ever claims the former.
* *symplectic action modulo Paulis and global phase* ≠ *a verified circuit*.
  Group orders are the former; `certificates/signed.py` produces the latter on
  request.

---

## 4. What this package does not do

Stated so a reader does not have to discover it by experiment:

* Gates across $\ell$ blocks are supported for the *transversal* partition
  (one gate per corresponding-qubit tuple).  Arbitrary interleavings of
  qubits from different blocks are expressible as a partition but have not
  been explored, and nothing here addresses codes of different lengths
  interacting.
* No non-Clifford, non-diagonal transversal gates (arbitrary single-qubit
  unitaries as an algebraic variety).
* No spacetime / circuit-level logical gates (cf. Bauer, arXiv:2607.26477).
* No exhaustive matching-orbit enumeration: `two_fold_group` and the one-block
  engine *sample* involutions, so they give positive fullness certificates and
  lower bounds, never proofs of emptiness.
* No decoder, no noise model, no threshold estimates.
* No external computer-algebra backend (GAP, MAGMA, MeatAxe). Logical group
  orders beyond the Schreier–Sims and recognition ranges are reported as lower
  bounds rather than delegated.
