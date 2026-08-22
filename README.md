# qec-transversal

**Exact and certified analysis of depth-one code-preserving gates for stabilizer quantum codes.**

An exact computational toolkit that, given a stabilizer code and a *prescribed
physical gate ansatz*, finds every code-preserving physical gate of that shape,
computes the logical Clifford or Clifford-hierarchy operation each one realises,
and reports whether the answer is complete — with a machine-checkable
certificate.

---

## Why this package exists

Ask a concrete question about a concrete code:

> *If I am allowed one arbitrary Clifford gate on each qubit — or on each pair
> of qubits from a fixed pairing — which of those depth-one layers preserve my
> code, what logical operations do they implement, and have I found all of
> them?*

Answering it well means answering three things at once:

1. **which physical transformations preserve the code** — a constraint problem;
2. **what logical operation each one realises** — a quotient computation in
   `Sp(2k, 2)`;
3. **whether the returned set is complete** — which is where most tools stop
   short, and where a truncated search silently becomes a false negative.

`qec-transversal` treats the third as a first-class output. Every result
carries a three-valued completeness field, and a computation that ran out of
budget reports `UNKNOWN` rather than "no gate exists". Every headline result
can be exported as a witness that a standalone checker — numpy only, importing
nothing from this package — re-verifies from scratch.

---

## Core capabilities

| Capability | API | Code class | Physical ansatz | Completeness |
|---|---|---|---|---|
| Strict site-dependent transversal Clifford | `strict_transversal_clifford` | any stabilizer code | one arbitrary single-qubit Clifford per qubit | **exact** (unit group of the preservation algebra); `UNKNOWN` if the algebra cannot be certified |
| Prescribed-partition Clifford | `partition_clifford_group` | any stabilizer code | one arbitrary Clifford per partition cell | **exact** when the unit group certifies or the algebra fits the enumeration cap; else `UNKNOWN` |
| Transversal across code blocks | `transversal_clifford_across_blocks` | any stabilizer code | one gate per corresponding-qubit tuple across `ℓ` blocks (transversal CNOT and friends) | **exact** — same solver, on the joint code `S^⊗ℓ` |
| CSS strict specialisation | `css_strict_transversal_clifford` | CSS | as above | **exact**, two GF(2) kernels, no enumeration |
| Fixed-matching (fold) layers | `matching_clifford_group` | CSS + an involution | diagonal `S`/`CZ` and `√X`/`XX` layer on the matching, plus fold Hadamard | **exact** for that ansatz; the Levi/CNOT factor is excluded (see `ansatz/twofold.py`) |
| Diagonal Clifford hierarchy, CSS | `diagonal_transversal_gates` | CSS | depth-one diagonal layer of level-`L` phase gates | **exact** kernel over `Z_{2^L}` at every level, with a Smith-form completeness certificate |
| Diagonal Clifford hierarchy, general | `diagonal_transversal_gates` | any stabilizer code | as above | **sound** always; complete when the exact support-coset enumeration fits or the code has no Z-type stabilizers — reported per call |
| Axis-frame sweep | `hierarchy.frames` | any stabilizer code | any single-qubit transversal gate | complete iff the `3^n` sweep is exhaustive **and** every frame reports complete |
| Permutation (SWAP-class) gates | `permutation_automorphism_group` | CSS | qubit permutations | **exact** when the characteristic weight classes are complete and span; else certified **lower bound** |
| Monomial gates | `monomial_clifford_group` | any stabilizer code | permutation ∘ per-qubit Clifford | **exact** when the whole stabilizer group is enumerable; else certified **lower bound** |
| One-block generated group | `one_block_clifford_group` | CSS | products of *all* depth-one one-block layers | one-sided: reaching `\|Sp(2k,2)\|` is a certificate; short of it is a **lower bound**, never a no-go |
| Two-fold transversal group | `ansatz.twofold.two_fold_group` | CSS | matching layers over sampled matchings, Levi factor included | fullness certificate or **lower bound** |
| Constructive target synthesis | `logical.synthesis.verify_logical_gate` | CSS | strict three-layer normal form | exact YES with an explicit witness, or a NO that is **complete for the strict class** |
| Sign-exact circuit verification | `certify_signs` | any stabilizer code | any result from any backend above | exact tableau check with an explicit Pauli correction, per generator |

---

## The unifying framework

Fix a stabilizer code `S ⊆ F_2^{2n}` and a partition `P` of its qubits. The
depth-one `P`-local ansatz asks for symplectic matrices of the form
`M = ⊕_{C∈P} M_C`. Solving `S M ⊆ S` directly is hard because
`∏_C Sp(2|C|,2)` is a group, not a linear space.

The move is to **drop invertibility**:

```math
A_{\mathcal P}(S) \;=\; \Big\{\, M=\bigoplus_{C\in\mathcal P} M_C \;:\; SM\subseteq S \,\Big\}
```

`A_P(S)` is a linear subspace — one GF(2) kernel — and it is closed under
multiplication (`SM ⊆ S` and `SM' ⊆ S` give `S(MM') ⊆ S`), so it is a
finite-dimensional unital `F_2`-algebra: the **preservation algebra**. The
physical code-preserving Clifford gates of the ansatz are then exactly its
symplectic units,

```math
A_{\mathcal P}(S)^{\times} \;\cap\; \prod_{C\in\mathcal P} \mathrm{Sp}(2|C|,2).
```

For singleton cells the intersection is free, because over `F_2`

```math
\mathrm{GL}(2,2)=\mathrm{SL}(2,2)=\mathrm{Sp}(2,2),
```

so **the strict site-dependent transversal Clifford group of any stabilizer
code is exactly the unit group of its preservation algebra.** Pairs give
fixed-matching two-local groups; one cell containing every qubit gives the whole
code-preserving Clifford group.

The unit group is computed without `2^{dim A}` enumeration, through the
classical structure route

```math
A \;\to\; J(A) \;\to\; A/J(A) \;\cong\; \prod_i M_{d_i}(\mathbb F_{q_i}) \;\to\; A^{\times},
```

with every stage verified rather than trusted: radical candidates are proven
nilpotent by explicit power computation, semisimplicity is proven
constructively, per-block generation is certified against `|GL(d,q)|` by a
Schreier–Sims chain, and any failure returns `UNKNOWN`.

None of the individual ingredients is new — the endomorphism-algebra view of
stabilizer codes goes back to Rains and was developed for transversal Clifford
classification by Dasu and Burton, the linearisation trick is Van den Nest–
Dehaene–De Moor, and the algebra algorithms are Cohen–Ivanyos–Wales and
Wedderburn. What this package contributes is the combination, computed exactly
for arbitrary codes and arbitrary partitions, with certificates.
[`docs/related_work.md`](docs/related_work.md) draws that boundary carefully.

The symplectic cut is **not** taken by sweeping `A^×` and filtering. Writing
`σ(M) = Ω Mᵀ Ω`, an element is blockwise symplectic exactly when `σ(M)M = 1`,
the fibers of `φ(u) = σ(u)u` are the left cosets of `G`, and `im φ` is the
orbit of `1` under the congruence action `a · u = σ(u) a u`. So

```math
|G| \;=\; |A^{\times}| \,/\, |\text{orbit of } 1|,
```

orbit-stabilizer *is* the index formula, and Schreier's lemma turns the
transversal into certified generators. Cost is linear in `|A^×|/|G|` instead of
`|A^×|`, so a *large* gate group is now *cheaper*: on `[[6,2,2]]` a partition
with `|A^×| = 1.17 × 10^15` is decided by a 1.4-million-point orbit. That is
what makes cells wider than three qubits decidable at all
(`algebra/unitary_group.py`, `method="phi"`).

Full derivations: [`docs/mathematics.md`](docs/mathematics.md).

---

## A result the framework produced

Chakraborty and Gottesman ([arXiv:2602.13395](https://arxiv.org/abs/2602.13395),
Thm. 1) prove that realizing the full logical Clifford group on `k` logical
qubits needs a **fixed** partition of width at least `k`, and give a tight
example: `k` blocks of the `[[7,1,3]]` code. They note it does not extend to
single-block codes and leave that open.

Sweeping *every* partition of a given width — 15,906 of them across six codes,
each solved to `COMPLETE`, zero `UNKNOWN` — the bound is necessary but not
sufficient, and the gap is code-dependent:

| code | k | d | structure | best logical image at width k |
|---|---|---|---|---|
| `[[14,2,3]]` = Steane ⊗ Steane | 2 | 3 | **decomposable** | **720 / 720 — full** |
| `[[4,2,2]]` | 2 | 2 | indecomposable | 48 / 720 |
| `[[6,2,2]]` | 2 | 2 | indecomposable | 60 / 720 |
| `[[8,2,3]]` | 2 | 3 | indecomposable | 6 / 720 |
| `[[10,2,4]]` | 2 | 4 | indecomposable | 8 / 720 |
| `[[8,3,2]]` | 3 | 2 | indecomposable | 384 / 1,451,520 |
| `[[8,3,3]]` | 3 | 3 | indecomposable | 1,920 / 1,451,520 |

Carried up through successive widths the crossing point pins exactly:
`w_fixed([[4,2,2]]) = 3` and `w_fixed([[6,2,2]]) = 4`, against a bound of 2 in
both cases — so it is not a function of `k`.

Two readings, at the strength the data supports. Secure: **the bound is not
sufficient**. Conjecture, *not* a result: for `k ≥ 2` a width-`k` fixed
partition carries `Sp(2k,2)` only when the code factorises into `k` codes of one
logical qubit each. Six indecomposable codes and one decomposable control are
evidence, not a proof.

Fault tolerance is a separate question, and cell width does not answer it: the
criterion is the **partition distance** `d_P` of
[`faulttolerance.py`](src/qec_transversal/faulttolerance.py) — the least number
of cells a logical operator's support meets, with one unflagged faulty gate
correctable exactly when `d_P ≥ 3`. On the best partition of each row the split
is total: the decomposable control has `d_P = 3` and is single-fault
correctable; all six indecomposable rows have `d_P ≤ 2` and are not. The
partitions that come closest to the full group are not the ones that protect it.

The sharper fact is the contrast with *varying* partitions: in all six
exhaustive rows, layers over **all** partitions of the same width generate the
entire logical Clifford group. Letting the pairing change between layers — not
making the gates bigger — is what buys the group.

Data: [`docs/zoo/width_census.json`](docs/zoo/width_census.json). Rendered with
the derivation at [the project page](https://muuuun.github.io/qec-transversal/#width).

---

## Quick start

```bash
pip install -e '.[full]'
```

### A CSS code

```python
from qec_transversal import CSSCode, REGISTRY, strict_transversal_clifford

steane = CSSCode(*REGISTRY["steane"].build())
result = strict_transversal_clifford(steane)

print(result.method)                # css-shear-kernel
print(result.completeness)          # COMPLETE
print(result.logical_group_order)   # 6  == |Sp(2,2)|: transversal S and H
```

### A non-CSS stabilizer code

```python
from qec_transversal import five_qubit_code, strict_transversal_clifford

result = strict_transversal_clifford(five_qubit_code())

print(result.method)                # preservation-algebra units (singleton partition)
print(result.completeness)          # COMPLETE
print(result.group_order)           # 3  — the cyclic (SH)^{otimes 5} gate
print(result.logical_group_order)   # 3
```

### A prescribed two-qubit partition

```python
from qec_transversal import CSSCode, REGISTRY, partition_clifford_group

c422 = CSSCode(*REGISTRY["c4-22"].build())          # the [[4,2,2]] code

singletons = partition_clifford_group(c422, [(0,), (1,), (2,), (3,)])
pairs      = partition_clifford_group(c422, [(0, 1), (2, 3)])

print(singletons.group_order, singletons.logical_group_order)   # 6 6
print(pairs.group_order,      pairs.logical_group_order)        # 384 48
```

Allowing two-local gates on a fixed pairing lifts the logical group of the
`[[4,2,2]]` code from order 6 to order 48 — and both numbers are `COMPLETE`,
so the gap is a fact about the code, not about the search.

### Gates across two code blocks

```python
from qec_transversal import (
    CSSCode, REGISTRY, five_qubit_code, transversal_clifford_across_blocks,
)

steane = CSSCode(*REGISTRY["steane"].build())

two_steane = transversal_clifford_across_blocks(steane, blocks=2)
two_perfect = transversal_clifford_across_blocks(five_qubit_code(), blocks=2)

print(two_steane.completeness, two_steane.logical_group_order)    # COMPLETE 720
print(two_perfect.completeness, two_perfect.logical_group_order)  # COMPLETE 18
```

Both codes encode two logical qubits across two blocks, and `|Sp(4,2)| = 720`.
Two Steane blocks realise the **entire** logical Clifford group from depth-one
two-block gates; two `[[5,1,3]]` blocks reach 18 of it. The second number is a
*certified negative* — not a search that gave up.

### A diagonal Clifford-hierarchy gate

```python
from qec_transversal import CSSCode, REGISTRY, diagonal_transversal_gates

qrm15 = CSSCode(*REGISTRY["qrm15"].build())         # [[15,1,3]] quantum Reed-Muller
result = diagonal_transversal_gates(qrm15, level=3)

print(result.completeness)                # COMPLETE
print(result.metadata["max_level"])       # 3  — a genuine logical T
print(result.metadata["has_level_gate"])  # True
```

### Certificate verification

Export a witness and re-check it with the standalone verifier, which imports
nothing from this package:

```python
from qec_transversal import CSSCode, REGISTRY
from qec_transversal.certificates.witness import export_strict_witness, write_witness

steane = CSSCode(*REGISTRY["steane"].build())
write_witness(export_strict_witness(steane, "steane"), "steane-witness.json")
```

```bash
python tools/check_witness.py steane-witness.json
```

The checker re-derives every constraint row from its recorded provenance, runs
its own elimination to confirm `rank(constraints) + dim(kernel) = n` — that is
what makes the kernel *complete* rather than merely valid — recomputes each
generator's logical action, and closes the logical group. It is mutation-tested
against eight classes of forged strict witness and seven classes of forged
general-stabilizer witness (`tests/test_witness.py`).

### Command line

```bash
qec-transversal list-codes
qec-transversal strict --code steane
qec-transversal partition --code c4-22 --cells "0,1;2,3"
qec-transversal diagonal --code qrm15 --level 3
qec-transversal analyze examples/steane.json          # the detailed CSS report
qec-transversal verify --code steane H 0              # constructive synthesis
```

---

## What "complete" means

This is the most important section of this README.

Every result reports one of three values, and the word *complete* is always
scoped to **the stated ansatz**:

| value | meaning |
|---|---|
| `COMPLETE` | the returned set is provably the entire solution set of that ansatz |
| `INCOMPLETE_LOWER_BOUND` | everything returned is certified, but the search was capped, sampled, or scoped to a subgroup — the truth can only be larger |
| `UNKNOWN` | a verification or a budget failed; **nothing** may be concluded, in either direction |

**`COMPLETE` is possible because the ansatz is solved, not searched.** For the
partition framework the solution set is exactly
`A_P(S)^× ∩ ∏_C Sp(2|C|,2)`, and `A_P(S)` is an exact kernel; computing its
whole unit group therefore settles the question with nothing left to miss. For
the CSS shear families and the `Z_{2^L}` diagonal kernels the same is true by
construction.

**What `COMPLETE` does *not* mean.** These are different statements and the
package never conflates them:

* *complete for the specified ansatz* — what is reported;
* *complete over all depth-one physical gates* — only true when the ansatz is
  the whole partition into one cell;
* *complete over all fault-tolerant logical gates of the code* — never claimed
  by anything here. Gate teleportation, code surgery and non-constant-depth
  circuits are all outside scope.

**Where completeness is genuinely one-sided.** `one_block_clifford_group` and
`two_fold_group` *sample* involutions rather than enumerating them. Reaching
`|Sp(2k,2)|` is a positive certificate of fullness; falling short is a lower
bound on what the code admits and is **never** evidence that the code cannot do
better. There is no `False` for `is_full`, only `True` or undecided.

**Modulo Pauli.** Reported group orders are symplectic actions modulo Pauli
operators and global phases. That is sound — the sign defect of a Clifford on
the stabilizer group is a linear character, so a Pauli correction always exists
and never changes the symplectic action — but it is not a circuit-level claim.
Close the gap for any result with one call:

```python
from qec_transversal import certify_signs

certificate = certify_signs(code, result)
print(certificate.checked, certificate.certified)
```

Every generator of every gate ansatz carries its dense physical matrix, so this
works uniformly. The exception is `one_block_clifford_group`, which collects
*logical* actions rather than physical layers: its records are skipped and the
certificate says so — and a certificate that checked nothing is never
`certified`.

**Terminology.** "Transversal" means different things in different papers.
[`docs/related_work.md`](docs/related_work.md) §3 fixes the usage here, and
every result carries an `ansatz` string spelling out its own gate class in
words.

---

## Installation

Python 3.9 or newer.

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

The only required runtime dependency is NumPy. Optional extras:

| extra | pulls in | needed for |
|---|---|---|
| `automorphism` | `python-igraph` (BLISS) | all permutation-based ansätze: Tanner-graph, characteristic-codeword, monomial |
| `signs` | `stim` | sign-exact circuit verification, and the general-stabilizer diagonal solver |
| `full` | both | everything |
| `dev` | both, plus `pytest`, `ruff`, `sympy` | development and the full test suite |

A minimal runtime install (`pip install .`) gives you the preservation-algebra
solvers, the CSS specialisations, the CSS diagonal hierarchy, group orders,
synthesis, and witness export.

---

## Tests

```bash
pytest
```

No `PYTHONPATH=src` incantation: `pyproject.toml` sets `pythonpath = ["src"]`,
so the suite runs from a bare checkout as well as from an editable install.

```bash
pytest -m "not slow"      # skip the large instances
pytest -q tests/test_cross_validation.py
```

The suite is not only smoke tests. Its load-bearing parts are:

* **specialised vs general agreement** — the CSS shear solver and the general
  preservation-algebra solver must produce the same group on every CSS code
  where both apply, after representation conversion;
* **brute force for small `n`** — the exact algebraic answer is compared
  against exhaustive enumeration of all `6^n` single-qubit Clifford assignments
  (and all cell-block assignments for pair partitions);
* **logical-action consistency** — for every returned physical generator:
  stabilizer preservation is re-verified, the logical action is independently
  re-derived by quotient projection, and the result is checked to be symplectic;
* **unit-group verification** — for small finite algebras, the computed
  generators are closed and compared against brute-force enumeration of all
  invertible elements;
* **certificate mutation tests** — forged witnesses and mutated Smith
  certificates must *fail* verification;
* **honest-failure tests** — capped and budget-exceeded computations must
  report `unknown`, never a small number dressed up as an answer.

---

## Documentation

* [`docs/mathematics.md`](docs/mathematics.md) — the framework: preservation
  algebras, the finite-algebra solver, logical actions, completeness, and each
  specialised backend.
* [`docs/related_work.md`](docs/related_work.md) — positioning against the
  literature, an explicit novelty boundary, terminology, and what this package
  does not do.
* [`docs/refactor_report.md`](docs/refactor_report.md) — the 0.1 → 0.2 module
  mapping, API changes, corrected claims, and remaining limitations.
* [`docs/qldpc-survey.md`](docs/qldpc-survey.md) — a dated census snapshot.
* The **Transversal Gate Zoo**, a certified census of the built-in registry:
  <https://muuuun.github.io/qec-transversal/> (generated by `docs/zoo/`).

Package layout, by concept rather than by development history:

```text
src/qec_transversal/
    api.py            the public entry points and the uniform result object
    codes/            code objects (CSS, general stabilizer) and the registry
    ansatz/           one module per physical gate class being searched
    algebra/          preservation algebras; certified radical / Wedderburn / unit group
    hierarchy/        diagonal gates in the Clifford hierarchy over Z_{2^L}
    logical/          logical action, group orders, recognition, synthesis, words
    certificates/     witnesses, verifiers, sign-exact circuit checks
    utils/            GF(2), symplectic, permutation, polynomial, modular primitives
```

---

## Citation

A paper is in preparation. Until then, please cite the software and the
underlying references you rely on:

```bibtex
@software{qiao_qec_transversal,
  title  = {qec-transversal: exact and certified analysis of depth-one
            code-preserving gates for stabilizer codes},
  author = {Qiao, Mu},
  year   = {2026},
  url    = {https://github.com/Muuuun/qec-transversal},
  note   = {Version 0.2.0. DOI to be assigned.}
}
```

The CSS strict-transversal parameter-code construction and the fixed-matching
structure implemented here are due to:

```bibtex
@misc{albert2026beyond,
  title         = {Beyond transversality: structure of {C}lifford circuits for {CSS} codes},
  author        = {Albert, Victor V.},
  year          = {2026},
  eprint        = {2608.05688},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph},
  url           = {https://arxiv.org/abs/2608.05688}
}
```

Further attributions — Dasu–Burton, Webster–Quintavalle–Bartlett, Sayginel et
al., Breuckmann–Burton, Rengaswamy et al., Zeng–Cross–Chuang,
Chakraborty–Gottesman, Holmes, Bauer — are given in
[`docs/related_work.md`](docs/related_work.md).

---

## License

MIT. No code or data is vendored from any referenced repository.
