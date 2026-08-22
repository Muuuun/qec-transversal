# Refactor report: 0.1 → 0.2 → 0.2.1

## 0.2.1 — three follow-ups aimed at the scaling and scope gaps

Recorded first because they change results, not just structure.

**The involution, and a strictly smaller algebra (`algebra/preservation.py`).**
The gate group is the *unitary group* of an algebra with involution,
`G = {M ∈ A^× : σ(M)M = 1}` with `σ(M) = ΩMᵀΩ` — but only if `σ` maps the
algebra to itself, and for `A(S)` it does not. The image is identified exactly:
`σ(A(S)) = A(N)` with `N` the normalizer, so the σ-stable object is
`A'(S) = A(S) ∩ A(N)`, and it provably yields the *same* gate group. It is
strictly smaller on multi-qubit cells: pair partitions of `[[4,2,2]]` 20 → 16,
`[[6,2,2]]` 28 → 20, `[[8,3,2]]` 27 → 22, `iceberg-8` 36 → 24. Three of those
crossed the enumeration cap, so results that were honestly `UNKNOWN` in 0.2 are
now exact — `[[6,2,2]]` pairs 3072, `[[8,3,2]]` pairs 6144, `iceberg-8` pairs
24576. This is the default; `partition_algebra(..., refine=False)` recovers the
old algebra. Proof and measurements in `docs/mathematics.md` §3.2; pinned by
`tests/test_involution.py` (25 tests, including that `A(S)` really is *not*
σ-stable, so nobody deletes the refinement as redundant).

**Gates across code blocks (`transversal_clifford_across_blocks`).** The
classical notion of transversal — one gate per corresponding-qubit tuple across
`ℓ` blocks — needed no new mathematics: it is the prescribed-partition problem
for the joint code `S^⊗ℓ` with cells `{i, n+i, …, (ℓ-1)n+i}`. New in
`codes/stabilizer.py`: `tensor_product`, `tensor_power`,
`corresponding_qubit_cells`. Two certified results of opposite sign, both
`COMPLETE`: two Steane blocks realise the **entire** `Sp(4,2)` logical Clifford
group (order 720) from depth-one two-block gates, while two `[[5,1,3]]` blocks
reach 18 of that same 720 — a certified negative. `tests/test_cross_block.py`.

**Sign-exactness for every backend (`certify_signs`).** In 0.2 only the strict
*diagonal* generators went through the sign-exact pipeline, because only they
carried a physical matrix. Every generator record of every ansatz now carries
its dense `2n × 2n` physical symplectic matrix, and `certify_signs(code,
result)` lifts each to an exact Stim tableau, solves the Pauli correction, and
re-verifies all stabilizer signs `+1`. `one_block_clifford_group` is the
deliberate exception — its records hold `2k × 2k` *logical* actions, so they are
skipped rather than misread, and shape is the discriminator so a logical action
can never be silently verified as if it were a gate. `tests/test_sign_exact.py`.

**Closed, 2026-08-22.** The symplectic cut used to be applied by enumerating
`A'^×` and filtering — 393,216 elements to find a group of 6,144 on `[[8,3,2]]`
pairs, and nothing at all once `|A^×|` passed the cap. The involution supplies
the structure to do better: the fibers of `φ(M) = σ(M)M` are exactly the left
cosets `GM`, so `|G| = |A^×| / |im φ|`, and `im φ` is the orbit of `1` under the
congruence action `a . u = σ(u) a u`. `algebra/unitary_group.py` computes that
orbit (bulk GF(2) products, BFS tree for the transversal) and returns both the
exact order and, by Schreier's lemma, a certified generating set;
`partition_units_via_structure(method="phi")` and `method="auto"` above the
sweep cap route through it. `[[6,2,2]]` with a 3-cell partition: 14,155,776
units cut by a 384-point orbit in 0.4 s. With a 5-cell: 1.17e15 units cut by a
1.4e6-point orbit. Both were `UNKNOWN`.

**Still open.** The orbit is `|A^×| / |G|`, so a *small* gate group inside a
huge unit group still overflows — the honest exit is `UNKNOWN`. Removing that
needs `|G|` in closed form: a σ-adapted Wedderburn decomposition plus Wall's
classification of the unitary groups of the simple factors.

---

# Refactor report: 0.1 → 0.2

The package grew feature-by-feature and its module layout recorded that
history: a flat directory of twenty-two modules whose names described the order
in which capabilities were added, a README that opened as a "CSS transversal
Clifford" tool, and scientific claims scattered across files at different
vintages. This release reorganises the package around one idea — the
preservation algebra of a stabilizer code on a prescribed partition — and makes
scope and completeness first-class, machine-readable outputs.

**Nothing about the numerical behaviour changed.** Every algorithm body was
moved by line slice rather than retyped, and the full 0.1 test suite (224
tests) passes unchanged apart from import paths.

---

## 1. Major structural changes

### Old → new module mapping

| 0.1 module | 0.2 location | what moved |
|---|---|---|
| `gf2.py` | `utils/gf2.py` | GF(2) elimination, nullspace, quotient complement, bit-packed kernel |
| | `utils/symplectic.py` | `symplectic_form`, `symplectic_product`, `is_symplectic` |
| `group.py` | `logical/group.py` | `GroupOrder`, `generated_group_order`, `schreier_sims_order` |
| | `utils/symplectic.py` | `symplectic_group_order` |
| `css.py` | `codes/css.py` | the `CSSCode` object |
| | `ansatz/strict_css.py` | shear families, `ParameterSpace`, `TransversalAnalysis` |
| `stabilizer.py` | `codes/stabilizer.py` | the `StabilizerCode` object, `five_qubit_code` |
| | `algebra/preservation.py` | `local_clifford_algebra`, `partition_algebra`, block helpers |
| | `ansatz/strict.py` | `LocalCliffordAnalysis`, `analyze_local_clifford` |
| | `ansatz/partition.py` | `PartitionCliffordAnalysis`, `partition_units_via_structure` |
| | `utils/symplectic.py` | `symplectic_gram_schmidt` |
| `unitgroup.py` | `algebra/finite_algebra.py` | `AlgebraF2` |
| | `algebra/radical.py` | Cohen–Ivanyos–Wales radical, nilpotency proofs, peeling, quotients |
| | `algebra/wedderburn.py` | constructive semisimple split, block identification |
| | `algebra/unit_group.py` | `unit_group`, `UnitGroupResult` |
| | `algebra/orders.py` | `\|GL(d,q)\|` |
| | `utils/polynomials.py` | charpoly, minimal polynomial, Berlekamp, GF(2)[x] arithmetic |
| `matching.py` | `ansatz/matching.py` | fixed-matching kernels, fold Hadamard |
| | `logical/group.py` | `logical_group_summary` |
| `hierarchy.py` | `hierarchy/css.py` | the CSS coset ladder |
| | `utils/modular.py` | `module_kernel`, 2-adic valuation |
| | `certificates/hierarchy.py` | Smith-form certificate and its independent checker |
| `axes.py` | `hierarchy/general.py` | general-stabilizer diagonal kernels |
| | `hierarchy/frames.py` | the axis-frame sweep |
| `automorphisms.py` | `ansatz/permutation.py` | Tanner-graph automorphisms, `describe_permutation` |
| | `utils/permutations.py` | `permutation_group_order` |
| | `utils/graph.py` | the `python-igraph` import guard |
| `codewordaut.py` | `ansatz/codeword_permutation.py` | characteristic-codeword row-space automorphisms |
| `monomial.py` | `ansatz/monomial.py` | CRSS GF(4) monomial automorphisms |
| `twofold.py` | `ansatz/twofold.py` | `N_2fold` sampling, Levi units |
| `dualities.py` | `ansatz/dualities.py` | structural ZX-duality candidates |
| `discovery.py` | `ansatz/discovery.py` | blind structural matching/permutation discovery |
| `oneblock.py` | `logical/generated.py` | `OneBlockAnalysis`, `analyze_one_block`, `factor_target` |
| | `logical/recognition.py` | the McLaughlin fullness-recognition route |
| | `logical/words.py` | `WordBSGS`, the word-tracking chain |
| | `utils/symplectic.py` | `symplectic_transvection` |
| | `utils/permutations.py` | symmetric-group element orders |
| | `utils/polynomials.py` | integer-bitmask GF(2)[x] arithmetic, matrix polynomials |
| `synthesis.py` | `logical/synthesis.py` | `verify_logical_gate`, `logical_target` |
| `witness.py` | `certificates/witness.py` | witness export |
| `signed.py` | `certificates/signed.py` | `SignedStabilizer`, `verify_sign_exact` |
| `phase.py` | `certificates/phase.py` | sign-exact diagonal circuit verification |
| `codes.py` | `codes/families.py` | the code-family constructors |
| | `codes/registry.py` | `NamedCode`, `REGISTRY` |
| — | `api.py` | **new**: the public entry points and `GateSearchResult` |
| — | `logical/action.py` | **new**: the shared logical-action projection |

### Why these boundaries

The dependency graph, not taste, decided most of them:

* `stabilizer.py` was importing `logical_group_summary` from `matching.py` —
  a general solver depending on a CSS-specific frontend. Moving group orders
  into `logical/group.py` removes that inversion.
* `monomial.py` was importing the private `_require_igraph` from
  `automorphisms.py`. The guard is now `utils/graph.py`, imported by all three
  BLISS-backed engines.
* `css.py` mixed a *code object* with an *ansatz solver*; `stabilizer.py` mixed
  a code object, two ansatz solvers, and the preservation-algebra construction.
  Splitting those is what makes the general/specialised distinction visible.
* `unitgroup.py` was 964 lines covering four distinct algorithms with different
  literature provenance; each is now its own module with its own citation.

### Duplication removed

| duplicated concept | copies in 0.1 | now |
|---|---|---|
| logical-action projection + residue check | 7 (`css`, `stabilizer` ×2, `matching`, `monomial`, `automorphisms`, `twofold`) | `logical/action.py::project_to_logical` |
| GF(2) linear solve / coordinate extraction | 4 (`synthesis`, `phase`, `signed`, `unitgroup`) | `utils/gf2.py::solve_gf2`, `coordinates_over` |
| left-kernel basis with tracking | 1 private (`signed`) | `utils/gf2.py::left_kernel_basis` |
| random involution sampler | 2 (`twofold`, `oneblock`) | `utils/permutations.py::random_involution` |
| igraph import guard | 3 | `utils/graph.py::require_igraph` |
| `\|GL(d,q)\|` | 1, but needed by two modules | `algebra/orders.py` |

Deliberately **not** unified: the two GF(2)[x] polynomial representations
(integer bitmasks for the recognition route, coefficient arrays for the algebra
solver). Both are independently validated; merging them would mean rewriting
two factoring routines to save a hundred lines. They now sit in one module,
`utils/polynomials.py`, with the duplication labelled rather than hidden.

### Renames, and the ones not made

| old name | new name | reason |
|---|---|---|
| `unitgroup.py` | `algebra/unit_group.py` (+ 3 siblings) | exposes the mathematical pipeline, not one function |
| `axes.py` | `hierarchy/general.py`, `hierarchy/frames.py` | "axes" named an implementation detail; the module did two unrelated things |
| `oneblock.py` | `logical/generated.py` | "one block" is scope, "generated logical group" is the object |
| `automorphisms.py` | `ansatz/permutation.py` | "automorphisms" is ambiguous across three different engines here |
| `codewordaut.py` | `ansatz/codeword_permutation.py` | abbreviation; and it is a *permutation* engine |
| `stabilizer.py` | four modules (above) | it had accumulated four responsibilities |
| `matching.py` | `ansatz/matching.py` | kept — the name is exactly right for the ansatz |
| `codes.py` | `codes/families.py` + `registry.py` | data vs constructors |
| `twofold.py` | `ansatz/twofold.py` | kept — it names Albert's `N_2fold` |

The distribution name (`qec-transversal`) and the import package name
(`qec_transversal`) are **unchanged**. Renaming them would break every existing
install and citation for a cosmetic gain; the README title carries the broader
scope instead.

---

## 2. Public API changes

### New

```python
from qec_transversal import (
    Completeness, GateSearchResult,
    strict_transversal_clifford, css_strict_transversal_clifford,
    partition_clifford_group, matching_clifford_group,
    diagonal_transversal_gates, monomial_clifford_group,
    permutation_automorphism_group, one_block_clifford_group,
)
```

Every one returns a `GateSearchResult` with the same fields: `method`,
`ansatz`, `completeness`, `generators`, `logical_generators`, `group_order`,
`logical_group_order`, `logical_group_order_is_exact`, `certificate`,
`metadata`, plus the raw backend object as `analysis`. `to_dict()` gives a
JSON-safe summary; `complete` is a boolean shorthand.

`Completeness` is a three-valued string enum: `COMPLETE`,
`INCOMPLETE_LOWER_BOUND`, `UNKNOWN`. No entry point can return "no gate exists"
as a consequence of a budget or a cap.

Also new: `CSSCode.to_stabilizer_code()`, the bridge that makes the
specialised/general cross-validation tests possible.

`diagonal_transversal_gates` now accepts a general `StabilizerCode` as well as
a `CSSCode` and routes to the appropriate solver, reporting which one ran in
`method` and reporting `INCOMPLETE_LOWER_BOUND` when the general engine is
sound but not provably complete. In 0.1 the two solvers had no shared entry
point and the general one was reachable only as `axes.diagonal_kernel_general`.

### Compatibility aliases

Every 0.1 import path still works. Nineteen thin alias modules re-export the
moved names — `qec_transversal.gf2`, `.group`, `.css`, `.stabilizer`,
`.matching`, `.monomial`, `.automorphisms`, `.codewordaut`, `.axes`,
`.twofold`, `.dualities`, `.discovery`, `.oneblock`, `.synthesis`, `.witness`,
`.signed`, `.phase`, `.unitgroup`, and the `codes` / `hierarchy` packages,
whose `__init__` files carry the old flat surface.

`tests/test_compat_aliases.py` asserts object *identity* through every alias,
so a future edit cannot leave an alias pointing at a stale copy. New code
should import from the new locations; the aliases are for existing scripts and
notebooks.

Two aliases carry a caveat: `qec_transversal.codes` and
`qec_transversal.hierarchy` are now packages, not modules, so
`import qec_transversal.codes as codes; codes.__file__` points at
`codes/__init__.py`. Attribute access is unchanged.

### CLI

`analyze`, `generate`, `list-codes` and `verify` behave exactly as in 0.1
(`analyze` still emits the detailed CSS report). Added: `strict`, `partition`,
`diagonal`, `monomial`, `automorphisms`, `one-block`, all emitting the uniform
`GateSearchResult` envelope. Input JSON now also accepts `"H"` (symplectic
`[X | Z]` rows) for non-CSS codes.

---

## 3. Documentation changes

### New documents

* `docs/mathematics.md` — the framework end to end: stabilizer codes as
  symplectic subspaces, the ansatz as a partition constraint, the preservation
  algebra and why it is an algebra, the singleton case where
  `GL(2,2) = Sp(2,2)`, the certified finite-algebra solver, logical actions,
  the three completeness levels, and each specialised backend with its scope.
* `docs/related_work.md` — replaces `docs/landscape.md` (deleted). Explicit
  three-way split between established theory implemented here, nearest prior
  art, and what may be distinctive; a terminology table for the six different
  meanings of "transversal"; and a section on what the package does not do.

### Outdated scientific claims corrected

| claim in 0.1 | status | correction |
|---|---|---|
| README opened as "a Python tool for finding and certifying strict-transversal logical Clifford gates of CSS quantum codes" | outdated | the package covers arbitrary stabilizer codes and seven gate ansätze; the README now opens on the general framework |
| `docs/landscape.md`: "Version 0.1 computes the complete strict-transversal parameter spaces … Fixed-matching two-local search, matching-orbit generation, non-CSS SAT/SMT search, Pauli dressing, and scalable matrix-group recognition remain future modules" | false since ~0.1.x | all but the SAT/SMT search were implemented; the roadmap was never updated |
| `docs/landscape.md` "Gap addressed here" positioned the package solely against the CSS parameter-code construction | incomplete | `related_work.md` now positions against eight research directions and names the nearest prior art (Dasu–Burton, arXiv:2507.10519) explicitly |
| README listed a 7-step "Roadmap" whose first six steps were done | stale | removed; remaining gaps are stated as scope limits in `related_work.md` §4 |
| README: "the two-local group as `A^× ∩ ∏ Sp(2\|C\|,2)`" appeared as one bullet among thirty | understated | this *is* the package's framework; it is now the README's "Unifying framework" section and the organising principle of the module layout |
| no document distinguished "complete for the ansatz" from "complete over all fault-tolerant gates" | a real risk of misreading | README has a dedicated "What complete means" section; every result carries an `ansatz` string |
| `docs/qldpc-survey.md` headline 1 read as a claim about check sparsity | already flagged in a 2026-08-13 note | note retained and the document re-labelled a dated snapshot in the docs index |
| no citation existed for the endomorphism-algebra viewpoint | attribution gap | Rains and Dasu–Burton (arXiv:2507.10519) are now cited as prior art for the central object |

Novelty language was reviewed throughout. The package does **not** claim to be
the first framework of its kind; `related_work.md` §1 states the possible
contribution as a hypothesis requiring a fuller literature review, and lists the
prior art each ingredient comes from.

---

## 4. New tests

| file | invariant validated |
|---|---|
| `tests/test_cross_validation.py::test_css_and_general_strict_solvers_agree` | the CSS shear solver and the general preservation-algebra solver produce the same *physical* group order and, after transporting through the change of logical basis, the same *set* of logical matrices. The two share no code path. |
| `…::test_general_strict_solver_matches_brute_force_on_small_codes` | exact algebra vs exhaustive enumeration of all `6^n` single-qubit Clifford assignments, on CSS and non-CSS codes including `[[5,1,3]]` |
| `…::test_singleton_partition_reproduces_the_strict_group` | the `GL(2,2) = Sp(2,2)` specialisation: singleton cells give exactly the strict group, physically and logically |
| `…::test_structured_and_enumeration_partition_routes_*` | the certified unit-group route and the `2^dim` sweep agree where both finish, and a decline is `UNKNOWN` with no order attached |
| `…::test_two_local_partition_strictly_contains_the_strict_group` | coarsening a partition can only add gates; the orders satisfy Lagrange |
| `…::test_every_*_generator_survives_an_independent_recheck` | for every returned generator: stabilizer preservation re-verified, logical action re-derived longhand (not via the shared helper), result checked symplectic |
| `…::test_logical_generators_of_every_public_entry_point_are_symplectic` | no backend can emit a logical image outside `Sp(2k,2)` |
| `…::test_exhausted_budget_reports_unknown_not_a_small_group` | a starved computation yields `UNKNOWN` and `group_order is None` |
| `tests/test_api.py` | result-shape uniformity across all eight entry points, JSON-safety, enum values, dispatch rules, self-describing `ansatz` strings, one-sidedness of the one-block verdict, scope reporting for permutation and monomial engines |
| `tests/test_readme_examples.py` | every executable README example runs and produces the printed number; every CLI example exits 0 and emits valid JSON; every API symbol named in the README exists |
| `tests/test_compat_aliases.py` | 70 legacy import paths resolve to the *same objects* as the new ones, plus the `codes` / `hierarchy` package facades |

Suite size: **224 tests before, 352 after** (128 added, 2 skipped — the two
skips are partition instances where a solver declines honestly and the test
says so rather than asserting through it). `ruff check --select F` is clean
across `src`, `tests`, `scripts`, `tools` and `docs/zoo`; the remaining
findings under the ambient extended rule set (`E741` on `l`, `E731` lambdas,
`E402` after `pytest.importorskip`, `E702` in tests) are pre-existing and
unchanged in kind from 0.1.

Already present in 0.1 and retained unchanged: unit-group fuzzing against
brute-force invertible-element counts (`test_unitgroup.py`), witness mutation
tests against eight classes of forged strict witness and seven of forged
general-stabilizer witness (`test_witness.py`), Smith-certificate
mutation tests (`test_certificates.py`), the exhaustive `6^n` completeness
sweep (`test_completeness.py`), and brute-force pair-partition validation
against all 720 elements of `Sp(4,2)` (`test_stabilizer.py`).

---

## 5. Remaining limitations

Stated so nobody has to discover them by experiment.

**Sampled, not exhaustive.**

* `logical/generated.py` (`analyze_one_block`) and `ansatz/twofold.py`
  (`two_fold_group`) *sample* involutions. `is_full` is `True` or undecided,
  never `False`. The involution supply is a mixture of structural candidates,
  BLISS-discovered dualities, affine candidates for `n = 2^m`, automorphism
  samples, and random matchings.
* `ansatz/discovery.py` searches a structured family of blockwise involutions
  under a time budget and per-family caps (`_FAMILY_CAP = 6`,
  `_HALF_ENUM_CAP = 60000`). Missing a matching costs coverage, never
  soundness.

**Capped.**

* `ansatz/strict.py` / `ansatz/partition.py` enumerate at most
  `2^{ENUMERATION_DIM_CAP}` with `ENUMERATION_DIM_CAP = 24`; above that only
  the structured route can answer, and if it declines the result is `UNKNOWN`.
  The 0.2.1 σ-refinement moved several instances under this cap but did not
  raise it.
* `partition_units_via_structure` refuses the symplectic cut when
  `|A^×|` exceeds `group_enumeration_cap = 2_000_000`, because the cut is done
  by enumerating the unit group and filtering.  This is the dominant scaling
  limit and the one with a known structural fix — see the 0.2.1 section.
* `logical/group.py` uses a Schreier–Sims node budget (2M points, tightened to
  1M above `k = 14`) and a closure cap; `logical/generated.py` additionally
  applies a memory budget of 256 MB to closure enumeration.
* `ansatz/codeword_permutation.py` caps selected codewords at 50 000 and total
  enumeration work at 2 000 000 message combinations; beyond that it reports a
  certified subgroup with a note, not the full group.
* `hierarchy/frames.py` is exhaustive only while `3^n ≤ frame_limit`
  (default 60 000, i.e. `n ≤ 10`); beyond it only the three uniform frames are
  tried.
* `hierarchy/general.py`'s exact engine needs
  `rank(A) + dim(T) ≤ 12`; beyond that it falls back to the sound engine and
  reports `complete = False` unless `z_dim == 0`.
* `logical/generated.py::factor_target` refuses `k > 8` and can fail to build
  its word chain from `k ≥ 6`; the generator provenance records a label and an
  involution index rather than the physical layer parameter, so a returned word
  is not yet directly executable on hardware.
* `ansatz/monomial.py` enumerates the whole stabilizer group only for
  `rank ≤ 14`; above that it is row-set scoped.
* `ansatz/twofold.py`'s Levi factor enumerates only `K = I + basis element`
  perturbations once the Levi algebra exceeds dimension 12.

**Heuristic (but never trusted).**

* The composition-flag route in `algebra/radical.py` is heuristic; everything
  derived from it is verified exactly downstream, so a bad flag can only
  produce an honest `unknown`.
* Block unit generators in `algebra/wedderburn.py` are found by random search,
  then *certified* against `|GL(d,q)|` by a Schreier–Sims chain. Failure to
  certify within 600 attempts returns `None`, not a guess.

**Scope, not budget.**

* Cross-block gates are supported only for the *transversal* partition, one
  gate per corresponding-qubit tuple (0.2.1).  Arbitrary interleavings of
  qubits from different blocks are expressible but unexplored, and blocks of
  different codes are supported by `tensor_product` but untested at scale.
* Sign-exactness is automatic only for the strict diagonal generators
  (`certificates/phase.py`). Fold, partition, monomial and one-block orders are
  symplectic actions modulo Paulis and global phases; they can be verified
  sign-exactly on demand via `certificates/signed.py` but are not by default.
* Witness export supports the CSS strict route and the *enumeration* route of
  the general stabilizer solver. A witness for the structured
  (Wedderburn/unit-group) route is not implemented — `export_stabilizer_witness`
  refuses it explicitly rather than emitting a weaker document.
* No external computer-algebra backend. Logical group orders beyond the
  Schreier–Sims and recognition ranges are reported as lower bounds.
* `codes/families.py` is dense NumPy `uint8`; the asymptotic floor is `rref`,
  so `n` well beyond a few thousand awaits an M4RI-class backend.

---

## 6. Potential paper-level contributions

Separated deliberately. See `docs/related_work.md` for the citations behind
every line of the first column.

### Established theory that this package implements

CSS strict-transversal parameter codes and the three-layer structure theorem
(Albert); fold-transversal and fixed-matching constructions
(Breuckmann–Burton, Eberhardt–Steffan, Albert); diagonal Clifford-hierarchy
characterisation (Webster–Quintavalle–Bartlett) and ansatz-parameterised
diagonal search (Bauer); automorphism-derived logical Cliffords (Sayginel et
al.); monomial GF(4) automorphisms (CRSS); the single-qubit transversal
structure theorem (Zeng–Cross–Chuang); radical and Wedderburn computation
(Cohen–Ivanyos–Wales, Friedl–Rónyai); Leon-style row-space automorphism
computation; the endomorphism-algebra view of stabilizer codes (Rains;
Dasu–Burton). None of this is claimed as new.

### New computational machinery in this package

These are engineering contributions — new implementations, not new mathematics:

* a *parameterised* preservation-algebra solver: one code path answers the
  strict, fixed-matching, arbitrary-partition and whole-block questions for
  arbitrary stabilizer codes, with the symplectic cut applied after the unit
  group rather than inside a search;
* a certified unit-group solver over `F_2` in which every stage — nilpotency,
  semisimplicity, block identification, per-block generation — carries a
  machine-checked certificate and a failed check degrades to `UNKNOWN`;
* the filtration-adapted basis construction for `1 + I`, with the explicit
  three-dimensional counterexample showing why an arbitrary RREF basis
  under-generates;
* Smith-form *completeness* certificates for `Z_{2^L}` kernels, verifiable by
  an independent checker;
* a McLaughlin-transvection recognition certificate for
  `G = Sp(2k,2)` at `k` far beyond order computation, with every
  sub-certificate exact and a three-valued verdict;
* the characteristic-codeword row-space automorphism engine with a
  Brouwer–Zimmermann-style completeness bound, giving MAGMA/GAP-class
  capability on a `python-igraph` dependency;
* standalone witness verifiers that import nothing from the package.

### Possible theoretical contributions, unproven and unpublished

Both are stated as hypotheses in `related_work.md` and neither should be
described as new in print without a fuller literature review:

1. Representing prescribed-partition code-preserving Clifford transformations
   of an arbitrary stabilizer code as the symplectic unit subgroup of a finite
   preservation algebra, and computing the resulting physical and logical
   groups exactly with explicit completeness certificates. The ingredients are
   all prior art; the combination and its parameterisation over the partition
   may not be.
2. "Proof-carrying computation" as a design discipline for gate search: exact
   answer + mathematical witness + independent verifier, across multiple
   backends, with a three-valued completeness field that never collapses a
   failed computation into a negative result.

A concrete, checkable claim that would need writing up separately: the
package's census contains codes whose certified two-local (`|C| = 2`) group is
strictly larger than their strict group by a factor that is exact and
certified on both sides — e.g. `[[4,2,2]]`, logical order 6 → 48. Whether the
resulting family of exact locality-vs-logical-group data says something new
about the Chakraborty–Gottesman ceiling or Holmes' depth lower bounds is an
open question this package can now supply evidence for, not one it answers.
