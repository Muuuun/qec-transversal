# Partition distance: the fault-tolerance criterion for a depth-one layer

This note records the corrected form of a claim that was circulating in the
project's notes — "if the minimum native gate size `ℓ_min` needed to realize a
logical gate is at least the code distance `d`, then the gate has no
fault-tolerant depth-one implementation" — and the machinery
(`qec_transversal.faulttolerance`) that decides the real criterion.

The claim is false as stated. `ℓ ≥ d` is not an obstruction, and the project's
own certified census already contains a counterexample to the converse reading.

## What actually governs one faulty gate

A depth-one layer of native gates factors as `U = ∏_C U_C` over the cells of a
partition `𝒫 = {C_1, …, C_m}` of the physical qubits. Because `U_C` acts only on
`C`, a fault in that native gate produces an error supported on `C` whether it
strikes before or after the gate — conjugating by `U_C` cannot move it out.
So the induced error set is exactly "any Pauli supported inside one cell", and
Knill–Laflamme decides correctability against *that* set.

For a stabilizer code with stabilizer group `S`, errors `E ∈ P_C`, `F ∈ P_D` are
distinguishable unless `E†F ∈ N(S) \ S`. With

    bwt_𝒫(L) = |{C ∈ 𝒫 : supp(L) ∩ C ≠ ∅}|
    d_𝒫      = min { bwt_𝒫(L) : L ∈ N(S) \ S }

this reads:

| situation | exact criterion |
|---|---|
| one unflagged faulty gate | `d_𝒫 ≥ 3` |
| `r` unflagged faulty gates | `d_𝒫 ≥ 2r + 1` |
| one *flagged* (located) faulty gate | `d_𝒫 ≥ 2` |

`d_𝒫`, not the cell size and not the code distance, is the quantity a depth-one
construction has to defend.

## How `ℓ` and `d` do and do not enter

Since a logical operator of block weight `b` has weight at most `bℓ` and at
least `d`:

    d_𝒫 ≥ ⌈d / ℓ⌉

which recovers the familiar **sufficient** condition `2ℓ < d ⟹ d_𝒫 ≥ 3`. It is
only sufficient. In the other direction the bound is vacuous: `ℓ ≥ d` gives
`d_𝒫 ≥ 1`, i.e. nothing. Distance says *some* nontrivial logical has weight `d`;
it never says that *every* set of `d` qubits contains one. The geometry of the
cells is what decides.

Two further corrections to the discarded argument:

- An `ℓ`-qubit Clifford `M` spreads a single-qubit Pauli to **at most** `ℓ`
  qubits. `spread(M)` (`σ(M) = max_i max_{P ∈ {X_i,Y_i,Z_i}} wt(MP)`) can be far
  below `ℓ` — a qubit permutation has `σ = 1` at any cell size. But a small
  spread is a fact about the *ideal* layer only: it bounds propagation through a
  correct gate, never the damage a faulty native gate does to its own outputs.
  Use `σ` to decide whether a coarse layer can be re-expressed on a finer
  partition, then compute `d_𝒫` on that finer partition.
- Weight `< d` guarantees only *detection*; correction of unknown-location
  errors needs `⌊(d-1)/2⌋`. The block-level analogue is the `2r + 1` row above.

## Deciding `d_𝒫 ≥ 3` is linear algebra, not SAT

`bwt_𝒫(L) ≤ 2` means `L` is supported inside `C ∪ D` for one of the `O(m²)` cell
unions, so the whole criterion reduces to `O(m²)` independent local questions:

> does `N(S) \ S` contain a Pauli supported inside `T = C ∪ D`?

Both halves are rank computations on `2|T|` columns:

- the Paulis on `T` commuting with `S` form the kernel `N_T` of the symplectic
  pairing against the checks that *touch* `T` (checks disjoint from `T`
  contribute no constraint — the LDPC-friendly restriction the CSS solver
  already uses);
- for `v ∈ N(S)`, the symplectic products against a paired logical basis are
  exactly `v`'s logical coordinates, so `v ∉ S` iff some product is nonzero.

`partition_distance(code, cells, max_blocks=2)` returns the exact `d_𝒫` with a
witness when one exists at that block weight, and otherwise the certificate
`d_𝒫 > 2`. Both verdicts are decided — the subset search at that size is
exhaustive — which is a stronger status than the one-sided `is_full` badges
elsewhere in this project.

The witness is an explicit obstruction, not a failed search:

- `bwt = 1`: one faulty gate on that cell produces a logical error directly.
- `bwt = 2`, `L = e_C + e_D`: the two single-gate errors `e_C` and `e_D` share a
  syndrome and differ by a logical, so no recovery corrects both.

## Certified counterexamples

Both are in `tests/test_faulttolerance.py`.

**`ℓ ≥ d` is not an obstruction.** Concatenate Steane outside Shor:
`[[7,1,3]] ∘ [[9,1,3]] = [[63,1,9]]`. Partition into the seven inner blocks, so
`ℓ = 9 ≥ d = 9`. Then

- `d_𝒫 = 3` exactly (verified by exhaustive search to block weight 3) — one
  entire faulty 9-qubit block is correctable;
- an explicit block-diagonal symplectic layer on those cells preserves the code
  and induces logical `H`; another induces logical `S`. Both are built by
  `encoded_lift`, which realizes any target logical Clifford as a physical
  Clifford preserving the code.

So a depth-one layer of native gates at cell size ≥ the distance is fault
tolerant and reaches the full single-qubit logical Clifford group. What this
does *not* claim is that `ℓ_min = 9` for that code — certifying `ℓ_min` would
require ruling out every finer partition. It kills the implication, not the
possibility that a `ℓ_min`-based criterion exists in some other form.

**Large `ℓ` is not required either.** `doubled41-2608.11160`, the
`[[41,1,9]]` doubled colour code already in the registry, is certified FULL for
the *strict* class: `ℓ = 1`, `d = 9`, logical group order 6 = `|Sp(2,2)|`, and
`d_𝒫 = d = 9` at the singleton partition. Steane is the same story at `d = 3`.

## The shape a real no-go would have to take

For a fixed partition, with `A_P(S)^×` the preservation algebra's unit group,

    G_𝒫 = A_P(S)^× ∩ ∏_{C ∈ 𝒫} Sp(2|C|, 2),    ρ : G_𝒫 → Sp(2k, 2)

the useful theorem is

    g ∈ ρ(G_𝒫)  ⟹  d_𝒫 ≤ 2,   for every admissible 𝒫.

Per partition the search then returns either a certificate that `g ∉ ρ(G_𝒫)`
(this project's existing `analyze_partition_clifford` / completeness machinery)
or a logical Pauli of block weight ≤ 2 (`partition_distance`). Two quantifier
traps to keep in view:

1. `ρ(G_𝒫)` is a group for each fixed `𝒫`; the **union** over partitions is not.
   Taking its generated closure describes a multi-layer circuit, which is
   exactly the varying-matching route the constant-depth theorems do not bound —
   and exactly what this project's `one-block group` column generates. A no-go
   about one layer says nothing there.
2. `Sp(2k,2)` is Cliffords modulo Paulis and phases. Addressable logical Paulis
   are always available at `ℓ = 1` regardless of their physical weight, so
   "addressable logical gate" must mean a **non-Pauli** Clifford, and stabilizer
   signs have to be checked separately (`certificates/signed`, `phase`).
