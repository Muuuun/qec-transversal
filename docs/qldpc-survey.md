# Strict-transversal survey of the built-in qLDPC registry

> **Dated snapshot (2026-08-10, 37 codes); scope note added 2026-08-13.**
> Headline 1 below is a statement about *the families tested in this run*, not
> about check sparsity.  The registry has since gained the self-dual bivariate
> bicycle codes of [arXiv:2510.05211](https://arxiv.org/abs/2510.05211):
> weight-8 LDPC codes that **do** carry a certified strict transversal H and S
> (logical group order 6).  Sparse codes with transversal non-Clifford gates
> are known too ([arXiv:2410.14662](https://arxiv.org/abs/2410.14662),
> [arXiv:2310.16982](https://arxiv.org/abs/2310.16982)).  Read the live census
> at <https://muuuun.github.io/qec-transversal/> for the current registry.
>
> **Scope, restated for 0.2 (2026-08-20).**  Everything below concerns *one*
> of the seven gate ansätze the package now implements: the **strict** class —
> one arbitrary single-qubit Clifford per qubit, no permutation, depth one.  A
> trivial result here says nothing about fixed-matching (fold) layers,
> prescribed-partition two-local layers, permutation gates, or diagonal
> hierarchy gates, all of which the same registry does carry and all of which
> the live census reports separately.  See
> [`../README.md`](../README.md) for the capability table and
> [`mathematics.md`](mathematics.md) §6 for what "complete" means per ansatz.

Run on 2026-08-10 with `qec-transversal` (post-0.1.0 working tree) over every
code in `qec_transversal.codes.REGISTRY`.  All 37 analyses are certified.
Timings are the full pipeline (build, `CSSCode`, analysis, report) on an
Apple-silicon laptop.  Reproduce any row with:

```bash
qec-transversal list-codes
qec-transversal analyze --code <name>
```

## Headline results

1. Every quantum LDPC family tested — bivariate bicycle (all seven Bravyi
   et al. Table 3 instances including the gross and two-gross codes),
   coprime-BB, trivariate bicycle, generalized bicycle, hypergraph product,
   La-cross, the Panteleev-Kalachev lifted product B1, and three Kasai-style
   quasi-cyclic codes up to `[[2352, 800]]` — has a **completely trivial
   strict-transversal group**: `A_Z = A_X = {0}`, so no depth-one layer of
   single-qubit `sqrt(Z)`/`sqrt(X)` gates acts as a nontrivial logical
   Clifford.  This matches the literature: the useful Clifford gates of these
   families are *fold*-transversal (Breuckmann-Burton arXiv:2202.06647,
   Eberhardt-Steffan arXiv:2407.03973) or automorphism-induced
   (arXiv:2409.18175), i.e. they require qubit permutations or two-local
   layers, which are outside the strict-transversal search space by
   definition (see the roadmap).

2. The positive controls all recover their textbook gates, with the exact
   logical group order computed by the new Schreier-Sims engine:

   | code | [[n,k,d]] | dim A_Z / A_X | logical group | note |
   |---|---|---|---|---|
   | `steane` | [[7,1,3]] | 1 / 1 | order 6 = full Sp(2,2) | transversal S and H |
   | `qrm15` | [[15,1,3]] | 5 / 0 | order 2 | transversal S from the T-code; 4 of 5 generators logically trivial |
   | `c4-22`, `c6-22` | [[4,2,2]], [[6,2,2]] | 1 / 1 | order 6 | global S-bar, H-bar action |
   | `tesseract` | [[16,6,4]] | 1 / 1 | order 6 | self-dual doubly-even |
   | `rm64` | [[64,20,8]] | 1 / 1 | order 6 | middle Reed-Muller RM(2,6) |
   | `grid-4x6`, `grid-6x8` | [[24,8,4]], [[48,24,4]] | 1 / 1 | order 6 | Albert bipartite-grid codes |

   For every self-dual entry the report's `structure` block shows
   `self_dual: true` and the all-ones vector in both parameter spaces —
   the Albert Cor. E.6 signature of a transversal Hadamard.

3. Negative topological controls (`toric-4`, `toric-10`, `surface-5`)
   are trivial, as expected.

## Registry sweep

| code | [[n,k,d]] | family | dim A_Z / A_X | time |
|---|---|---|---|---|
| bb72 | [[72,12,6]] | bivariate bicycle | 0 / 0 | <0.1s |
| bb90 | [[90,8,10]] | bivariate bicycle | 0 / 0 | <0.1s |
| bb108 | [[108,8,10]] | bivariate bicycle | 0 / 0 | <0.1s |
| gross | [[144,12,12]] | bivariate bicycle | 0 / 0 | <0.1s |
| two-gross | [[288,12,18]] | bivariate bicycle | 0 / 0 | <0.1s |
| bb360 | [[360,12,<=24]] | bivariate bicycle | 0 / 0 | 0.1s |
| bb756 | [[756,16,<=34]] | bivariate bicycle | 0 / 0 | 0.3s |
| bb54 | [[54,8,6]] | bivariate bicycle | 0 / 0 | <0.1s |
| bb98-symmetric | [[98,6,12]] | symmetric BB | 0 / 0 | <0.1s |
| bb162-symmetric | [[162,8,12]] | symmetric BB | 0 / 0 | <0.1s |
| coprime30..154 | five codes | coprime BB | 0 / 0 | <0.1s |
| trivariate30 | [[30,4,5]] | trivariate bicycle | 0 / 0 | <0.1s |
| gb48, gb46, gb126 | PK A3, A4, A2 | generalized bicycle | 0 / 0 | <0.1s |
| hgp-hamming | [[58,16,3]] | hypergraph product | 0 / 0 | <0.1s |
| lacross65 | [[65,9,4]] | La-cross | 0 / 0 | <0.1s |
| lacross400 | [[400,16,8]] | La-cross | 0 / 0 | 0.1s |
| lifted-b1 | [[882,24,<=24]] | lifted product | 0 / 0 | 0.3s |
| kasai-binary-294 | [[294,100]] | Kasai QC pair | 0 / 0 | 0.1s |
| kasai-binary-1104 | [[1104,554]] | Kasai QC pair | 0 / 0 | 1.3s |
| kasai-gf256-2352 | [[2352,800]] | Kasai GF(256) | 0 / 0 | 7.4s |

Notes on the two symmetric BB codes: Eberhardt-Steffan construct rich
*fold-transversal* gate groups for `bb98-symmetric` and `bb162-symmetric`
(CZ/S-type gates across the ZX-duality fold).  Those gates use two-qubit CZ
layers and qubit permutations, so a trivial *strict*-transversal result here
is consistent, and quantifies how much of their gate group genuinely needs
the fold: all of it.

Note on Kasai codes ("quantum error correction near the coding-theoretical
bound", Komoto-Kasai arXiv:2412.21171): the registry builds the exact
girth-12 binary orthogonal pair of arXiv:2501.13444 (Definition 6) and a
GF(256) lift with the canonical separable label assignment of
arXiv:2510.25583 expanded through companion matrices.  Separable labels
keep the binary rank, giving `k = e((L-4)P + 2)` (here 800) instead of the
published full-rank `k = e(L-4)P` (784); the support structure, orthogonality
and girth match the paper.

## Is the method actually decisive? Exhaustive validation

The tool only searches the two *diagonal* families (`sqrt(Z)`/`sqrt(X)`
layers).  A fair question is whether that misses strict-transversal gates
that mix X and Z per qubit (Hadamard-type layers, or a different single-qubit
Clifford on every qubit).  Albert's theorem (arXiv:2608.05688, Eqs. 18-23)
says no: the diagonal families generate the *entire* strict-transversal
Clifford group mod Paulis, through the normal form
`g = H(t) U_Z(q) U_X(p)` with `q * p = 0`.

We validated this exhaustively on 46 codes: for every code with `n <= 8`,
all `6^n` assignments of arbitrary single-qubit Cliffords (one per qubit,
up to 1.68 million assignments for the `[[8,2,2]]` toric code) were checked
for stabilizer preservation, and the surviving set was compared against the
closure of the tool's `A_Z`/`A_X` generators.  On 40 random CSS codes
(`n = 2..6`, including degenerate and `k = 0` cases) and on `[[4,2,2]]`,
`[[6,2,2]]`, Steane, the `d = 2` surface and toric codes, and a `k = 0`
grid code, three quantities agreed **exactly in every case**:

1. the brute-force strict-transversal group (physical, mod Pauli),
2. the group generated by the diagonal `A_Z`/`A_X` layers alone,
3. Albert's Eq. 23 counting formula.

The induced logical groups and the existence verdicts also agreed in every
case.  In particular the toric code's brute force found *only the identity*
among all 1.68M single-qubit-Clifford layers - so a trivial `A_Z`/`A_X`
answer really is a proof of non-existence for strict-transversal gates, not
a blind spot of the diagonal-only search.  A fast version of this check is
kept in `tests/test_completeness.py`.

Caveat on scope: this certifies the *strict* (depth-one, single-qubit,
no-permutation) class.  Fold-transversal gates, which add qubit permutations
and two-qubit CZ layers, are a strictly larger class - that is where BB-code
gates live.  As of 0.2 that class is implemented too
(`matching_clifford_group`, `partition_clifford_group`,
`permutation_automorphism_group`, `one_block_clifford_group`); the sentence
that used to end here calling it "the roadmap's next step" was written before
those existed and has been corrected.

## Reproducing

```bash
qec-transversal analyze --code gross | python -m json.tool | head
qec-transversal generate two-gross -o two-gross.json
qec-transversal analyze two-gross.json --compact
```
