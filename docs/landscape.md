# Existing-work landscape

Search performed on 2026-08-10 using the paper identifier, exact title, and
algorithm terms on arXiv and GitHub; qLDPC entry and cross-check results
added 2026-08-19.

## Closest projects

- [qLDPCOrg/qLDPC](https://github.com/qLDPCOrg/qLDPC) is the broadest
  construction-and-analysis library in the space: code families (BB, HGP,
  lifted product, quantum Tanner), a GAP-backed abstract-algebra module, and
  `get_transversal_ops` — SWAP-transversal logical Cliffords via the code
  automorphism method of arXiv:2409.18175.  Complementary along the same
  axis as autqec: its transversal route searches (and hard-requires
  GAP+GUAVA or MAGMA); it does not decide completeness or certify
  nonexistence.  Its SWAP-transversal class is a subclass of this repo's
  monomial class, which makes it a natural cross-check target.

- [valbert4/two-fold-transversal](https://github.com/valbert4/two-fold-transversal)
  is the official code-and-certificate repository for
  [arXiv:2608.05688](https://arxiv.org/abs/2608.05688). It contains certified
  survey data, fixed-matching search scripts, and special-case exhaustive
  enumerations. Its scripts are research-reproduction artifacts tied to the
  repository datasets; it does not expose an installable API or CLI accepting
  arbitrary `H_X,H_Z` matrices.
- [hsayginel/autqec](https://github.com/hsayginel/autqec) is an installable
  Python package for finding logical Clifford gates through code
  automorphisms. It is complementary: its primary search representation is a
  related binary-code automorphism group, optionally using MAGMA or Bliss.
- [QuantumClifford.jl issue #434](https://github.com/QuantumSavory/QuantumClifford.jl/issues/434)
  tracks a proposed automorphism-based implementation, but is not an
  implementation of the parameter-code and matching algorithms of the 2026
  paper.

## Gap addressed here

`qec-transversal` provides a small public API and reproducible CLI whose direct
input is an arbitrary pair of binary CSS check matrices. Version 0.1 computes
the complete strict-transversal parameter spaces `A_Z,A_X`, constructs a paired
logical basis, projects every physical generator to `Sp(2k,2)`, and emits
machine-checkable certificates. It intentionally does not duplicate the
official repository's survey datasets or bespoke certification scripts.

## Evidence boundary

The completeness claim in version 0.1 is only for strict-transversal Clifford
gates of CSS codes, modulo Pauli phases. Fixed-matching two-local search,
matching-orbit generation, non-CSS SAT/SMT search, Pauli dressing, and scalable
matrix-group recognition remain future modules.

