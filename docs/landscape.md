# Existing-work landscape

Search performed on 2026-08-10 using the paper identifier, exact title, and
algorithm terms on arXiv and GitHub.

## Closest projects

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

