# qec-transversal

`qec-transversal` finds and certifies strict-transversal logical Clifford
generators of a CSS stabilizer code directly from its binary check matrices.
It implements the parameter-code construction in Victor V. Albert's
[*Beyond transversality: structure of Clifford circuits for CSS codes*](https://arxiv.org/abs/2608.05688).

The input is

```text
H_X, H_Z over GF(2), with H_X H_Z^T = 0.
```

The output includes complete bases of

```text
A_Z = {a : a * C_X is contained in C_Z}
A_X = {b : b * C_Z is contained in C_X},
```

the corresponding physical `sqrt(Z)` / `sqrt(X)` supports, a paired logical
Pauli basis, each generator's logical symplectic matrix, small-group order, and
verification certificates. Here `*` denotes coordinatewise multiplication.

## Status

Version 0.1 covers the complete strict-transversal CSS Clifford group modulo
Paulis. It does **not yet** cover fixed-matching two-local layers, matching
search, non-CSS codes, or phase-sensitive Pauli dressing. See
[the implementation landscape](docs/landscape.md) for how this differs from
the paper's official certificate repository and `autqec`.

## Install and run

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
qec-transversal analyze examples/steane.json -o steane-result.json
```

The JSON input accepts nested binary rows or bit strings:

```json
{
  "H_X": ["1111000", "1100110", "1010101"],
  "H_Z": ["1111000", "1100110", "1010101"]
}
```

Use `--include-constraints` to include the nullspace constraint matrices and
`--include-physical` to include every `2n x 2n` physical symplectic matrix.
The built-in logical group closure is exact until `--group-cap` elements; a
capped result is labelled as a lower bound, never as a negative result.

## Python API

```python
from qec_transversal import CSSCode

code = CSSCode(H_X, H_Z)
analysis = code.analyze_transversal()

print(analysis.a_z.basis)
print(analysis.a_x.basis)
print(analysis.to_dict(group_cap=100_000))
```

## What is certified

For every reported parameter-space basis and physical generator, the tool
checks:

- `H_X H_Z^T = 0` and rank-derived `[[n,k]]`;
- the parameter is in the relevant exact nullspace;
- the physical matrix is symplectic and preserves the stabilizer row space;
- the closed-form logical shear agrees with an independent projection on
  `C^perp / C`;
- the logical matrix is symplectic.

The logical group order is computed by explicit closure only when it fits under
the configured cap. GAP/MeatAxe backends are planned for larger `k`.

## Roadmap

1. Fixed-matching exact `S_M^Z`, `S_M^X`, and `L_M` solvers.
2. Geometry-, Tanner-graph-, and symmetry-guided matching generation.
3. GAP/MeatAxe logical-group recognition and target-gate membership words.
4. Three-layer circuit compression and phase-sensitive Pauli dressing.
5. Non-CSS transversal search through SAT/SMT.

## Development

```bash
python -m pytest
ruff check .
```

## References

- [Paper: arXiv:2608.05688](https://arxiv.org/abs/2608.05688)
- [Official paper code and certificates](https://github.com/valbert4/two-fold-transversal)
- [autqec](https://github.com/hsayginel/autqec)

## License

MIT. No code or data is vendored from the related repositories.

