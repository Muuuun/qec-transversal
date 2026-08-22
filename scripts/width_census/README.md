# Reproducing `docs/zoo/width_census.json`

Exhaustive fixed-partition sweeps: for a code and a width cap `w`, solve
*every* partition of the qubits into cells of size ≤ `w` and record the best
logical image. A row is a certified negative only when it reports
`unknown = 0`, i.e. every partition came back `Completeness.COMPLETE`.

Run from the repository root with this directory on `PYTHONPATH` (the shard
scripts import `staircase.partitions_max_cell`):

```bash
export PYTHONPATH=scripts/width_census

# registry codes: one process, all widths listed
python scripts/width_census/staircase2.py c4-22:2 c4-22:3 c6-22:2 c6-22:3 c6-22:4

# larger sweeps, sharded across cores (writes shard_<code>_<width>_<i>.json)
for i in $(seq 0 7); do
  python scripts/width_census/shard.py cube-832 3 $i 8 /tmp/out &
done; wait

# codetables best-known codes, fetched and cached by
# scripts/codetables_n7_census.py (n:k:width:shard:shards:outdir)
for i in $(seq 0 7); do
  python scripts/width_census/ct_shard.py 10 2 2 $i 8 /tmp/out &
done; wait

# the decomposable control (Chakraborty-Gottesman's tight example)
python scripts/width_census/stacked.py

# is a code a tensor product?  exhaustive bipartition test
python -c "import decomp, ..."   # decomposition(code) -> bipartition or None
```

`xval.py` is the cross-validation of the φ-index cut against the enumeration
route (1086 partitions, zero mismatches); `wide2.py` runs single wide
partitions through `method="phi"` and prints `|A^x|`, the orbit size and `|G|`.

Partition distances quoted alongside the census come from
`qec_transversal.faulttolerance.partition_distance` on each row's best
partition.

These are research scripts, not part of the package API: they print and dump
JSON, and the merge step for sharded runs is a few lines of `json.load` plus
`logical.group.schreier_sims_order` over the union of the logical generators.
