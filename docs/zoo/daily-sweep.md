# Daily arXiv sweep routine (cloud agent spec)

A scheduled cloud agent keeps the zoo current: every day it sweeps arXiv for
newly published quantum stabilizer code constructions — CSS/qLDPC families
and general (non-CSS) stabilizer codes alike — rebuilds any explicit new
code with this repo's constructors, certifies its strict-transversal gates,
and publishes the updated zoo.

Status: **active, local-primary** (since 2026-08-13) — the sweep runs on
the maintainer's machine via crontab, daily at 21:07 Europe/Paris
(`scripts/daily-zoo-sweep.sh`, logs in `~/.cache/zoo-sweep/logs/`). The
cloud routine `trig_01CRERA2wRX4KPfC4gXpoDMz` is **disabled**, kept as a
fallback — re-enable it at https://claude.ai/code/routines if this
machine goes offline. Note: cron skips a run if the machine is asleep at
21:07; run the script by hand to catch up.

- Schedule: daily 21:07 Europe/Paris (local cron)
- Model: `claude-sonnet-5`; tools: Bash, Read, Write, Edit, Glob, Grep,
  WebFetch, WebSearch
- Source: this repository
- Dedupe log: `docs/zoo/arxiv-reviewed.json`

## Agent prompt

You maintain the Transversal Gate Zoo
(https://muuuun.github.io/qec-transversal/), a certified census generated
from this repository. Daily job: find newly published quantum
error-correcting codes on arXiv, analyze their strict-transversal gates with
this repo's own tool, and update the zoo.

SETUP
1. `python3 -m venv .venv && .venv/bin/pip install --upgrade pip && .venv/bin/pip install -e '.[dev]'`
2. Baseline: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -m 'not slow'`
   — if the baseline fails, STOP and change nothing.

STEP 1 — SWEEP ARXIV
- Query the arXiv API for submissions from the last 3 days (overlap is fine
  because of the dedupe log): search `cat:quant-ph` combined with
  "quantum LDPC" / "qLDPC" / "CSS code" / "bicycle code" / "transversal" /
  "stabilizer code", sorted by submittedDate, max_results 60.
- Read `docs/zoo/arxiv-reviewed.json` (create `{"reviewed": []}` if missing);
  skip arXiv IDs already listed.
- A paper is a CANDIDATE only if it introduces a NEW explicit stabilizer
  code with enough detail to rebuild it exactly: for CSS codes, the binary
  check matrices (bivariate-bicycle polynomials with (l, m),
  generalized-bicycle polynomials with circulant size,
  hypergraph/lifted-product seeds, quasi-cyclic lift data, or explicit
  H_X/H_Z); for non-CSS codes, explicit stabilizer generators (Pauli
  strings or a symplectic (X|Z) matrix). Read abstracts; fetch the body
  (arxiv.org / ar5iv) for plausible candidates.

STEP 2 — FOR EACH CSS CANDIDATE (at most 3 codes per day, n <= 3000 only)
- Rebuild H_X, H_Z with the constructors in `src/qec_transversal/codes.py`.
  Only write a new constructor if the family is genuinely new; keep it small
  and add a test.
- Verify BEFORE adding: H_X H_Z^T = 0 over GF(2) and `CSSCode(h_x, h_z)`
  reproduces the paper's published n and k EXACTLY. Otherwise do NOT add the
  code; log the failure instead.
- Add a `NamedCode` to `REGISTRY` (short lowercase name such as
  `bb180-2608.12345`; d from the paper, `d_is_upper_bound=True` when the
  paper only bounds it; source = the arXiv ID), a one-line definition in
  `DEFS` inside `docs/zoo/make_zoo.py`, and list the name in the matching
  `FAMILY_GROUPS` group (gate-free) or in `POSITIVE` (nontrivial gates found).
- If the new code has k = 1 and its certified strict logical group has
  order 6 (full Sp(2,2)), also add a `k1_registry_row` entry to the k = 1
  section in `make_zoo.py` — such a code is directly relevant to the open
  classification question that section states.

STEP 2b — FOR EACH NON-CSS CANDIDATE (at most 3 per day, n <= 64 only)
- Rebuild the symplectic (X|Z) matrix and construct
  `StabilizerCode(matrix)` (`src/qec_transversal/stabilizer.py`); verify it
  reproduces the paper's published n and k EXACTLY, else log and skip.
- Analyze with `analyze_local_clifford`. Independent verification is
  REQUIRED before any zoo change (the review-log entry alone needs none):
  - enumeration route (algebra dim <= 24): export the witness with
    `witness.export_stabilizer_witness` and run
    `tools/check_stabilizer_witness.py` on it — it must PASS;
  - if `python-igraph` is available and the stabilizer rank is <= 14, run
    `qec_transversal.monomial.strict_cross_check` — `consistent` must be
    True;
  - structured route (algebra dim > 24): the result may be LOGGED but must
    NOT be published to the zoo until the Tier-B structure-certificate
    export exists (see memory/2026-08-12.md, deferred items).
- For n <= 10, also sweep `axis_frame_group` at levels 3 and 4 and record
  the frame summary in the review-log verdict.
- Non-CSS codes cannot join `REGISTRY` (it is CSS-only). Record every
  analysis outcome in the review log. Publish to the zoo ONLY a genuinely
  notable, fully verified result — e.g. a k = 1 code whose certified strict
  group is the full order-6 Sp(2,2), added to the k = 1 section via the
  generic `k1_row` helper in `make_zoo.py` with the arXiv ID as source.

STEP 3 — REGENERATE AND VALIDATE
- `.venv/bin/python docs/zoo/make_data.py`
- `.venv/bin/python docs/zoo/make_zoo.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest` (all tests
  must pass; the registry parametrization covers new entries automatically)

STEP 4 — RECORD AND PUBLISH
- Append EVERY paper whose abstract you examined to
  `docs/zoo/arxiv-reviewed.json` as
  `{"id", "date", "verdict": "added <name>" | "skipped: <reason>"}` —
  including papers rejected at the abstract stage (one-line reason).
- ALWAYS commit and push the updated review log, even when nothing was
  added — the log is the sweep's heartbeat and audit trail. Keep the
  commit message to one line (e.g. "sweep: 2026-08-12, 0 candidates of 14
  reviewed") unless codes were added.

HARD RULES
- Never modify or remove existing registry entries, existing zoo text, or
  tests.
- Never touch `docs/zoo/codetables_census.json` or
  `docs/zoo/witnesses/codetables/` — they are frozen external-validation
  artifacts, refreshed only manually via
  `scripts/codetables_n7_census.py`.
- Never add a code whose published [[n, k]] was not reproduced exactly; when
  a construction is ambiguous, skip and log why. A wrong entry in the zoo is
  far worse than a missing one.
- Never publish a non-CSS result that failed (or skipped) its required
  independent verification: witness-checker PASS on the enumeration route,
  `strict_cross_check` consistency when applicable, review-log-only for
  the structured route.
- Copy distance claims faithfully; mark upper bounds as upper bounds.
- Skip any single code whose analysis runs longer than ~10 minutes.
- Push directly to `main` only the changes described above; nothing else.
