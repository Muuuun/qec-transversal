# Daily arXiv sweep routine (cloud agent spec)

A scheduled cloud agent keeps the zoo current: every day it sweeps arXiv for
newly published CSS/qLDPC code constructions, rebuilds any explicit new code
with this repo's constructors, certifies its strict-transversal gates, and
publishes the updated zoo.

Status: **spec ready, awaiting activation** — creating the routine requires
the repo owner to connect GitHub to claude.ai once
(https://claude.ai/code/onboarding?magic=github-app-setup), because the cloud
agent needs push access to `main` (GitHub Pages redeploys from `docs/`).

- Schedule: daily at 07:00 UTC (09:00 Paris) — cron `0 7 * * *`
- Model: `claude-sonnet-5`; tools: Bash, Read, Write, Edit, Glob, Grep,
  WebFetch, WebSearch
- Source: this repository
- Dedupe log: `docs/zoo/arxiv-reviewed.json` (created on first run)

## Agent prompt

You maintain the Strict-Transversal Gate Zoo
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
  "quantum LDPC" / "qLDPC" / "CSS code" / "bicycle code" / "transversal",
  sorted by submittedDate, max_results 60.
- Read `docs/zoo/arxiv-reviewed.json` (create `{"reviewed": []}` if missing);
  skip arXiv IDs already listed.
- A paper is a CANDIDATE only if it introduces a NEW explicit CSS code with
  enough detail to rebuild the exact binary check matrices
  (bivariate-bicycle polynomials with (l, m), generalized-bicycle polynomials
  with circulant size, hypergraph/lifted-product seeds, quasi-cyclic lift
  data, or explicit H_X/H_Z). Read abstracts; fetch the body (arxiv.org /
  ar5iv) for plausible candidates.

STEP 2 — FOR EACH CANDIDATE (at most 3 codes per day, n <= 3000 only)
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

STEP 3 — REGENERATE AND VALIDATE
- `.venv/bin/python docs/zoo/make_data.py`
- `.venv/bin/python docs/zoo/make_zoo.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest` (all tests
  must pass; the registry parametrization covers new entries automatically)

STEP 4 — RECORD AND PUBLISH
- Append every reviewed paper to `docs/zoo/arxiv-reviewed.json` as
  `{"id", "date", "verdict": "added <name>" | "skipped: <reason>"}`.
- If at least one code was added OR at least one candidate was reviewed,
  commit with a descriptive message and push to `main` (Pages redeploys the
  zoo automatically). If the sweep surfaced nothing relevant, exit WITHOUT
  committing.

HARD RULES
- Never modify or remove existing registry entries, existing zoo text, or
  tests.
- Never add a code whose published [[n, k]] was not reproduced exactly; when
  a construction is ambiguous, skip and log why. A wrong entry in the zoo is
  far worse than a missing one.
- Copy distance claims faithfully; mark upper bounds as upper bounds.
- Skip any single code whose analysis runs longer than ~10 minutes.
- Push directly to `main` only the changes described above; nothing else.
