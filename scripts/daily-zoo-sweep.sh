#!/bin/bash
# Daily local runner for the Transversal Gate Zoo arXiv sweep — the
# PRIMARY daily operation since 2026-08-13 (the cloud routine
# trig_01CRERA2wRX4KPfC4gXpoDMz is disabled; re-enable at
# https://claude.ai/code/routines if this machine goes offline).
# Works in an isolated clone so the interactive working tree is never touched.
# Installed in crontab at 21:07 local; follows docs/zoo/daily-sweep.md and
# logs to ~/.cache/zoo-sweep/logs/.
set -euo pipefail

export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"

WORKDIR="$HOME/.cache/zoo-sweep"
CLONE="$WORKDIR/qec-transversal"
LOGDIR="$WORKDIR/logs"
mkdir -p "$LOGDIR"

if [ -d "$CLONE/.git" ]; then
    git -C "$CLONE" fetch origin
    git -C "$CLONE" reset --hard origin/main
    git -C "$CLONE" clean -fd
else
    git clone https://github.com/Muuuun/qec-transversal.git "$CLONE"
fi

cd "$CLONE"
claude --dangerously-skip-permissions -p \
  "Read the file docs/zoo/daily-sweep.md in this repository and execute its \
'Agent prompt' section exactly as written. It is your complete task \
specification for today's run: sweep arXiv for new explicit stabilizer \
codes (CSS/qLDPC and non-CSS alike), rebuild and certify any \
reconstructible ones with this repo's tool, update the zoo, validate with \
the test suite, log every reviewed paper in docs/zoo/arxiv-reviewed.json, \
and push to main only under the conditions the spec states. Obey its HARD \
RULES without exception. You are running headless on the maintainer's \
machine inside a dedicated clone; git push uses the locally authenticated \
gh credentials."
