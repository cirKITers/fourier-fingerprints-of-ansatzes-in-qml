#!/usr/bin/env bash
# The engine. Run from the repo root, so its store is the default ./.fluksio.
# RUNS is the number of runs (and python workers) at once, DEVICES the JAX CPU
# device count per worker, over which jaqsi splits each circuit batch; keep
# RUNS x DEVICES near the core count and RUNS within memory. MAX_RSS (MB)
# retires a warm worker that holds more when its node returns (0: never).
#
#     RUNS=2 DEVICES=4 dev/serve.sh
set -u
cd "$(dirname "$0")/.."
exec env JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES="${DEVICES:-1}" uv run fluksio serve --port "${PORT:-8766}" \
  --max-runs "${RUNS:-10}" --max-cascades "${RUNS:-10}" --max-workers "${RUNS:-10}" \
  --worker-max-rss "${MAX_RSS:-0}"
