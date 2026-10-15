#!/usr/bin/env bash
# ============================================================================
# run_benchmark.sh — Why5 5-why protocol benchmark driver
#
# Three phases:
#   1. self-test (mock, perfect student)   — harness must score 100% / PASS
#   2. detect-test (mock, flawed student)  — harness MUST catch the flaws
#   3. live benchmark                       — real endpoint (skipped in MOCK=1)
#
# Usage:
#   MOCK=1 ./run_benchmark.sh                       # offline: phases 1-2 only
#   BENCH_URL=http://localhost:8001/v1 \
#     BENCH_MODEL=ckpt/why5-sft ./run_benchmark.sh  # full run incl. live
#
# Exit code: 0 = all phase expectations met, 1 = otherwise.
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"

N="${N:-60}"
CURVE_N="${CURVE_N:-10}"
BENCH_URL="${BENCH_URL:-http://localhost:8001/v1}"
BENCH_MODEL="${BENCH_MODEL:-why5-sft}"
MOCK="${MOCK:-0}"
RC=0

banner() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }

# ---- resolve python: venv from setup_5090_novita.sh, then system python ----
find_python() {
  for p in "${WHY5_VENV:-}" \
           "${WORK_DIR:-${HOME:-/root}/why5}/venv/bin/python" \
           "${HOME:-/root}/why5/venv/bin/python"; do
    [ -n "$p" ] && [ -x "$p" ] && { echo "$p"; return 0; }
  done
  # PATH interpreter first (venv users often activate one); hard-coded
  # absolute paths only as last resort
  if command -v python3 >/dev/null 2>&1; then command -v python3; return 0; fi
  if command -v python >/dev/null 2>&1; then command -v python; return 0; fi
  for p in /usr/local/bin/python3 /usr/bin/python3; do
    [ -x "$p" ] && { echo "$p"; return 0; }
  done
  return 1
}
if ! PYBIN=$(find_python); then
  cat >&2 <<EOF

ERROR: no python (>= 3.10) found on this machine.

Fix: run the environment setup first, then re-run this script (it will
find the venv automatically):
    bash ../setup/setup_5090_novita.sh          # installs miniforge + venv

Or point this script at an existing venv:
    WHY5_VENV=/path/to/venv ./run_benchmark.sh
EOF
  exit 1
fi
echo "using python: $PYBIN"
case "$PYBIN" in
  */venv/bin/python) : ;;
  *) echo "note: using SYSTEM python (not the setup venv) — if imports fail," \
     "run: bash ../setup/setup_5090_novita.sh" ;;
esac

# ---- phase 1: harness self-test — perfect mock student must pass everything
banner "phase 1: self-test (perfect mock student)"
"$PYBIN" why5_bench.py --mock --n "$N" --curve-n 0 \
  --out benchmark_selftest.json
if [ $? -eq 0 ]; then
  echo "phase 1 OK: perfect student scored above all gates"
else
  echo "phase 1 FAILED: perfect student did not pass — the HARNESS is broken" >&2
  RC=1
fi

# ---- phase 2: detection test — flawed mock student must FAIL (and be caught)
banner "phase 2: detection test (flawed mock student)"
"$PYBIN" why5_bench.py --mock --inject-flaws --n "$N" --curve-n 0 \
  --out benchmark_flawed.json
if [ $? -ne 0 ]; then
  echo "phase 2 OK: flawed student correctly detected below gates"
else
  echo "phase 2 FAILED: flawed student passed the gates — harness cannot detect errors" >&2
  RC=1
fi

# ---- phase 3: live benchmark against a served student
if [ "$MOCK" != "1" ]; then
  banner "phase 3: live benchmark ($BENCH_URL)"
  if ! curl -sf -o /dev/null "$BENCH_URL/models"; then
    echo "endpoint $BENCH_URL not reachable — start it with:" >&2
    echo "  $PYBIN -m vllm.entrypoints.openai.api_server --model $BENCH_MODEL --port 8001" >&2
    RC=1
  else
    "$PYBIN" why5_bench.py --url "$BENCH_URL" --model "$BENCH_MODEL" \
      --n "$N" --curve-n "$CURVE_N" --out benchmark_report.json
    [ $? -eq 0 ] || RC=1
  fi
else
  banner "phase 3: skipped (MOCK=1)"
fi

banner "result"
if [ "$RC" -eq 0 ]; then
  echo "ALL PHASE EXPECTATIONS MET"
else
  echo "BENCHMARK FAILED — see reports above" >&2
fi
exit "$RC"
