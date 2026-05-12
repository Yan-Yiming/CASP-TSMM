#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BLAS="${BLAS:-openblas}"
THREADS="${OMP_NUM_THREADS:-$(nproc)}"
RESULT_ROOT="${RESULT_ROOT:-web/results}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RESULT_DIR="$RESULT_ROOT/$RUN_ID"

MODE_ARGS=(--required-only)
EXTRA_ARGS=()
START_WEB="${START_WEB:-1}"

for arg in "$@"; do
    case "$arg" in
        --all)
            MODE_ARGS=(--all)
            ;;
        --required-only)
            MODE_ARGS=(--required-only)
            ;;
        --no-web)
            START_WEB=0
            ;;
        *)
            EXTRA_ARGS+=("$arg")
            ;;
    esac
done

export OMP_NUM_THREADS="$THREADS"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$THREADS}"
export MKL_DYNAMIC="${MKL_DYNAMIC:-FALSE}"

echo "=== TSMM local row/col run ==="
echo "BLAS=$BLAS OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "Result dir: $RESULT_DIR"

make BLAS="$BLAS" -j"$(nproc)"
mkdir -p "$RESULT_DIR"
BENCHMARK_BIN="${BENCHMARK_BIN:-./obj/benchmark}"

WEB_PID=""
if [ "$START_WEB" = "1" ]; then
    python3 web/server.py --port 8080 --results "$RESULT_DIR" &
    WEB_PID=$!
    echo "Dashboard: http://localhost:8080"
fi

cleanup() {
    if [ -n "$WEB_PID" ]; then
        kill "$WEB_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

for layout in row col; do
    echo "=== Running layout: $layout ==="
    "$BENCHMARK_BIN" --output-dir "$RESULT_DIR" "${MODE_ARGS[@]}" --layout "$layout" "${EXTRA_ARGS[@]}"
done

python3 scripts/collect_gflops.py "$RESULT_DIR" \
    --csv "$RESULT_DIR/gflops.csv" \
    --json "$RESULT_DIR/gflops_summary.json"

echo "GFLOPS CSV: $RESULT_DIR/gflops.csv"
echo "Summary JSON: $RESULT_DIR/gflops_summary.json"
