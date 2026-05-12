#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PARTITION="${SLURM_PARTITION:-cpu}"
CPUS_PER_TASK="${CPUS_PER_TASK:-96}"
NUMA_NODES="${NUMA_NODES:-0-3}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
MEM="${MEM:-64G}"
BLAS="${BLAS:-mkl}"

ALL_PROBLEMS=false
DRY_RUN=false
EXTRA_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --all) ALL_PROBLEMS=true ;;
        --required-only) ALL_PROBLEMS=false ;;
        --dry-run) DRY_RUN=true ;;
        *) EXTRA_ARGS+=("$arg") ;;
    esac
done

if [ "$ALL_PROBLEMS" = false ]; then
    MODE_ARGS="--required-only"
else
    MODE_ARGS="--all"
fi

EXTRA_ARGS_TEXT="${EXTRA_ARGS[*]}"

JOBSCRIPT=$(cat <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=tsmm_bench
#SBATCH --partition=$PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --time=$TIME_LIMIT
#SBATCH --mem=$MEM
#SBATCH --output=logs/tsmm_%j.out
#SBATCH --error=logs/tsmm_%j.err

set -euo pipefail
cd "\$SLURM_SUBMIT_DIR"
mkdir -p logs web/results

module load intel/2020 mkl/2020 gcc/10 || true

make BLAS=$BLAS AVX512=1 -j$CPUS_PER_TASK
BENCHMARK_BIN="./obj/benchmark"

export OMP_NUM_THREADS=$CPUS_PER_TASK
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export MKL_NUM_THREADS=$CPUS_PER_TASK
export MKL_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=$CPUS_PER_TASK
export KMP_AFFINITY=granularity=fine,compact,1,0

RUN_ID="\${SLURM_JOB_ID}_\$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="web/results/\$RUN_ID"
mkdir -p "\$RESULT_DIR"

echo "=== Job info ==="
echo "SLURM_JOB_ID=\$SLURM_JOB_ID"
echo "SLURM_NODELIST=\$SLURM_NODELIST"
echo "OMP_NUM_THREADS=\$OMP_NUM_THREADS"
echo "NUMA_NODES=$NUMA_NODES"
lscpu | grep -E 'Model name|Socket|Core|NUMA|MHz' || true
numactl --hardware || true

for layout in row col; do
    echo "=== Running layout: \$layout ==="
    numactl --cpunodebind=$NUMA_NODES --interleave=all \\
        "\$BENCHMARK_BIN" --output-dir "\$RESULT_DIR" $MODE_ARGS --layout "\$layout" $EXTRA_ARGS_TEXT
done

python3 scripts/collect_gflops.py "\$RESULT_DIR" \\
    --csv "\$RESULT_DIR/gflops.csv" \\
    --json "\$RESULT_DIR/gflops_summary.json"

echo "Result dir: \$RESULT_DIR"
echo "GFLOPS CSV: \$RESULT_DIR/gflops.csv"
echo "Summary JSON: \$RESULT_DIR/gflops_summary.json"
EOF
)

echo "=== Slurm job script ==="
echo "$JOBSCRIPT"
echo

if [ "$DRY_RUN" = true ]; then
    echo "Dry run only; not submitting."
    exit 0
fi

mkdir -p logs
TMP_SCRIPT="$(mktemp /tmp/tsmm_job_XXXXX.sh)"
printf '%s\n' "$JOBSCRIPT" > "$TMP_SCRIPT"
JOB_ID="$(sbatch "$TMP_SCRIPT" | awk '{print $NF}')"
rm -f "$TMP_SCRIPT"

echo "Submitted job $JOB_ID"
echo "Monitor: squeue -j $JOB_ID"
echo "Log: tail -f logs/tsmm_${JOB_ID}.out"
