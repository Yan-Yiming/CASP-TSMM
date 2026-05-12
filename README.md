# CASP-TSMM

Tall-Skinny Matrix Multiplication benchmark and optimization project.

The project evaluates:

```text
C = A^T * B
A: k x m, B: k x n, C: m x n, double precision
```

Both row-major and column-major storage are supported by the benchmark. The local and Slurm runners execute both layouts for the four required tasks by default.

## Project Layout

```text
CASP-TSMM/
|-- src/
|   |-- benchmark.cpp       # benchmark harness, timing, correctness, JSON output
|   |-- tsmm.hpp            # shared kernel interface and layout helpers
|   |-- tsmm_registry.cpp   # automatic kernel registry
|   |-- reference.cpp       # MKL/OpenBLAS dgemm reference, or built-in fallback
|   `-- tsmm/
|       |-- naive.cpp           # serial baseline
|       |-- openmp_kernel.cpp   # OpenMP parallel baseline
|       |-- blocked.cpp         # cache-blocked OpenMP kernel
|       |-- avx512.cpp          # AVX-512 kernels
|       `-- opt.cpp             # combined optimized kernel
|-- scripts/
|   |-- collect_gflops.py   # collect all GFLOPS rows into CSV/JSON
|   |-- run_local.sh        # local row/col benchmark and dashboard runner
|   |-- run_local.ps1       # Windows/MSVC local row/col runner
|   `-- submit_slurm.sh     # Slurm submission helper for the target cluster
|-- web/
|   |-- index.html          # live dashboard
|   `-- server.py           # lightweight Python server, no Docker
`-- Makefile
```

Generated files such as `benchmark`, `benchmark.exe`, `web/results/`, and `logs/` are not source files.
Build intermediates and executables are placed under `obj/`.

## Implementations

| Name | Description |
| --- | --- |
| `reference` | CBLAS `dgemm` reference using MKL/OpenBLAS, or a built-in fallback with `BLAS=none` |
| `naive` | serial three-loop TSMM |
| `openmp` | OpenMP parallelization over rows of C |
| `blocked` | cache-blocked OpenMP kernel, tunable with `TSMM_IB/JB/LB` |
| `avx512` | row-major AVX-512 vectorized kernel |
| `avx512_omp` | AVX-512 plus OpenMP |
| `opt` | best-effort combined kernel with special handling for small `m` |

## Build And Run

```bash
# One-key local run: required tasks, row-major and col-major
bash scripts/run_local.sh --required-only

# Optional: all tasks, row-major and col-major
bash scripts/run_local.sh --all
```

On this Windows workspace, use the PowerShell runner:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_local.ps1
```

```bash
# Manual smoke test. If --output is omitted, the file name includes layout and timestamp.
make BLAS=none
./obj/benchmark --required-only --layout row --warmup 1 --runs 1
```

Benchmark options:

```text
--required-only       run only the four required problems
--all                 run required and optional problems
--layout row|col      choose row-major or column-major storage
--output-dir DIR      write results_<layout>_<timestamp>.json into DIR
--output PATH         explicit output path, bypassing timestamp naming
--warmup N            warmup iterations, default 10
--runs N              timed iterations, default 20
--no-correctness      skip comparison with the reference result
```

For very large problems the harness automatically caps warmup to 3 and timed runs to 5 to keep evaluation practical.

## Dashboard

```bash
make web
# open http://localhost:8080
```

Or run benchmark and dashboard together:

```bash
bash scripts/run_local.sh --required-only
```

Each run writes timestamped JSON files under `web/results/<run-id>/`, plus `gflops.csv` and `gflops_summary.json`. The dashboard reads the newest result file it can find.

## Adding A Kernel

Add a new `.cpp` file under `src/tsmm/`, implement a function matching `TsmmKernel`, then register it:

```cpp
#include "../tsmm.hpp"

void tsmm_my_kernel(int m, int n, int k, const double* A, const double* B, double* C, Layout layout) {
    // compute C = A^T * B
}

REGISTER_TSMM_IMPL("my_kernel", tsmm_my_kernel);
```

The Makefile compiles `src/tsmm/*.cpp`, and the benchmark discovers registered kernels automatically.

## Slurm Target Run

The target CPU is Intel Xeon Platinum 9242 with 96 cores, 4 NUMA nodes, and AVX-512. The submission helper requests all 96 CPUs and runs both row-major and col-major required tasks by default:

```bash
bash scripts/submit_slurm.sh              # required problems
bash scripts/submit_slurm.sh --all        # required + optional
bash scripts/submit_slurm.sh --dry-run    # print the generated job script
```

The job uses `numactl --cpunodebind=0-3 --interleave=all`, `OMP_PROC_BIND=spread`, and `OMP_PLACES=cores`. Because the benchmark initializes matrices serially, interleaving pages across all four NUMA nodes is the most robust placement without changing measurement code; it avoids putting all pages on one NUMA node before the 96 OpenMP threads start computing.

Useful environment overrides:

```bash
PARTITION=cpu CPUS_PER_TASK=96 NUMA_NODES=0-3 BLAS=mkl bash scripts/submit_slurm.sh
```

The final metric is GFLOPS per task and speedup versus the `reference` `dgemm` baseline. The dashboard also reports the geometric mean speedup across the required problems.
