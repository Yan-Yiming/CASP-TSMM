# TSMM Work Log

## 2026-06-18 18:14:35 CST

### 改动

- 将 `src/tsmm/opt.cpp` 从初级通用实现改为按 required 规模分派的优化版本。
- 新增 row-major 特化路径：
  - `(4000,16000,128)`：`IB=4, JB=40`，AVX-512，C 使用 non-temporal store。
  - `(8,16,16000)`：沿 k 维切分，最多 8 线程 partial reduction。
  - `(32,16000,16)`：`IB=4, JB=32`，限制最多 48 线程。
  - `(144,144,144)`：`IB=9, JB=24`，限制最多 32 线程。
- 新增 col-major 特化路径：
  - `(4000,16000,128)`：pack A 到 `Apack[l*M+i]`，`IB=16, JB=8`，C 使用 non-temporal store。
  - `(32,16000,16)`：将小 A 转置到 aligned buffer，一次处理 4 列 B。
  - `(144,144,144)`：一次处理 2 列 B，限制最多 32 线程。
- 修复旧版 `row_tiny_output_large_k` 非 OpenMP fallback 中 `tmp[0].data()` 的可编译性风险，新实现不再使用该错误路径。

### 本地验证

- 执行：
  ```bash
  make clean
  make BLAS=none -j2
  ```
  结果：编译通过。

- 执行：
  ```bash
  make clean
  make BLAS=openblas -j2
  ```
  结果：当前本地环境缺少 `cblas.h`，OpenBLAS 编译失败，不能在本地用 BLAS reference 跑完整 benchmark。

- 执行：
  ```bash
  make clean
  make BLAS=none AVX512=1 -j2
  ```
  结果：AVX-512 分支编译通过。

### 当前结论

- 代码已完成第一轮 required 规模特化，能够通过本地 `BLAS=none` 和 `AVX512=1` 编译检查。
- 因为本地没有 OpenBLAS 头文件，且 `BLAS=none` 的大矩阵 reference 是串行 fallback，不适合在本地跑完整 required 正确性/性能。
- 下一步必须在目标集群上用 MKL 运行，确认 correctness 和实际 speedup。

### 下一步命令

在目标集群登录节点编译：

```bash
module purge
module load gcc/10.2.0
module load intel/2022.1
module load python/3.8.6
make clean
make CXX=icpc BLAS=mkl AVX512=1 -j16
```

提交 required：

```bash
bash scripts/submit_slurm.sh
```

收到结果后记录：

- `logs/tsmm_<job_id>.out`
- `web/results/<run-id>/gflops.csv`
- `web/results/<run-id>/gflops_summary.json`

重点比较：

- required geomean speedup 是否提升。
- `(4000,16000,128)` 的 row/col 是否正确且是否受益于 streaming store。
- `(8,16,16000)` 是否因限制 8 线程而优于全 96 线程。
- `(144,144,144)` 是否因限制 32 线程而降低 OpenMP 开销。

## 2026-06-19 16:35:34 CST

### 改动

- 按工作要求先阅读 `before_work.txt`，并重新阅读 `project_target.md`、`plan.txt`、`README.md`、`work_log.md`。
- 通读当前评测框架、参考实现、Slurm 脚本、结果收集脚本、Web 页面和 `src/tsmm/opt.cpp`。
- 确认当前主线源码只编译 `src/*.cpp` 和 `src/tsmm/*.cpp`；根目录 `ref.cpp` 是实验候选/参考代码，不参与默认构建。
- 本次未修改核心算法代码，只追加本工作日志。

### 本地验证

- 执行：
  ```bash
  make clean
  make BLAS=none -j2
  ```
  结果：编译通过。

- 执行：
  ```bash
  make clean
  make BLAS=none AVX512=1 -j2
  ```
  结果：AVX-512 分支编译通过。

- `make clean` 时出现一次 `Clock skew detected` warning，表现为 `obj/src/tsmm/opt.d` 时间戳比当前系统时间略超前；清理和后续编译均完成。

### 当前结论

- 当前 `opt.cpp` 已经是按 required 规模分派的第一轮优化版本，和 2026-06-18 日志记录一致。
- 本地环境仍然只能做 `BLAS=none` 编译检查；因为 fallback reference 是串行实现，不适合直接跑 required 大规模正确性/性能。
- 目标机 MKL + Slurm baseline 仍是下一步必要工作，只有拿到 row/col required 的实际 GFLOPS 和 speedup 后，才能判断各个特化 kernel 是否保留或调整。

### 下一步命令

在目标集群登录节点编译：

```bash
module purge
module load gcc/10.2.0
module load intel/2022.1
module load python/3.8.6
make clean
make CXX=icpc BLAS=mkl AVX512=1 -j16
```

提交 required baseline：

```bash
bash scripts/submit_slurm.sh
```

收到结果后优先检查：

- row/col 两个 JSON 中 `opt` 是否全部 `correct=true`。
- `web/results/<run-id>/gflops.csv` 中 required 八项的 `speedup`。
- 如果某项错误，先修 correctness；如果全部正确，再按 geomean 最差项继续优化。
