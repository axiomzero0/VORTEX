# VORTEX vs CPython 3.16 — Head-to-Head Benchmark

This directory holds the harness, workloads, and results for comparing VORTEX
against CPython 3.16 (built from `cpython/main` at `Python 3.16.0a0`).

See `REPORT.md` for the full analysis. Quick summary:

- 30 iterations per workload, both sides
- VORTEX run in-process via `vortex_bench` (fresh Vm per iter)
- CPython run via fresh subprocess per iter (`./python script.py`)
- 10 workloads matching `bench_cases.hpp` exactly

## Reproduce

```bash
# 1. Build CPython 3.16 from main
git clone --depth 1 https://github.com/python/cpython.git /path/to/cpython
cd /path/to/cpython && ./configure && make -j2

# 2. Run CPython bench
python3 run_cpython_bench.py \
    --python /path/to/cpython/python \
    --workloads workloads \
    --iterations 30 \
    --out results/cpython_results.csv

# 3. Run VORTEX bench (from VORTEX repo root)
cd ../.. && ./build/benchmarks/vortex_bench --iterations 30 \
    > benchmarks/vs_cpython/results/vortex_results.csv

# 4. Combine
python3 combine_bench_results.py \
    results/vortex_results.csv \
    results/cpython_results.csv \
    results/combined_results.csv
```
