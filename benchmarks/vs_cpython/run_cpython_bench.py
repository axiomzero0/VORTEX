#!/usr/bin/env python3
"""
CPython 3.16 reference benchmark harness.

Mirrors the VORTEX bench_main.cpp protocol:
- For each workload, run the .py file in a fresh CPython subprocess
  N iterations, measure wall-clock per iteration, report
  median/mean/p99 in nanoseconds.
- CSV columns match VORTEX exactly: name,iterations,median_ns,mean_ns,p99_ns,correctness
- "fresh subprocess per iter" mirrors VORTEX's "fresh Vm per iter" —
  end-to-end wall-clock measurement of "time to get the answer".

Usage:
    python3 run_cpython_bench.py --python /path/to/python3.16 \\
        --workloads /path/to/workloads --iterations 30 \\
        --out /path/to/cpython_results.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


# (name, expected_stdout) — must stay in lockstep with bench_cases.hpp
WORKLOADS = [
    ("int_arith",          "49995000\n"),
    ("float_arith",        "15000.0\n"),
    ("fib_recursion",      "6765\n"),
    ("loop_sum",           "4999950000\n"),
    ("loop_branch",        "5000 5000\n"),
    ("list_build",         "49995000\n"),
    ("list_concat_string", "0123456789\n"),
    ("nested_loops",       "10000\n"),
    ("mixed_int_float",    "500500\nTrue\n"),
    ("ackermann",          "9\n"),
]


def measure_once(python_bin: str, script: Path) -> tuple[int, str]:
    """Run the script once in a fresh subprocess; return (elapsed_ns, stdout)."""
    t0 = time.perf_counter_ns()
    proc = subprocess.run(
        [python_bin, str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    t1 = time.perf_counter_ns()
    # decode stdout (CPython's print() emits \n line endings on Linux)
    out = proc.stdout.decode("utf-8", errors="replace")
    return (t1 - t0, out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", required=True, help="path to python3.16 binary")
    ap.add_argument("--workloads", required=True, help="dir containing .py files")
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--out", required=True, help="CSV output path")
    ap.add_argument("--filter", default=None, help="only run matching workload name")
    args = ap.parse_args()

    python_bin = args.python
    if not os.path.isfile(python_bin) or not os.access(python_bin, os.X_OK):
        sys.stderr.write(f"error: {python_bin} not executable\n")
        return 2

    workloads_dir = Path(args.workloads)
    if not workloads_dir.is_dir():
        sys.stderr.write(f"error: {workloads_dir} not a directory\n")
        return 2

    # Print the python version banner to stderr for traceability
    ver_proc = subprocess.run(
        [python_bin, "--version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    sys.stderr.write(f"[cpython bench] binary: {python_bin}\n")
    sys.stderr.write(f"[cpython bench] version: {ver_proc.stdout.decode().strip()}\n")
    sys.stderr.write(f"[cpython bench] iterations: {args.iterations}\n")

    rows = []
    for name, expected in WORKLOADS:
        if args.filter and args.filter != name:
            continue
        script = workloads_dir / f"{name}.py"
        if not script.is_file():
            sys.stderr.write(f"warn: {script} missing, skipping {name}\n")
            continue

        samples_ns = []
        correctness = "ok"
        for _ in range(args.iterations):
            elapsed_ns, out = measure_once(python_bin, script)
            samples_ns.append(elapsed_ns)
            if correctness == "ok" and out != expected:
                correctness = f"FAIL(got={out!r} expected={expected!r})"

        samples_ns.sort()
        median_ns = int(statistics.median(samples_ns))
        mean_ns = int(statistics.mean(samples_ns))
        p99_idx = max(0, int(len(samples_ns) * 0.99) - 1)
        p99_ns = int(samples_ns[p99_idx])

        rows.append({
            "name": name,
            "iterations": args.iterations,
            "median_ns": median_ns,
            "mean_ns": mean_ns,
            "p99_ns": p99_ns,
            "correctness": correctness,
        })
        sys.stderr.write(
            f"[cpython bench] {name:20s} median={median_ns:>10d}ns "
            f"mean={mean_ns:>10d}ns p99={p99_ns:>10d}ns {correctness}\n"
        )

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "iterations", "median_ns",
                                          "mean_ns", "p99_ns", "correctness"])
        w.writeheader()
        w.writerows(rows)
    sys.stderr.write(f"[cpython bench] wrote {args.out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
