#!/usr/bin/env python3
"""
Combine VORTEX and CPython benchmark CSVs into a side-by-side comparison.
Outputs both a combined CSV and a Markdown table to stdout.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path


def load_csv(path: Path) -> dict[str, dict[str, int]]:
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[row["name"]] = {
                "median_ns": int(row["median_ns"]),
                "mean_ns": int(row["mean_ns"]),
                "p99_ns": int(row["p99_ns"]),
                "correctness": row["correctness"],
            }
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        sys.stderr.write(
            f"usage: {argv[0]} <vortex.csv> <cpython.csv> <out.csv>\n"
        )
        return 2
    vortex_path, cpython_path, out_path = argv[1], argv[2], argv[3]
    vortex = load_csv(Path(vortex_path))
    cpython = load_csv(Path(cpython_path))

    # workload order matches bench_cases.hpp
    order = [
        "int_arith",
        "float_arith",
        "fib_recursion",
        "loop_sum",
        "loop_branch",
        "list_build",
        "list_concat_string",
        "nested_loops",
        "mixed_int_float",
        "ackermann",
    ]

    rows = []
    for name in order:
        v = vortex.get(name, {})
        c = cpython.get(name, {})
        v_med = v.get("median_ns", 0)
        c_med = c.get("median_ns", 0)
        speedup = c_med / v_med if v_med else 0.0
        verdict = "VORTEX" if v_med < c_med else "CPython"
        rows.append({
            "name": name,
            "vortex_median_ns": v_med,
            "vortex_mean_ns": v.get("mean_ns", 0),
            "vortex_p99_ns": v.get("p99_ns", 0),
            "vortex_correctness": v.get("correctness", ""),
            "cpython_median_ns": c_med,
            "cpython_mean_ns": c.get("mean_ns", 0),
            "cpython_p99_ns": c.get("p99_ns", 0),
            "cpython_correctness": c.get("correctness", ""),
            "vortex_vs_cpython_speedup_x": round(speedup, 3),
            "winner": verdict,
        })

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    sys.stderr.write(f"wrote {out_path}\n")

    # markdown table to stdout
    print("| workload | VORTEX median ns | CPython 3.16 median ns | VORTEX vs CPython | winner |")
    print("|---|---:|---:|---:|:---|")
    for r in rows:
        print(
            f"| {r['name']} | {r['vortex_median_ns']:,} | "
            f"{r['cpython_median_ns']:,} | "
            f"{r['vortex_vs_cpython_speedup_x']}x | {r['winner']} |"
        )

    # summary
    vortex_wins = sum(1 for r in rows if r["winner"] == "VORTEX")
    cpython_wins = sum(1 for r in rows if r["winner"] == "CPython")
    print()
    print(f"VORTEX wins {vortex_wins}/{len(rows)}; CPython 3.16 wins {cpython_wins}/{len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
