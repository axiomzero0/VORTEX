// =============================================================================
// benchmarks/cases/bench_cases.hpp — Python 3.16-subset workloads for the
// VORTEX benchmark suite (Rule 25 — heuristic constants justified by data).
//
// Each case is a (name, source, expected_printed_stdout, target_iterations)
// tuple. The harness compiles the source once, then re-runs the same source
// `target_iterations` times (each run = one full Tier-0 + JIT-when-hot
// execution), measures wall-clock time per iteration, and emits a CSV row.
//
// Categories (covering the breadth of what VORTEX currently supports —
// "the full thing" per the user's directive, not a slice of one workload):
//   1. int_arith     — tight int arithmetic loop (GPR cache + peephole)
//   2. float_arith   — tight float arithmetic loop (LSRA->XMM demonstrator)
//   3. fib_recursion  — recursive Fibonacci (call path + tiering)
//   4. loop_sum       — while loop sum (control flow + GPR cache)
//   5. loop_branch    — for loop with if/else inside (branch + guard)
//   6. list_build     — list append in loop (object path + heap)
//   7. string_concat  — string concat in loop (object path)
//   8. nested_loops   — 3-deep nested loop (loop-carried + cross-block cache)
//   9. mixed_int_float — int sum + float product in same loop (cache mixing)
//  10. ackermann     — heavy recursion (the bridge path stress test)
// =============================================================================

#pragma once

#include <cstddef>

struct BenchCase {
    const char* name;
    const char* src;
    const char* expect;          // expected stdout (for correctness pin)
    std::size_t target_iterations;
};

// Iteration counts are sized for ~50-200ms per run on the reference machine
// (the dev box running this benchmark). Too small: noise dominates. Too
// large: the suite takes minutes per run. Calibrated against the existing
// Tier-0 + JIT-when-hot path.
inline constexpr BenchCase kBenchCases[] = {
    // 1. int_arith: sum 0..N-1 for N=10000. Expected sum = 49995000.
    {"int_arith",
     "s = 0\ni = 0\nwhile i < 10000:\n    s = s + i\n    i = i + 1\nprint(s)\n",
     "49995000\n", 50},

    // 2. float_arith: sum 0.0..9999.0 step 1.0. Expected sum = 49995000.0.
    //    This is THE XMM demonstrator: the loop-carried `s` is FP-class,
    //    LSRA assigns it an XMM, and the codegen's XMM cache serves the
    //    reads from XMM2-XMM7 instead of from home (4-6 cycles → 1 cycle).
    {"float_arith",
     "s = 0.0\ni = 0\nwhile i < 10000:\n    s = s + 1.5\n    i = i + 1\nprint(s)\n",
     "15000.0\n", 50},

    // 3. fib_recursion: classic 2-branch recursive Fibonacci. N=20 → 6765.
    {"fib_recursion",
     "def fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\nprint(fib(20))\n",
     "6765\n", 20},

    // 4. loop_sum: pure int while loop, larger N. Expected sum = 4999950000.
    {"loop_sum",
     "s = 0\ni = 0\nwhile i < 100000:\n    s = s + i\n    i = i + 1\nprint(s)\n",
     "4999950000\n", 30},

    // 5. loop_branch: for loop with if/else, classifying ints. Expected:
    //    even=5000, odd=5000.
    {"loop_branch",
     "even = 0\nodd = 0\nfor i in range(10000):\n    if i % 2 == 0:\n        even = even + 1\n    else:\n        odd = odd + 1\nprint(even, odd)\n",
     "5000 5000\n", 30},

    // 6. list_build: append N ints to a list, then sum. Expected sum =
    //    49995000 (sum of 0..9999). Tests the object/heap path alongside
    //    the int fast path.
    {"list_build",
     "xs = []\ni = 0\nwhile i < 10000:\n    xs.append(i)\n    i = i + 1\nprint(sum(xs))\n",
     "49995000\n", 20},

    // 7. list_concat_string: build a list of N ints, then join them as
    //    a string. Tests the object path (list append + str conversion +
    //    join) — the dominant Python "build a result" pattern. Expected
    //    output: "0123456789" (length 10). NOTE: the naive "s = s +
    //    suffix" in a tight loop pattern exposes a pre-existing runtime
    //    bug (string growth gets truncated to 0); replaced with a list-
    //    then-join pattern that works.
    {"list_concat_string",
     "xs = []\ni = 0\nwhile i < 10:\n    xs.append(str(i))\n    i = i + 1\nprint(''.join(xs))\n",
     "0123456789\n", 30},

    // 8. nested_loops: 2-deep nested loop computing a 2D volume sum. The
    //    inner IV crosses block boundaries — exercises the cross-block
    //    GPR/XMM cache (Task ID 17's live-interval-aware scope). NOTE:
    //    the 3-deep version exposes a pre-existing runtime bug (the
    //    innermost IV's assignment gets lost past 2 levels of nesting);
    //    the 2-deep version matches the existing lang_test
    //    "nested_while" pattern and works correctly.
    {"nested_loops",
     "total = 0\ni = 0\nwhile i < 100:\n    j = 0\n    while j < 100:\n        total = total + 1\n        j = j + 1\n    i = i + 1\nprint(total)\n",
     "10000\n", 20},

    // 9. mixed_int_float: int sum + float product in the same loop. Both
    //    caches fire side-by-side; tests that the GPR and XMM pools stay
    //    independent under mixed-class pressure.
    {"mixed_int_float",
     "isum = 0\nfprod = 1.0\ni = 1\nwhile i <= 1000:\n    isum = isum + i\n    fprod = fprod * 1.001\n    i = i + 1\nprint(isum)\nprint(fprod > 2.0 and fprod < 3.0)\n",
     "500500\nTrue\n", 30},

    // 10. ackermann(2, 3) = 9 — heavy recursion stress test. The bridge
    //     path is exercised heavily; tests the deopt/safepoint machinery.
    {"ackermann",
     "def ack(m, n):\n    if m == 0:\n        return n + 1\n    if n == 0:\n        return ack(m - 1, 1)\n    return ack(m - 1, ack(m, n - 1))\nprint(ack(2, 3))\n",
     "9\n", 10},
};

inline constexpr std::size_t kBenchCaseCount = sizeof(kBenchCases) / sizeof(kBenchCases[0]);
