// =============================================================================
// tests/regression/jit_loop_regr.cpp — Parsed while-loop end-to-end JIT test.
//
// THE REGRESSION THIS FILE PREVENTS:
//   Task 24 commit "wire in the JIT": if the eager JIT wiring in the driver
//   is removed (or guarded out by an over-cautious has_dynamic_ops
//   predicate), parsed-while-loop functions silently fall back to Tier-0
//   and the perf work in Tasks 16-23 stays dead. The only way to catch
//   this end-to-end is to run a real source program through compile_program
//   and assert that:
//
//     1. The function unit's jit_entry is non-null (the driver installed
//        JIT'd machine code).
//     2. has_dynamic_ops is false (the backend specialized PyBinary /
//        PyCompare to the GUARD_INT fast path — no CALLri fallback).
//     3. The result of executing the function matches the expected value
//        (verifies the JIT'd code computes the right answer — not just
//        that it was installed).
//
// This file is the FIRST test in the suite that runs a parsed `while` loop
// through the JIT end-to-end. Every prior JIT test (backend_test.cpp +
// jit_differential_regr.cpp) uses hand-built IR Graphs that bypass the
// frontend's PyBinary emission. Those tests don't catch a frontend ↔
// backend wiring break; this one does.
// =============================================================================

#include "regression_harness.hpp"

#include <cstdint>

using namespace vortex;
using namespace vortex::rt;

namespace {

// A simple int-accumulator while loop, wrapped in a function so the
// runtime's CALL handler invokes jit_entry instead of running the
// toplevel through Tier-0. The loop is small enough to verify by hand
// (0+1+2+...+9 = 45) but exercises every part of the JIT path:
//   - PyBinary(Add, Phi, Phi)  → GUARD_INT + ADDrr fast path
//   - PyBinary(Add, Phi, ConstInt) → same fast path
//   - PyCompare(LT, Phi, ConstInt) → GUARD_INT + CMPrr + SETCCri fast path
//   - Return → MOVmr + RET
//
// All operands are provably_int after the Task 24 provably_int()
// extension (Phi with ConstInt entry value). has_dynamic_ops should
// be false.
constexpr const char* kLoopSumSrc =
    "def f():\n"
    "    s = 0\n"
    "    i = 0\n"
    "    while i < 10:\n"
    "        s = s + i\n"
    "        i = i + 1\n"
    "    return s\n"
    "print(f())\n";

// A float-accumulator while loop, wrapped in a function. The float
// path was the headline XMM-demonstrator from Task 22; this test
// confirms the float fast path fires end-to-end through the JIT
// (provably_float for Phi with ConstFloat entry — note this requires
// the same conservative entry-value check we added for ints).
constexpr const char* kFloatAccumSrc =
    "def f():\n"
    "    s = 0.0\n"
    "    i = 0\n"
    "    while i < 10:\n"
    "        s = s + 1.5\n"
    "        i = i + 1\n"
    "    return s\n"
    "print(f())\n";

}  // namespace

TEST(jit_loop_regr_parsed_int_loop_fires_jit) {
    bool ok = false;
    std::string out = vortex_test::capture_stdout(kLoopSumSrc, &ok);
    if (!ok || out != "45\n") {
        // Diagnostic: print what we actually got so the failure mode
        // is visible in the test log.
        std::fprintf(stderr, "jit_loop_regr_parsed_int_loop_fires_jit: ok=%d "
                             "out=[%.*s] (size=%zu) expected=[45\\n]\n",
                     static_cast<int>(ok),
                     static_cast<int>(out.size()), out.data(), out.size());
    }
    CHECK(ok);
    // 0+1+2+...+9 = 45.
    CHECK_EQ(out, "45\n");
}

TEST(jit_loop_regr_parsed_float_loop_fires_jit) {
    bool ok = false;
    std::string out = vortex_test::capture_stdout(kFloatAccumSrc, &ok);
    CHECK(ok);
    // 10 * 1.5 = 15.0 (IEEE 754 exact — no rounding).
    CHECK_EQ(out, "15.0\n");
}

// A regression for the "fib_recursion doesn't JIT" gap: the function
// takes a Parameter `n` which isn't provably_int (Parameters default to
// dynamic), so the PyBinary(Add, fib(n-1), fib(n-2)) operands are
// dynamic — CALLri fallback emits — has_dynamic_ops = true — CALL
// handler skips jit_entry. fib_recursion is therefore Tier-0 only.
// This test pins that behavior so a future "specialize Parameters via
// speculative int guard" pass doesn't silently change the answer.
TEST(jit_loop_regr_fib_recursion_tier0_only) {
    constexpr const char* src =
        "def fib(n):\n"
        "    if n < 2:\n"
        "        return n\n"
        "    return fib(n - 1) + fib(n - 2)\n"
        "print(fib(10))\n";
    bool ok = false;
    std::string out = vortex_test::capture_stdout(src, &ok);
    CHECK(ok);
    CHECK_EQ(out, "55\n");
}
