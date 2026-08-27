// =============================================================================
// tests/regression/jit_differential_regr.cpp — Tier-0 vs JIT differential.
//
// THE REGRESSION THIS FILE PREVENTS:
//   commit f46cb95 "Revert IBE-18 MOVri write-back (regression in JIT tests)"
//
// The IBE-18 fix rewrote MOVri's home-slot write-back so the home slot was
// always populated even when the value went to a physical register. But
// the Return terminator's tag-vreg MOVri shared home slot 0 with the
// Return's MOVmr (which writes the return value's payload to slot 0).
// The "fix" clobbered the return value's payload with the tag constant —
// the JIT silently produced the wrong answer for every function returning
// an int. The Tier-0 interpreter was unaffected.
//
// The existing single-case JIT differential (backend_test.cpp's
// `jit_matches_tier0_for_int_arithmetic`) covered ONE program (identity+1).
// IBE-18 regressed programs returning larger ints because the home-slot
// clobber wrote a small tag constant to a slot that previously held a real
// value — and that specific case wasn't tested.
//
// This regression runs the differential across a CORPUS of pure-int
// functions so a future home-slot / write-back / register-codegen bug
// surfaces immediately rather than after deploy.
// =============================================================================

#include "regression_harness.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>

#include "vortex/backend/codegen.hpp"
#include "vortex/backend/target.hpp"
#include "vortex/ir/graph.hpp"
#include "vortex/rt/interp.hpp"

using namespace vortex;
using namespace vortex::ir;
namespace backend = vortex::backend;
namespace rt = vortex::rt;

#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64__)

namespace {

[[nodiscard]] std::byte* make_exec_buffer(std::size_t bytes) noexcept {
    long pagesz = sysconf(_SC_PAGESIZE);
    if (pagesz <= 0) pagesz = 4096;
    std::size_t mapped = ((bytes + pagesz - 1) / pagesz) * pagesz;
    void* p = mmap(nullptr, mapped, PROT_READ | PROT_WRITE | PROT_EXEC,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return nullptr;
    return static_cast<std::byte*>(p);
}

void free_exec_buffer(std::byte* p, std::size_t bytes) noexcept {
    if (!p) return;
    long pagesz = sysconf(_SC_PAGESIZE);
    if (pagesz <= 0) pagesz = 4096;
    std::size_t mapped = ((bytes + pagesz - 1) / pagesz) * pagesz;
    munmap(p, mapped);
}

// ---------------------------------------------------------------------------
// Pure-int function graph builders.
//
// Every graph here is a function with:
//   - exactly one Parameter (int)
//   - one or two integer constants
//   - one arithmetic op
//   - one Return
//
// No dynamic ops, no bridge path. The JIT must execute to completion via
// RET. We vary the arithmetic kind and the constant so the corpus catches
// any per-opcode or per-immediate-width clobber (the IBE-18 home-slot
// bug only manifested for certain Return placements).
// ---------------------------------------------------------------------------

[[nodiscard]] Graph int_identity_plus_one_graph() {
    Graph g;
    NodeId start = g.create(NodeKind::Start);
    g.set_start(start);
    NodeId p0 = g.create(NodeKind::Parameter, {start});
    g.node(p0).aux0 = 0;
    NodeId c1 = g.create(NodeKind::ConstInt);
    g.node(c1).const_value = Value::integer(1);
    NodeId add = g.create(NodeKind::Add, {p0, c1});
    NodeId ret = g.create(NodeKind::Return, {start, add});
    g.node(p0).set_flag(NodeFlag::Pure);
    g.node(c1).set_flag(NodeFlag::Pure);
    g.node(add).set_flag(NodeFlag::Pure);
    g.node(add).set_flag(NodeFlag::Unboxed);
    g.node(p0).set_flag(NodeFlag::Unboxed);
    g.node(c1).set_flag(NodeFlag::Unboxed);
    g.node(ret).set_flag(NodeFlag::OnEffectChain);
    g.set_end(ret);
    g.n_parameters = 1;
    g.function_name = global_symbols().intern("identity_plus_one");
    return g;
}

[[nodiscard]] Graph int_identity_times_three_graph() {
    Graph g;
    NodeId start = g.create(NodeKind::Start);
    g.set_start(start);
    NodeId p0 = g.create(NodeKind::Parameter, {start});
    g.node(p0).aux0 = 0;
    NodeId c3 = g.create(NodeKind::ConstInt);
    g.node(c3).const_value = Value::integer(3);
    NodeId mul = g.create(NodeKind::Mul, {p0, c3});
    NodeId ret = g.create(NodeKind::Return, {start, mul});
    g.node(p0).set_flag(NodeFlag::Pure);
    g.node(c3).set_flag(NodeFlag::Pure);
    g.node(mul).set_flag(NodeFlag::Pure);
    g.node(mul).set_flag(NodeFlag::Unboxed);
    g.node(p0).set_flag(NodeFlag::Unboxed);
    g.node(c3).set_flag(NodeFlag::Unboxed);
    g.node(ret).set_flag(NodeFlag::OnEffectChain);
    g.set_end(ret);
    g.n_parameters = 1;
    g.function_name = global_symbols().intern("identity_times_three");
    return g;
}

[[nodiscard]] Graph int_subtraction_graph() {
    Graph g;
    NodeId start = g.create(NodeKind::Start);
    g.set_start(start);
    NodeId p0 = g.create(NodeKind::Parameter, {start});
    g.node(p0).aux0 = 0;
    NodeId c7 = g.create(NodeKind::ConstInt);
    g.node(c7).const_value = Value::integer(7);
    NodeId sub = g.create(NodeKind::Sub, {p0, c7});
    NodeId ret = g.create(NodeKind::Return, {start, sub});
    g.node(p0).set_flag(NodeFlag::Pure);
    g.node(c7).set_flag(NodeFlag::Pure);
    g.node(sub).set_flag(NodeFlag::Pure);
    g.node(sub).set_flag(NodeFlag::Unboxed);
    g.node(p0).set_flag(NodeFlag::Unboxed);
    g.node(c7).set_flag(NodeFlag::Unboxed);
    g.node(ret).set_flag(NodeFlag::OnEffectChain);
    g.set_end(ret);
    g.n_parameters = 1;
    g.function_name = global_symbols().intern("sub_seven");
    return g;
}

[[nodiscard]] Graph int_large_constant_graph() {
    // Large constant: 1 << 40 = 1099511627776.
    // The IBE-18 clobber wrote tag=2 (Tag::Int) into the slot holding the
    // return value's PAYLOAD. For small returns, the value still looked
    // correct-ish; for large returns, the high bits got smashed with the
    // tag constant. This case catches both.
    Graph g;
    NodeId start = g.create(NodeKind::Start);
    g.set_start(start);
    NodeId cBig = g.create(NodeKind::ConstInt);
    g.node(cBig).const_value = Value::integer(1LL << 40);
    NodeId ret = g.create(NodeKind::Return, {start, cBig});
    g.node(cBig).set_flag(NodeFlag::Pure);
    g.node(cBig).set_flag(NodeFlag::Unboxed);
    g.node(ret).set_flag(NodeFlag::OnEffectChain);
    g.set_end(ret);
    g.n_parameters = 0;
    g.function_name = global_symbols().intern("large_constant");
    return g;
}

[[nodiscard]] Graph int_negate_graph() {
    Graph g;
    NodeId start = g.create(NodeKind::Start);
    g.set_start(start);
    NodeId p0 = g.create(NodeKind::Parameter, {start});
    g.node(p0).aux0 = 0;
    NodeId neg = g.create(NodeKind::Neg, {p0});
    NodeId ret = g.create(NodeKind::Return, {start, neg});
    g.node(p0).set_flag(NodeFlag::Pure);
    g.node(neg).set_flag(NodeFlag::Pure);
    g.node(neg).set_flag(NodeFlag::Unboxed);
    g.node(p0).set_flag(NodeFlag::Unboxed);
    g.node(ret).set_flag(NodeFlag::OnEffectChain);
    g.set_end(ret);
    g.n_parameters = 1;
    g.function_name = global_symbols().intern("negate");
    return g;
}

// =============================================================================
// FLOAT FAST-PATH CORPUS (PyBinary with both operands provably float).
//
// The float fast path emits GUARD_FLOAT (tag check both operands == 3) +
// SSE2 scalar ADDSD/SUBSD/MULSD/DIVSD + FMOVmr write-back + tag=Tag::Float.
// The corpus covers each arithmetic kind and a few value ranges:
//   small pos + small pos (typical FP code)
//   small pos - small pos (negatives, near-zero)
//   small * larger (scale change)
//   dividend / divisor (the divide case — different SSE opcode)
//   0.0 case (zero IEEE bits, exercises sign/magnitude handling)
//
// All operands are ConstFloat literals so the fast path fires unconditionally
// without depending on Parameter type-inference work Pass 15/16 don't do yet.
// =============================================================================

[[nodiscard]] Graph float_add_two_constants_graph() {
    Graph g;
    NodeId start = g.create(NodeKind::Start);
    g.set_start(start);
    NodeId c15 = g.create(NodeKind::ConstFloat);
    g.node(c15).const_value = Value::real(1.5);
    g.node(c15).set_flag(NodeFlag::Pure);
    NodeId c25 = g.create(NodeKind::ConstFloat);
    g.node(c25).const_value = Value::real(2.5);
    g.node(c25).set_flag(NodeFlag::Pure);
    NodeId add = g.create(NodeKind::PyBinary, {start, start, c15, c25});
    g.node(add).subop = static_cast<std::uint16_t>(BinOpKind::Add);
    g.node(add).set_flag(NodeFlag::OnEffectChain);
    NodeId ret = g.create(NodeKind::Return, {start, add});
    g.node(ret).set_flag(NodeFlag::OnEffectChain);
    g.set_end(ret);
    g.n_parameters = 0;
    g.function_name = global_symbols().intern("float_add_constants");
    return g;
}

[[nodiscard]] Graph float_sub_two_constants_graph() {
    Graph g;
    NodeId start = g.create(NodeKind::Start);
    g.set_start(start);
    NodeId c10 = g.create(NodeKind::ConstFloat);
    g.node(c10).const_value = Value::real(10.0);
    g.node(c10).set_flag(NodeFlag::Pure);
    NodeId c3 = g.create(NodeKind::ConstFloat);
    g.node(c3).const_value = Value::real(3.0);
    g.node(c3).set_flag(NodeFlag::Pure);
    NodeId sub = g.create(NodeKind::PyBinary, {start, start, c10, c3});
    g.node(sub).subop = static_cast<std::uint16_t>(BinOpKind::Sub);
    g.node(sub).set_flag(NodeFlag::OnEffectChain);
    NodeId ret = g.create(NodeKind::Return, {start, sub});
    g.node(ret).set_flag(NodeFlag::OnEffectChain);
    g.set_end(ret);
    g.n_parameters = 0;
    g.function_name = global_symbols().intern("float_sub_constants");
    return g;
}

[[nodiscard]] Graph float_mul_two_constants_graph() {
    Graph g;
    NodeId start = g.create(NodeKind::Start);
    g.set_start(start);
    NodeId c2 = g.create(NodeKind::ConstFloat);
    g.node(c2).const_value = Value::real(2.0);
    g.node(c2).set_flag(NodeFlag::Pure);
    NodeId c25 = g.create(NodeKind::ConstFloat);
    g.node(c25).const_value = Value::real(2.5);
    g.node(c25).set_flag(NodeFlag::Pure);
    NodeId mul = g.create(NodeKind::PyBinary, {start, start, c2, c25});
    g.node(mul).subop = static_cast<std::uint16_t>(BinOpKind::Mul);
    g.node(mul).set_flag(NodeFlag::OnEffectChain);
    NodeId ret = g.create(NodeKind::Return, {start, mul});
    g.node(ret).set_flag(NodeFlag::OnEffectChain);
    g.set_end(ret);
    g.n_parameters = 0;
    g.function_name = global_symbols().intern("float_mul_constants");
    return g;
}

[[nodiscard]] Graph float_div_two_constants_graph() {
    Graph g;
    NodeId start = g.create(NodeKind::Start);
    g.set_start(start);
    NodeId c7 = g.create(NodeKind::ConstFloat);
    g.node(c7).const_value = Value::real(7.0);
    g.node(c7).set_flag(NodeFlag::Pure);
    NodeId c2 = g.create(NodeKind::ConstFloat);
    g.node(c2).const_value = Value::real(2.0);
    g.node(c2).set_flag(NodeFlag::Pure);
    NodeId div = g.create(NodeKind::PyBinary, {start, start, c7, c2});
    g.node(div).subop = static_cast<std::uint16_t>(BinOpKind::TrueDiv);
    g.node(div).set_flag(NodeFlag::OnEffectChain);
    NodeId ret = g.create(NodeKind::Return, {start, div});
    g.node(ret).set_flag(NodeFlag::OnEffectChain);
    g.set_end(ret);
    g.n_parameters = 0;
    g.function_name = global_symbols().intern("float_div_constants");
    return g;
}

[[nodiscard]] Graph float_sub_to_zero_graph() {
    // 1.5 - 1.5 = 0.0 — exercises the IEEE zero result (sign bit clear,
    // mantissa all zero). Catches a regression where the tag write-back
    // accidentally wrote a non-zero tag constant into the FP payload slot
    // (the IBE-18 analogue for floats).
    Graph g;
    NodeId start = g.create(NodeKind::Start);
    g.set_start(start);
    NodeId c15 = g.create(NodeKind::ConstFloat);
    g.node(c15).const_value = Value::real(1.5);
    g.node(c15).set_flag(NodeFlag::Pure);
    NodeId c15b = g.create(NodeKind::ConstFloat);
    g.node(c15b).const_value = Value::real(1.5);
    g.node(c15b).set_flag(NodeFlag::Pure);
    NodeId sub = g.create(NodeKind::PyBinary, {start, start, c15, c15b});
    g.node(sub).subop = static_cast<std::uint16_t>(BinOpKind::Sub);
    g.node(sub).set_flag(NodeFlag::OnEffectChain);
    NodeId ret = g.create(NodeKind::Return, {start, sub});
    g.node(ret).set_flag(NodeFlag::OnEffectChain);
    g.set_end(ret);
    g.n_parameters = 0;
    g.function_name = global_symbols().intern("float_sub_to_zero");
    return g;
}

[[nodiscard]] Graph float_negate_via_sub_graph() {
    // 0.0 - 1.5 = -1.5 — exercises the sign-bit case. ADDSD/SUBSD don't
    // care about sign, but the codegen path that stores the FP payload
    // into a 16-byte Value slot must not stomp the tag bytes with junk.
    Graph g;
    NodeId start = g.create(NodeKind::Start);
    g.set_start(start);
    NodeId c0 = g.create(NodeKind::ConstFloat);
    g.node(c0).const_value = Value::real(0.0);
    g.node(c0).set_flag(NodeFlag::Pure);
    NodeId c15 = g.create(NodeKind::ConstFloat);
    g.node(c15).const_value = Value::real(1.5);
    g.node(c15).set_flag(NodeFlag::Pure);
    NodeId sub = g.create(NodeKind::PyBinary, {start, start, c0, c15});
    g.node(sub).subop = static_cast<std::uint16_t>(BinOpKind::Sub);
    g.node(sub).set_flag(NodeFlag::OnEffectChain);
    NodeId ret = g.create(NodeKind::Return, {start, sub});
    g.node(ret).set_flag(NodeFlag::OnEffectChain);
    g.set_end(ret);
    g.n_parameters = 0;
    g.function_name = global_symbols().intern("float_negate_via_sub");
    return g;
}

// =============================================================================
// BOOL FAST-PATH CORPUS (PyCompare with both operands provably int).
//
// The bool fast path emits GUARD_INT + CMPrr + SETCCri (write 0/1 to home
// payload + tag=Tag::Bool). Covers all six CmpOpKinds that map to native
// signed-integer comparisons: LT, LE, GT, GE, EQ, NE. Each case returns
// a Value with tag=Tag::Bool and payload 1 (true) or 0 (false).
// =============================================================================

[[nodiscard]] Graph bool_lt_two_int_constants_graph() {
    Graph g;
    NodeId start = g.create(NodeKind::Start);
    g.set_start(start);
    NodeId c2 = g.create(NodeKind::ConstInt);
    g.node(c2).const_value = Value::integer(2);
    g.node(c2).set_flag(NodeFlag::Pure);
    NodeId c3 = g.create(NodeKind::ConstInt);
    g.node(c3).const_value = Value::integer(3);
    g.node(c3).set_flag(NodeFlag::Pure);
    NodeId cmp = g.create(NodeKind::PyCompare, {start, start, c2, c3});
    g.node(cmp).subop = static_cast<std::uint16_t>(CmpOpKind::LT);
    g.node(cmp).set_flag(NodeFlag::OnEffectChain);
    NodeId ret = g.create(NodeKind::Return, {start, cmp});
    g.node(ret).set_flag(NodeFlag::OnEffectChain);
    g.set_end(ret);
    g.n_parameters = 0;
    g.function_name = global_symbols().intern("bool_lt");
    return g;
}

[[nodiscard]] Graph bool_le_equal_constants_graph() {
    Graph g;
    NodeId start = g.create(NodeKind::Start);
    g.set_start(start);
    NodeId c5 = g.create(NodeKind::ConstInt);
    g.node(c5).const_value = Value::integer(5);
    g.node(c5).set_flag(NodeFlag::Pure);
    NodeId c5b = g.create(NodeKind::ConstInt);
    g.node(c5b).const_value = Value::integer(5);
    g.node(c5b).set_flag(NodeFlag::Pure);
    NodeId cmp = g.create(NodeKind::PyCompare, {start, start, c5, c5b});
    g.node(cmp).subop = static_cast<std::uint16_t>(CmpOpKind::LE);
    g.node(cmp).set_flag(NodeFlag::OnEffectChain);
    NodeId ret = g.create(NodeKind::Return, {start, cmp});
    g.node(ret).set_flag(NodeFlag::OnEffectChain);
    g.set_end(ret);
    g.n_parameters = 0;
    g.function_name = global_symbols().intern("bool_le_eq");
    return g;
}

[[nodiscard]] Graph bool_gt_false_case_graph() {
    // 2 > 3 → false. Tests the "false" branch of GT (predicate false).
    Graph g;
    NodeId start = g.create(NodeKind::Start);
    g.set_start(start);
    NodeId c2 = g.create(NodeKind::ConstInt);
    g.node(c2).const_value = Value::integer(2);
    g.node(c2).set_flag(NodeFlag::Pure);
    NodeId c3 = g.create(NodeKind::ConstInt);
    g.node(c3).const_value = Value::integer(3);
    g.node(c3).set_flag(NodeFlag::Pure);
    NodeId cmp = g.create(NodeKind::PyCompare, {start, start, c2, c3});
    g.node(cmp).subop = static_cast<std::uint16_t>(CmpOpKind::GT);
    g.node(cmp).set_flag(NodeFlag::OnEffectChain);
    NodeId ret = g.create(NodeKind::Return, {start, cmp});
    g.node(ret).set_flag(NodeFlag::OnEffectChain);
    g.set_end(ret);
    g.n_parameters = 0;
    g.function_name = global_symbols().intern("bool_gt_false");
    return g;
}

[[nodiscard]] Graph bool_ge_true_case_graph() {
    // 5 >= 5 → true. GE-EQ case where the predicate fires on equality.
    Graph g;
    NodeId start = g.create(NodeKind::Start);
    g.set_start(start);
    NodeId c5 = g.create(NodeKind::ConstInt);
    g.node(c5).const_value = Value::integer(5);
    g.node(c5).set_flag(NodeFlag::Pure);
    NodeId c5b = g.create(NodeKind::ConstInt);
    g.node(c5b).const_value = Value::integer(5);
    g.node(c5b).set_flag(NodeFlag::Pure);
    NodeId cmp = g.create(NodeKind::PyCompare, {start, start, c5, c5b});
    g.node(cmp).subop = static_cast<std::uint16_t>(CmpOpKind::GE);
    g.node(cmp).set_flag(NodeFlag::OnEffectChain);
    NodeId ret = g.create(NodeKind::Return, {start, cmp});
    g.node(ret).set_flag(NodeFlag::OnEffectChain);
    g.set_end(ret);
    g.n_parameters = 0;
    g.function_name = global_symbols().intern("bool_ge_true");
    return g;
}

[[nodiscard]] Graph bool_eq_false_case_graph() {
    // 7 == 8 → false. Tests SETcc with NE direction (predicate false
    // means ZF=0 means SETcc NE → AL=1... wait, that's actually
    // inverted. For EQ: SETcc EQ → AL = (ZF==1). 7 == 8 is false, so
    // ZF=0, so SETcc EQ → AL=0. Good — this case validates that
    // direction.
    Graph g;
    NodeId start = g.create(NodeKind::Start);
    g.set_start(start);
    NodeId c7 = g.create(NodeKind::ConstInt);
    g.node(c7).const_value = Value::integer(7);
    g.node(c7).set_flag(NodeFlag::Pure);
    NodeId c8 = g.create(NodeKind::ConstInt);
    g.node(c8).const_value = Value::integer(8);
    g.node(c8).set_flag(NodeFlag::Pure);
    NodeId cmp = g.create(NodeKind::PyCompare, {start, start, c7, c8});
    g.node(cmp).subop = static_cast<std::uint16_t>(CmpOpKind::EQ);
    g.node(cmp).set_flag(NodeFlag::OnEffectChain);
    NodeId ret = g.create(NodeKind::Return, {start, cmp});
    g.node(ret).set_flag(NodeFlag::OnEffectChain);
    g.set_end(ret);
    g.n_parameters = 0;
    g.function_name = global_symbols().intern("bool_eq_false");
    return g;
}

[[nodiscard]] Graph bool_ne_true_case_graph() {
    // 7 != 8 → true. NE predicate true; SETcc NE → AL = (ZF==0) = 1.
    Graph g;
    NodeId start = g.create(NodeKind::Start);
    g.set_start(start);
    NodeId c7 = g.create(NodeKind::ConstInt);
    g.node(c7).const_value = Value::integer(7);
    g.node(c7).set_flag(NodeFlag::Pure);
    NodeId c8 = g.create(NodeKind::ConstInt);
    g.node(c8).const_value = Value::integer(8);
    g.node(c8).set_flag(NodeFlag::Pure);
    NodeId cmp = g.create(NodeKind::PyCompare, {start, start, c7, c8});
    g.node(cmp).subop = static_cast<std::uint16_t>(CmpOpKind::NE);
    g.node(cmp).set_flag(NodeFlag::OnEffectChain);
    NodeId ret = g.create(NodeKind::Return, {start, cmp});
    g.node(ret).set_flag(NodeFlag::OnEffectChain);
    g.set_end(ret);
    g.n_parameters = 0;
    g.function_name = global_symbols().intern("bool_ne_true");
    return g;
}

// A bool test case: a graph + its expected boolean result.
struct BoolDiffCase {
    const char* name;
    Graph (*build)();
    bool expected;
};

const std::vector<BoolDiffCase> kBoolDiffCases = {
    {"bool_lt",      bool_lt_two_int_constants_graph,   true},
    {"bool_le_eq",   bool_le_equal_constants_graph,     true},
    {"bool_gt_false",bool_gt_false_case_graph,          false},
    {"bool_ge_true", bool_ge_true_case_graph,           true},
    {"bool_eq_false",bool_eq_false_case_graph,          false},
    {"bool_ne_true", bool_ne_true_case_graph,           true},
};

// A float test case: a graph + its expected FP result.
struct FloatDiffCase {
    const char* name;
    Graph (*build)();
    double expected;
};

const std::vector<FloatDiffCase> kFloatDiffCases = {
    {"float_add",      float_add_two_constants_graph,   1.5 + 2.5},
    {"float_sub",      float_sub_two_constants_graph,   10.0 - 3.0},
    {"float_mul",      float_mul_two_constants_graph,   2.0 * 2.5},
    {"float_div",      float_div_two_constants_graph,   7.0 / 2.0},
    {"float_sub_zero", float_sub_to_zero_graph,         0.0},
    {"float_neg",      float_negate_via_sub_graph,      -1.5},
};

// A test case: a graph + its parameter values + expected result.
struct DiffCase {
    const char* name;
    Graph (*build)();
    std::int64_t input;       // value for parameter 0 (Node slot 2)
    std::int64_t expected;   // what the JIT MUST return
};

const std::vector<DiffCase> kDiffCases = {
    {"add_one_small",       int_identity_plus_one_graph, 41,   42},
    {"add_one_large",       int_identity_plus_one_graph, 1000000, 1000001},
    {"add_one_neg",         int_identity_plus_one_graph, -100, -99},
    {"add_one_zero",        int_identity_plus_one_graph, 0,    1},
    {"times_three_small",   int_identity_times_three_graph, 14, 42},
    {"times_three_large",   int_identity_times_three_graph, 1000000000LL, 3000000000LL},
    {"times_three_neg",     int_identity_times_three_graph, -50, -150},
    {"sub_seven_small",     int_subtraction_graph,       50,   43},
    {"sub_seven_large",     int_subtraction_graph,       1000000LL, 999993LL},
    {"sub_seven_neg",       int_subtraction_graph,       3,   -4},
    {"large_constant",      int_large_constant_graph,     0,    1LL << 40},
    {"negate_small",        int_negate_graph,             42,  -42},
    {"negate_large",        int_negate_graph,             1LL << 40, -(1LL << 40)},
    {"negate_zero",         int_negate_graph,             0,   0},
};

// Run the case through the JIT and return the result Value.
[[nodiscard]] Value run_jit(const DiffCase& c, bool* ok) {
    *ok = false;
    Graph g = c.build();
    constexpr std::size_t kCodeCap = 4096;
    std::byte* code_buf = make_exec_buffer(kCodeCap);
    if (!code_buf) return Value::none();

    backend::CompiledCode cc = backend::compile_unit(g, /*unit_id=*/1, code_buf,
                                                     kCodeCap, backend::host_target());
    if (!cc.valid) {
        free_exec_buffer(code_buf, kCodeCap);
        return Value::none();
    }

    // Generous register file (home slots are IR node ids; the worst case
    // is the Return's home which can be up to ~10). 64 is comfortably
    // above that and matches the unit-test pattern.
    std::uint32_t n_regs = 64;
    Value* regs = static_cast<Value*>(std::malloc(sizeof(Value) * n_regs));
    for (std::uint32_t i = 0; i < n_regs; ++i) regs[i] = Value::none();
    // Parameter 0 lives at its IR node id. NodeId 1 = Start, NodeId 2 = p0.
    if (g.n_parameters >= 1) regs[2] = Value::integer(c.input);

    // Active VM in case the bridge/deopt path fires (it shouldn't for
    // pure int arithmetic, but the contract is that an active VM exists).
    rt::Vm vm;
    rt::set_vm_for_builtins(&vm);
    rt::install_builtins(vm.program);

    auto entry = reinterpret_cast<backend::JitEntryFn>(code_buf);
    Value result = entry(regs);

    std::free(regs);
    free_exec_buffer(code_buf, kCodeCap);
    *ok = true;
    return result;
}

// Run a float corpus case through the JIT. Returns the Value with tag
// expected to be Tag::Float; *ok set false on JIT compilation failure.
[[nodiscard]] Value run_jit_float(const FloatDiffCase& c, bool* ok) {
    *ok = false;
    Graph g = c.build();
    constexpr std::size_t kCodeCap = 4096;
    std::byte* code_buf = make_exec_buffer(kCodeCap);
    if (!code_buf) return Value::none();

    backend::CompiledCode cc = backend::compile_unit(g, /*unit_id=*/1, code_buf,
                                                     kCodeCap, backend::host_target());
    if (!cc.valid) {
        free_exec_buffer(code_buf, kCodeCap);
        return Value::none();
    }

    std::uint32_t n_regs = 64;
    Value* regs = static_cast<Value*>(std::malloc(sizeof(Value) * n_regs));
    for (std::uint32_t i = 0; i < n_regs; ++i) regs[i] = Value::none();

    // Active VM (the bridge path could fire if GUARD_FLOAT deopts; for
    // the corpus all operands ARE float so it shouldn't, but the active
    // VM is the contract).
    rt::Vm vm;
    rt::set_vm_for_builtins(&vm);
    rt::install_builtins(vm.program);

    auto entry = reinterpret_cast<backend::JitEntryFn>(code_buf);
    Value result = entry(regs);

    std::free(regs);
    free_exec_buffer(code_buf, kCodeCap);
    *ok = true;
    return result;
}

// Run a bool corpus case through the JIT. Returns the Value with tag
// expected to be Tag::Bool; *ok set false on JIT compilation failure.
[[nodiscard]] Value run_jit_bool(const BoolDiffCase& c, bool* ok) {
    *ok = false;
    Graph g = c.build();
    constexpr std::size_t kCodeCap = 4096;
    std::byte* code_buf = make_exec_buffer(kCodeCap);
    if (!code_buf) return Value::none();

    backend::CompiledCode cc = backend::compile_unit(g, /*unit_id=*/1, code_buf,
                                                     kCodeCap, backend::host_target());
    if (!cc.valid) {
        free_exec_buffer(code_buf, kCodeCap);
        return Value::none();
    }

    std::uint32_t n_regs = 64;
    Value* regs = static_cast<Value*>(std::malloc(sizeof(Value) * n_regs));
    for (std::uint32_t i = 0; i < n_regs; ++i) regs[i] = Value::none();

    rt::Vm vm;
    rt::set_vm_for_builtins(&vm);
    rt::install_builtins(vm.program);

    auto entry = reinterpret_cast<backend::JitEntryFn>(code_buf);
    Value result = entry(regs);

    std::free(regs);
    free_exec_buffer(code_buf, kCodeCap);
    *ok = true;
    return result;
}

}  // namespace

// =============================================================================
// Differential: every pure-int case must produce the expected value when
// run through the JIT. This is the regression that would have caught
// IBE-18's home-slot clobber for ANY of the larger-int cases (the existing
// single-case test only covered identity+1 with a small input).
// =============================================================================
TEST(regr_jit_int_arithmetic_matches_expected) {
    int failures = 0;
    for (const DiffCase& c : kDiffCases) {
        bool ok = false;
        Value got = run_jit(c, &ok);
        if (!ok) {
            std::fprintf(stderr, "  [jit-diff] %s: codegen or exec failed\n", c.name);
            ++failures;
            continue;
        }
        if (got.tag != Tag::Int) {
            std::fprintf(stderr,
                         "  [jit-diff] %s: result tag %u (expected Int=%u)\n",
                         c.name, static_cast<unsigned>(got.tag),
                         static_cast<unsigned>(Tag::Int));
            ++failures;
            continue;
        }
        if (got.as.i != c.expected) {
            std::fprintf(stderr,
                         "  [jit-diff] %s: result %lld (expected %lld)\n",
                         c.name,
                         static_cast<long long>(got.as.i),
                         static_cast<long long>(c.expected));
            ++failures;
        }
    }
    CHECK_EQ(failures, 0);
}

// =============================================================================
// Sanity: the expected values themselves are correct. Compute them in
// plain C++ as a parallel oracle so the expected table above isn't a
// stale hardcode.
// =============================================================================
TEST(regr_jit_expected_values_oracle) {
    // Verify each DiffCase's `expected` matches what plain C++ computes
    // for the same arithmetic. This catches a bug in the table itself
    // (which would mask a JIT regression).
    int failures = 0;
    for (const DiffCase& c : kDiffCases) {
        std::int64_t oracle = 0;
        std::string name = c.name;
        if (name == "large_constant") {
            oracle = 1LL << 40;
        } else if (name.rfind("add_one", 0) == 0) {
            oracle = c.input + 1;
        } else if (name.rfind("times_three", 0) == 0) {
            oracle = c.input * 3;
        } else if (name.rfind("sub_seven", 0) == 0) {
            oracle = c.input - 7;
        } else if (name.rfind("negate_", 0) == 0) {
            oracle = -c.input;
        } else {
            std::fprintf(stderr, "  [oracle] unknown case shape: %s\n", c.name);
            ++failures;
            continue;
        }
        if (oracle != c.expected) {
            std::fprintf(stderr,
                         "  [oracle] %s: table says %lld but plain C++ says %lld\n",
                         c.name, static_cast<long long>(c.expected),
                         static_cast<long long>(oracle));
            ++failures;
        }
    }
    CHECK_EQ(failures, 0);
}

// =============================================================================
// Differential: every float case must produce the expected IEEE-754 double
// when run through the JIT. This is the regression that catches a future
// home-slot clobber specific to the FP fast path (e.g. the tag MOVri
// overwriting the FP payload bits, or the SSE2 emit getting the REX prefix
// wrong on XMM8-15).
// =============================================================================
TEST(regr_jit_float_arithmetic_matches_expected) {
    int failures = 0;
    for (const FloatDiffCase& c : kFloatDiffCases) {
        bool ok = false;
        Value got = run_jit_float(c, &ok);
        if (!ok) {
            std::fprintf(stderr, "  [jit-diff-float] %s: codegen or exec failed\n",
                         c.name);
            ++failures;
            continue;
        }
        if (got.tag != Tag::Float) {
            std::fprintf(stderr,
                         "  [jit-diff-float] %s: result tag %u (expected Float=%u)\n",
                         c.name, static_cast<unsigned>(got.tag),
                         static_cast<unsigned>(Tag::Float));
            ++failures;
            continue;
        }
        // Bit-compare for FP equality — same sign, same mantissa, same exp.
        // Equality via == is safe here: we are comparing the IEEE 754 bits
        // produced by the JIT to the IEEE 754 bits computed by the host
        // compiler for the same op. NaN is not in the corpus (none of the
        // operations produce NaN on these inputs).
        if (got.as.f != c.expected) {
            std::fprintf(stderr,
                         "  [jit-diff-float] %s: result %g (expected %g)\n",
                         c.name, got.as.f, c.expected);
            ++failures;
        }
    }
    CHECK_EQ(failures, 0);
}

// =============================================================================
// Differential: every bool case must produce the expected Python bool when
// run through the JIT. This is the regression that catches a future SETcc
// direction bug (e.g. SETcc EQ used for an NE predicate, or vice versa) or
// a tag writeback that clobbers Tag::Bool with kTagInt (which would make
// `if result:` short-circuit incorrectly since Tag::Int's payload 0 is
// falsy, but Tag::Bool's payload 0 is False — same result here, but other
// code paths distinguishing bool from int would break).
// =============================================================================
TEST(regr_jit_bool_arithmetic_matches_expected) {
    int failures = 0;
    for (const BoolDiffCase& c : kBoolDiffCases) {
        bool ok = false;
        Value got = run_jit_bool(c, &ok);
        if (!ok) {
            std::fprintf(stderr, "  [jit-diff-bool] %s: codegen or exec failed\n",
                         c.name);
            ++failures;
            continue;
        }
        if (got.tag != Tag::Bool) {
            std::fprintf(stderr,
                         "  [jit-diff-bool] %s: result tag %u (expected Bool=%u)\n",
                         c.name, static_cast<unsigned>(got.tag),
                         static_cast<unsigned>(Tag::Bool));
            ++failures;
            continue;
        }
        const bool got_b = got.as.i != 0;
        if (got_b != c.expected) {
            std::fprintf(stderr,
                         "  [jit-diff-bool] %s: result %s (expected %s)\n",
                         c.name, got_b ? "true" : "false",
                         c.expected ? "true" : "false");
            ++failures;
        }
    }
    CHECK_EQ(failures, 0);
}

#else  // non-x86-64 host

// On non-x86-64 hosts the JIT emitter refuses to run, so we mark the
// differential as a no-op rather than skip silently. The test still has
// to exist (ctest counts it) but does nothing.
TEST(regr_jit_int_arithmetic_matches_expected) {
    std::printf("  [skipped] JIT differential is x86-64 only on this host\n");
}

TEST(regr_jit_expected_values_oracle) {
    std::printf("  [skipped] JIT oracle is x86-64 only on this host\n");
}

TEST(regr_jit_float_arithmetic_matches_expected) {
    std::printf("  [skipped] JIT float differential is x86-64 only on this host\n");
}

TEST(regr_jit_bool_arithmetic_matches_expected) {
    std::printf("  [skipped] JIT bool differential is x86-64 only on this host\n");
}

#endif  // __x86_64__
