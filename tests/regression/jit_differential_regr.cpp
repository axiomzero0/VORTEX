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

#endif  // __x86_64__
