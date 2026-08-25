// =============================================================================
// tests/unit/backend_test.cpp — target layer + backend gating tests.
//
// Portability contract (Rules 24, 27): every machine fact is QUERIED — from
// the compiler's architecture identification, from real hardware probes
// (CPUID / getauxval / CTR_EL0), or from a CMake override — never hardcoded
// into a struct default or a cfg constant. These tests pin that contract:
//
//   - descriptors match the compiled architecture (an AArch64 build can
//     never produce an x86-64 descriptor, and vice versa)
//   - register partitions are internally consistent (no duplicates, frame
//     roles excluded, stack/frame/platform registers never allocatable)
//   - the x86-64 emitter REFUSES foreign-arch descriptors: zero bytes, not
//     garbage that happens to run on the build machine
//   - polyhedral (Pass 33) is opt-in at BOTH gates — pipeline filter and
//     the pass itself — and opting in does not change program results
// =============================================================================

#include "harness.hpp"

#include <cstdio>
#include <string>
#include <unistd.h>

#include "vortex/backend/codegen.hpp"
#include "vortex/backend/target.hpp"
#include "vortex/frontend/lowering.hpp"
#include "vortex/frontend/parser.hpp"
#include "vortex/ir/graph.hpp"
#include "vortex/passes/pass_pipeline.hpp"
#include "vortex/passes/passes_fwd.hpp"
#include "vortex/rt/driver.hpp"
#include "vortex/rt/interp.hpp"
#include "vortex/support/arena.hpp"
#include "vortex/support/symbol_table.hpp"

using namespace vortex;
using namespace vortex::backend;
using namespace vortex::ir;
namespace fe = vortex::fe;
namespace passes = vortex::passes;
namespace rt = vortex::rt;

// --- compile-time contract ----------------------------------------------------

static_assert(compiled_arch() != Arch::Unknown,
              "the header static_assert already enforces this; double check");

// The AOT descriptor can never claim an architecture other than the one this
// compiler was built for — the old hard-coded X86_64 default is dead.
static_assert(aot_target().architecture == compiled_arch());

// MCond is a dense, arch-neutral enumeration (it once carried raw x86-64
// opcode bytes as its values).
static_assert(enum_size(MCond::Count) == 6);
static_assert(static_cast<std::uint8_t>(MCond::EQ) == 0);
static_assert(static_cast<std::uint8_t>(MCond::GT) == 5);

// Cost classification is total and arch-neutral.
static_assert(cost_class(MOp::IMULrr) == CostClass::Mul);
static_assert(cost_class(MOp::MOVmr) == CostClass::Store);
static_assert(cost_class(MOp::CALLri) == CostClass::Call);

// The latency model must be ordered like real machines are.
static_assert(aot_target().latency(CostClass::Div) > aot_target().latency(CostClass::Alu));
static_assert(aot_target().latency(CostClass::Mul) > aot_target().latency(CostClass::Alu));
static_assert(aot_target().latency(CostClass::Call) >= aot_target().latency(CostClass::Branch));

// --- helpers ------------------------------------------------------------------

namespace {

[[nodiscard]] bool partition_consistent(const TargetDescriptor& t) noexcept {
    if (t.allocatable_gprs == 0 || t.allocatable_gprs > kMaxAllocatable) return false;
    if (t.allocatable_gprs + enum_size(ReservedGPR::Count) > t.gpr_count) return false;
    // no duplicated encodings in the allocatable set
    for (std::uint32_t i = 0; i < t.allocatable_gprs; ++i) {
        for (std::uint32_t j = i + 1; j < t.allocatable_gprs; ++j) {
            if (t.allocatable[i] == t.allocatable[j]) return false;
        }
    }
    // roles are distinct and never allocatable
    for (std::uint32_t r = 0; r < enum_size(ReservedGPR::Count); ++r) {
        for (std::uint32_t s = r + 1; s < enum_size(ReservedGPR::Count); ++s) {
            if (t.reserved[r] == t.reserved[s]) return false;
        }
        for (std::uint32_t i = 0; i < t.allocatable_gprs; ++i) {
            if (t.allocatable[i] == t.reserved[r]) return false;
        }
    }
    return true;
}

[[nodiscard]] bool any_encoding(const TargetDescriptor& t, std::uint8_t enc) noexcept {
    for (std::uint32_t i = 0; i < t.allocatable_gprs; ++i) {
        if (t.allocatable[i] == enc) return true;
    }
    return false;
}

[[nodiscard]] Graph tiny_graph() {
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
    g.set_end(ret);
    g.n_parameters = 1;
    g.function_name = global_symbols().intern("tiny");
    return g;
}

// While-loop form: the induction variables are genuine loop phis (the
// for-in-range form's iteration values come from IterNext nodes, which the
// pass's affine-index legality check correctly refuses to treat as IVs).
const char* kNestSrc =
    "def f(a):\n"
    "    total = 0\n"
    "    i = 0\n"
    "    while i < 3:\n"
    "        j = 0\n"
    "        while j < 3:\n"
    "            total = total + a[j]\n"
    "            j = j + 1\n"
    "        i = i + 1\n"
    "    return total\n";

// Lower the nested-loop function to its Sea-of-Nodes graph (frontend only).
[[nodiscard]] Graph lower_nested_loops(bool* ok) {
    *ok = false;
    Graph empty;
    BumpArena arena;
    Result<fe::Module*> ast = fe::compile_to_ast(arena, kNestSrc);
    if (!ast) return empty;
    fe::LowerContext lctx;
    stdx::small_vector<SymbolId, 8> no_caps;
    SymbolId mod = global_symbols().intern("nestmod");
    Result<fe::LoweredUnit> top =
        fe::lower_unit(**ast, lctx, nullptr, mod, no_caps, false, 0xFFFFFFFF);
    if (!top || (*top).children.empty()) return empty;
    fe::PendingFunction& f = (*top).children[0];
    Result<fe::LoweredUnit> fu = fe::lower_unit(**ast, lctx, f.def, f.name, f.captures,
                                                false, f.code_unit_hint);
    if (!fu) return empty;
    *ok = true;
    return (*fu).graph;
}

[[nodiscard]] bool has_interchange_hint(const Graph& g) noexcept {
    bool hinted = false;
    g.for_each_live([&](NodeId id) {
        if (g.node(id).kind == NodeKind::Loop && g.node(id).aux1 == 1) hinted = true;
    });
    return hinted;
}

}  // namespace

// --- target descriptor ----------------------------------------------------------

TEST(target_host_matches_compiled_arch) {
    const TargetDescriptor& host = host_target();
    CHECK(host.architecture == compiled_arch());
    CHECK(detect_host_target().architecture == compiled_arch());

    // Probed machine facts are sane whatever the host is.
    CHECK(host.gpr_count >= 16);
    CHECK(host.simd_width_bytes >= 16);
    CHECK(host.simd_width_bytes <= 64);
    CHECK(host.simd_width_bytes % 16 == 0);
    CHECK(host.cache_line_bytes >= 16);
    CHECK(host.cache_line_bytes <= 256);
    CHECK((host.cache_line_bytes & (host.cache_line_bytes - 1)) == 0);
    CHECK(partition_consistent(host));
}

TEST(target_x8664_partition_facts) {
    constexpr TargetDescriptor t = x86_64_baseline();
    CHECK(t.architecture == Arch::X86_64);
    CHECK(t.gpr_count == 16);
    CHECK(t.allocatable_gprs == 10);
    CHECK(t.simd_width_bytes == 16);   // SSE2 is the ISA floor, not a guess
    CHECK(t.reserved[static_cast<std::uint32_t>(ReservedGPR::FrameBase)] == x86::R12);
    CHECK(t.reserved[static_cast<std::uint32_t>(ReservedGPR::ConstPool)] == x86::R13);
    CHECK(t.reserved[static_cast<std::uint32_t>(ReservedGPR::VMContext)] == x86::R14);
    CHECK(t.reserved[static_cast<std::uint32_t>(ReservedGPR::DeoptCtx)] == x86::R15);
    CHECK(!any_encoding(t, x86::RSP));   // stack pointer never allocatable
    CHECK(!any_encoding(t, x86::RBP));   // frame pointer never allocatable
    CHECK(partition_consistent(t));
}

TEST(target_aarch64_partition_facts) {
    constexpr TargetDescriptor t = aarch64_baseline();
    CHECK(t.architecture == Arch::AArch64);
    CHECK(t.gpr_count == 31);
    CHECK(t.allocatable_gprs == 24);
    CHECK(t.simd_width_bytes == 16);   // ASIMD is the armv8-a floor
    CHECK(t.reserved[static_cast<std::uint32_t>(ReservedGPR::FrameBase)] == 27);
    CHECK(t.reserved[static_cast<std::uint32_t>(ReservedGPR::ConstPool)] == 26);
    CHECK(t.reserved[static_cast<std::uint32_t>(ReservedGPR::VMContext)] == 25);
    CHECK(t.reserved[static_cast<std::uint32_t>(ReservedGPR::DeoptCtx)] == 24);
    CHECK(!any_encoding(t, aarch64::kPlatform));     // x18 platform register
    CHECK(!any_encoding(t, aarch64::kFramePointer)); // x29
    CHECK(!any_encoding(t, aarch64::kLinkRegister)); // x30
    CHECK(t.has(TargetFeature::ASIMD));
    CHECK(partition_consistent(t));
}

TEST(target_latency_model_is_ordered) {
    const TargetDescriptor& t = host_target();
    CHECK(t.latency(CostClass::Div) > t.latency(CostClass::Alu));
    CHECK(t.latency(CostClass::Mul) > t.latency(CostClass::Alu));
    CHECK(t.latency(CostClass::Load) >= t.latency(CostClass::Move));
    CHECK(t.latency(CostClass::Call) >= t.latency(CostClass::Branch));
    CHECK(t.latency(CostClass::Move) > 0);
}

// --- codegen architecture gate ----------------------------------------------------

TEST(codegen_refuses_foreign_architecture) {
    Graph g = tiny_graph();
    std::byte buffer[512];

    // x86-64 emitter + AArch64 descriptor: refuse, emit NOTHING.
    CompiledCode foreign = compile_unit(g, 1, buffer, sizeof(buffer), aarch64_baseline());
    CHECK(!foreign.valid);
    CHECK_EQ(foreign.code_size, std::size_t{0});

    // Undescribed target: refuse as well.
    TargetDescriptor unknown{};
    CompiledCode blank = compile_unit(g, 1, buffer, sizeof(buffer), unknown);
    CHECK(!blank.valid);
    CHECK_EQ(blank.code_size, std::size_t{0});

    // No buffer: refuse.
    CompiledCode nobuf = compile_unit(g, 1, nullptr, 0, host_target());
    CHECK(!nobuf.valid);

    // Positive control on the native arch: the gate is not a blanket veto.
    if (compiled_arch() == Arch::X86_64) {
        CompiledCode native = compile_unit(g, 1, buffer, sizeof(buffer), host_target());
        CHECK(native.valid);
        CHECK(native.code_size > 0);
    }
}

// --- polyhedral opt-in -------------------------------------------------------------

TEST(polyhedral_pipeline_gate) {
    // Tier 2, no opt-in: excluded.
    passes::PassContext ctx;
    ctx.tier = passes::TierMode::Tier2;
    passes::TierFilter f2{ctx.tier};
    CHECK(!f2.include("33_polyhedral", ctx));
    CHECK(f2.include("03_trivial_dce", ctx));

    // Tier 2, opted in: included.
    ctx.options.set(passes::OptOption::Polyhedral);
    CHECK(f2.include("33_polyhedral", ctx));

    // Tier 1: budget gate excludes it even when opted in.
    passes::PassContext t1;
    t1.tier = passes::TierMode::Tier1;
    t1.options.set(passes::OptOption::Polyhedral);
    passes::TierFilter f1{t1.tier};
    CHECK(!f1.include("33_polyhedral", t1));

    // Tier 3 honors the opt-in flag symmetrically.
    passes::PassContext t3;
    t3.tier = passes::TierMode::Tier3;
    passes::TierFilter f3{t3.tier};
    CHECK(!f3.include("33_polyhedral", t3));
    t3.options.set(passes::OptOption::Polyhedral);
    CHECK(f3.include("33_polyhedral", t3));
}

TEST(polyhedral_pass_self_gate) {
    // Gate 2 (defense in depth): direct invocation without the flag is a
    // no-op even though the graph is a perfect interchange candidate.
    bool ok = false;
    Graph g = lower_nested_loops(&ok);
    CHECK(ok);
    if (!ok) return;

    passes::PassContext off;
    off.tier = passes::TierMode::Tier2;   // tier alone must NOT enable it
    passes::P33_PolyhedralOptimization p33;
    Result<passes::PassResult> r_off = p33.run(g, off);
    CHECK(r_off.has_value());
    CHECK(!has_interchange_hint(g));   // opt-out: nothing touched

    // Opted in: the analysis runs and records the interchange hint on the nest.
    passes::PassContext on;
    on.tier = passes::TierMode::Tier2;
    on.options.set(passes::OptOption::Polyhedral);
    Result<passes::PassResult> r_on = p33.run(g, on);
    CHECK(r_on.has_value());
    CHECK(has_interchange_hint(g));    // opted in: transform recorded
}

TEST(polyhedral_end_to_end_differential) {
    const char* src =
        "def f(a):\n"
        "    total = 0\n"
        "    i = 0\n"
        "    while i < 3:\n"
        "        j = 0\n"
        "        while j < 3:\n"
        "            total = total + a[j]\n"
        "            j = j + 1\n"
        "        i = i + 1\n"
        "    return total\n"
        "print(f([1, 2, 3]))\n";

    auto run = [&](const rt::CompileOptions& opts, bool* ok, std::string* out) {
        rt::Vm vm;
        rt::set_vm_for_builtins(&vm);
        rt::install_builtins(vm.program);
        std::fflush(stdout);
        int saved = dup(fileno(stdout));
        FILE* cap = tmpfile();
        dup2(fileno(cap), fileno(stdout));
        Result<Value> r = rt::run_source(vm, src, opts);
        std::fflush(stdout);
        dup2(saved, fileno(stdout));
        close(saved);
        long size = ftell(cap);
        rewind(cap);
        out->assign(static_cast<std::size_t>(size > 0 ? size : 0), ' ');
        if (size > 0) fread(out->data(), 1, static_cast<size_t>(size), cap);
        fclose(cap);
        *ok = r.has_value();
    };

    bool ok1 = false, ok2 = false;
    std::string out1, out2;
    rt::CompileOptions off;             // default: polyhedral NOT requested
    rt::CompileOptions on;
    on.polyhedral = true;

    run(off, &ok1, &out1);
    run(on, &ok2, &out2);

    CHECK(ok1);
    CHECK(ok2);
    // 3 outer x (1+2+3) = 18; opting in must not change observable results.
    CHECK_EQ(out1, std::string("18\n"));
    CHECK_EQ(out2, out1);
}
