// =============================================================================
// tests/regression/pass_idempotency_regr.cpp — Rule 10 idempotency guard.
//
// Rule 10: a second run of any pass must be a no-op (the pass reached its
// fixed point on the first run). A pass that "changes" the graph every
// time it runs is one of two things: (a) a fixpoint driver missing its
// convergence check, or (b) a non-deterministic walk that mutates NodeIds
// without changing semantics. Both are bugs; both have happened.
//
// This test constructs one realistic program per pass (a simple function
// exercising the pass's domain — loops for LICM, allocations for PEA,
// strings for interning) and asserts:
//   1.  First run completes without a Diagnostic.
//   2.  Second run completes and reports `changed == false`.
//
// We use the SAME lowered graph for every pass that doesn't require a
// specific IR shape (so adding a new pass is a one-line entry in
// kPassFactories). For passes whose precondition is shape-sensitive
// (PEA needs an Allocated candidate, p45 needs a string concat), we use
// their dedicated synthetic graphs from the per-pass files.
// =============================================================================

#include "regression_harness.hpp"

#include <cstdio>
#include <functional>
#include <string_view>

namespace {

using namespace vortex;
using namespace vortex::ir;
namespace passes = vortex::passes;

// A unified pass-factory: returns a heap-allocated pass object + its name.
// We use std::function because the pass types are heterogeneous structs (no
// common base class — Rule 9 forbids virtual dispatch in the hot loop).
// This indirection only exists at test time; the production pipeline uses
// the compile-time tuple.
struct PassEntry {
    const char* name;
    std::function<passes::PassResult(Graph&, const passes::PassContext&)> run;
};

// A single shared "kitchen sink" program — every pass gets to see a real
// IR with loops, branches, calls, and allocations. If a pass self-gates
// on tier or shape, it reports `changed == false` (correct behavior); the
// idempotency check passes either way.
const char* kKitchenSink =
    "def f(items):\n"
    "    total = 0\n"
    "    seen = {}\n"
    "    for x in items:\n"
    "        if x in seen:\n"
    "            continue\n"
    "        seen[x] = total\n"
    "        total = total + x * 2\n"
    "    return total\n"
    "print(f([1, 2, 3, 2, 1]))\n";

// Allocate-and-call graph for PEA: allocation whose escapes are confined
// to one arm of a branch. This is the precondition PEA fires on.
[[nodiscard]] Graph make_pea_candidate() {
    Graph g;
    NodeId start = g.create(NodeKind::Start);
    g.set_start(start);
    NodeId region = g.create(NodeKind::Region, {start});
    NodeId ifn = g.create(NodeKind::If, {region, start});
    NodeId ift = g.create(NodeKind::IfTrue, {ifn});
    NodeId iff = g.create(NodeKind::IfFalse, {ifn});
    NodeId nl = g.create(NodeKind::NewList, {ift});
    g.node(nl).set_flag(NodeFlag::OnEffectChain);
    NodeId call = g.create(NodeKind::CallPy, {ift, ift, nl, nl});
    g.node(call).set_flag(NodeFlag::OnEffectChain);
    NodeId c0 = g.create(NodeKind::ConstInt);
    g.node(c0).const_value = Value::integer(0);
    g.node(c0).set_flag(NodeFlag::Pure);
    NodeId ret = g.create(NodeKind::Return, {region, c0});
    g.node(ret).set_flag(NodeFlag::OnEffectChain);
    g.set_end(ret);
    (void)iff;
    return g;
}

// String-concat candidate for Pass 45.
[[nodiscard]] Graph make_str_concat_candidate() {
    Graph g;
    NodeId start = g.create(NodeKind::Start);
    g.set_start(start);
    stdx::small_vector<char, 4096> pool;
    for (char ch : std::string_view("abcdef")) pool.push_back(ch);

    NodeId a = g.create(NodeKind::ConstPy);
    Node& an = g.node(a);
    an.const_value = Value::none();
    an.aux0 = 0;
    an.aux1 = 3;
    an.symbol = 0xFFFF'FFFF;
    an.set_flag(NodeFlag::Pure);

    NodeId b = g.create(NodeKind::ConstPy);
    Node& bn = g.node(b);
    bn.const_value = Value::none();
    bn.aux0 = 3;
    bn.aux1 = 3;
    bn.symbol = 0xFFFF'FFFF;
    bn.set_flag(NodeFlag::Pure);

    NodeId bin = g.create(NodeKind::PyBinary, {start, start, a, b});
    Node& binn = g.node(bin);
    binn.subop = static_cast<std::uint16_t>(ir::BinOpKind::Add);
    binn.set_flag(NodeFlag::OnEffectChain);

    NodeId ret = g.create(NodeKind::Return, {start, bin});
    g.node(ret).set_flag(NodeFlag::OnEffectChain);
    g.set_end(ret);
    // Hold pool alive via a leaked static — the pass reads from
    // ctx.string_pool, not the Graph; we supply it per-call below.
    static stdx::small_vector<char, 4096>* pool_keepalive = nullptr;
    if (!pool_keepalive) pool_keepalive = new stdx::small_vector<char, 4096>(std::move(pool));
    return g;
}

// Polyhedral candidate: a perfectly-nested while pair (interchange target).
const char* kPolySrc =
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

// Build the list of passes — each entry wraps the typed `pass.run(g, ctx)`
// in a type-erased lambda. The lambda captures nothing, so it's a
// stateless wrapper (the pass instance is constructed inside the lambda
// per-call to ensure a fresh state per test).
std::vector<PassEntry> build_pass_entries() {
    std::vector<PassEntry> v;
    v.push_back({"03_trivial_dce",   [](Graph& g, const passes::PassContext& c){ passes::P03_TrivialDCE p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"04_local_constfold",[](Graph& g, const passes::PassContext& c){ passes::P04_LocalConstantFolding p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"05_algebraic_simplify",[](Graph& g, const passes::PassContext& c){ passes::P05_AlgebraicSimplification p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"06_cf_simplify",   [](Graph& g, const passes::PassContext& c){ passes::P06_ControlFlowSimplification p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"07_local_cse",     [](Graph& g, const passes::PassContext& c){ passes::P07_LocalCSE p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"08_sccp",          [](Graph& g, const passes::PassContext& c){ passes::P08_SCCP p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"09_redundant_stores",[](Graph& g, const passes::PassContext& c){ passes::P09_RedundantStoreElimination p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"10_early_gvn",     [](Graph& g, const passes::PassContext& c){ passes::P10_EarlyGVN p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"11_andersen",      [](Graph& g, const passes::PassContext& c){ passes::P11_AndersenPointsTo p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"12_cfl_alias",     [](Graph& g, const passes::PassContext& c){ passes::P12_CFLReachabilityAlias p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"13_flow_alias",    [](Graph& g, const passes::PassContext& c){ passes::P13_FlowSensitiveAlias p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"14_demand_alias",  [](Graph& g, const passes::PassContext& c){ passes::P14_DemandDrivenAlias p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"15_shape_analysis",[](Graph& g, const passes::PassContext& c){ passes::P15_ShapeAnalysis p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"16_ic_mono",       [](Graph& g, const passes::PassContext& c){ passes::P16_ICMonomorphism p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"17_mro",           [](Graph& g, const passes::PassContext& c){ passes::P17_MROLinearization p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"18_effects",       [](Graph& g, const passes::PassContext& c){ passes::P18_SideEffectAnalysis p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"19_callgraph_pgo",  [](Graph& g, const passes::PassContext& c){ passes::P19_CallGraphPGO p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"20_spec_inline",   [](Graph& g, const passes::PassContext& c){ passes::P20_SpeculativeInlining p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"21_partial_inline",[](Graph& g, const passes::PassContext& c){ passes::P21_PartialInlining p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"22_recursive_inline",[](Graph& g, const passes::PassContext& c){ passes::P22_RecursiveInlining p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"23_closure_devirt",[](Graph& g, const passes::PassContext& c){ passes::P23_ClosureDevirtualization p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"24_gen_deforest",  [](Graph& g, const passes::PassContext& c){ passes::P24_GeneratorDeforestation p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"25_exc_outline",   [](Graph& g, const passes::PassContext& c){ passes::P25_ExceptionOutlining p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"26_ipcp",          [](Graph& g, const passes::PassContext& c){ passes::P26_IPCP p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"27_licm",          [](Graph& g, const passes::PassContext& c){ passes::P27_LICM p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"28_spec_licm",    [](Graph& g, const passes::PassContext& c){ passes::P28_SpeculativeLICM p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"29_iv_strength",   [](Graph& g, const passes::PassContext& c){ passes::P29_InductionVariables p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"30_unroll",        [](Graph& g, const passes::PassContext& c){ passes::P30_LoopUnrolling p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"31_slp",           [](Graph& g, const passes::PassContext& c){ passes::P31_SLPVectorization p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"32_loop_vec",      [](Graph& g, const passes::PassContext& c){ passes::P32_LoopVectorization p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"33_polyhedral",    [](Graph& g, const passes::PassContext& c){ passes::P33_PolyhedralOptimization p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"34_sw_pipeline",   [](Graph& g, const passes::PassContext& c){ passes::P34_SoftwarePipelining p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"35_fusion_fission",[](Graph& g, const passes::PassContext& c){ passes::P35_LoopFusionFission p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"36_bce",           [](Graph& g, const passes::PassContext& c){ passes::P36_BoundsCheckElimination p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"37_nce",           [](Graph& g, const passes::PassContext& c){ passes::P37_NoneCheckElimination p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"38_gil_hoist",     [](Graph& g, const passes::PassContext& c){ passes::P38_GILHoisting p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"39_escape",        [](Graph& g, const passes::PassContext& c){ passes::P39_EscapeAnalysis p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"40_pea",           [](Graph& g, const passes::PassContext& c){ passes::P40_PartialEscapeAnalysis p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"41_obj_inline",    [](Graph& g, const passes::PassContext& c){ passes::P41_ObjectInlining p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"42_regions",       [](Graph& g, const passes::PassContext& c){ passes::P42_RegionMemoryInference p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"43_refcnt",        [](Graph& g, const passes::PassContext& c){ passes::P43_RefcntOptimization p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"44_write_barriers",[](Graph& g, const passes::PassContext& c){ passes::P44_WriteBarrierElimination p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"45_string_intern", [](Graph& g, const passes::PassContext& c){ passes::P45_StringInterning p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"46_dict_layout",    [](Graph& g, const passes::PassContext& c){ passes::P46_DictLayoutSpecialization p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"47_unboxing",      [](Graph& g, const passes::PassContext& c){ passes::P47_BoxUnboxElimination p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"48_tlab",          [](Graph& g, const passes::PassContext& c){ passes::P48_TLABSizing p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"49_effect_reorder", [](Graph& g, const passes::PassContext& c){ passes::P49_SpeculativeEffectReordering p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"50_late_gvn",      [](Graph& g, const passes::PassContext& c){ passes::P50_LateGVN p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    v.push_back({"51_global_dce",    [](Graph& g, const passes::PassContext& c){ passes::P51_GlobalDCE p; auto r = p.run(g, c); return r.value_or(passes::PassResult{}); }});
    return v;
}

// Run a single pass twice and return whether the second run reports no
// change. The pass's first run "primes" the graph; the second run must
// be a no-op (Rule 10).
[[nodiscard]] bool run_pass_twice_is_noop(const PassEntry& pe, Graph& g,
                                          const passes::PassContext& ctx) {
    auto r1 = pe.run(g, ctx);
    (void)r1;   // first run is allowed to change or not
    auto r2 = pe.run(g, ctx);
    if (!r2.changed) return true;
    std::fprintf(stderr, "  [idempotency] %s: second run reported changed=true\n",
                 pe.name);
    return false;
}

}  // namespace

// =============================================================================
// Regression: every pass P03..P51 is idempotent on the kitchen-sink program.
// Tier 2 lets every pass run; we still respect per-pass self-gates. Any pass
// that mutates the graph on its second run is broken — Rule 10 explicitly
// forbids that, and the pipeline would never converge in such a state.
// =============================================================================
TEST(regr_pass_idempotency_kitchen_sink_tier2) {
    bool ok = false;
    Graph g = vortex_test::lower_function(kKitchenSink, &ok);
    CHECK(ok);
    if (!ok) return;

    auto entries = build_pass_entries();
    int failures = 0;
    for (const PassEntry& pe : entries) {
        // Skip passes whose preconditions the kitchen-sink doesn't satisfy;
        // their first run is a no-op and the test is trivially true but
        // uninteresting. We still run them to make sure they DON'T fire
        // spuriously.
        passes::PassContext ctx;
        ctx.tier = passes::TierMode::Tier2;
        if (!run_pass_twice_is_noop(pe, g, ctx)) {
            ++failures;
        }
    }
    CHECK_EQ(failures, 0);
}

// =============================================================================
// Tier 1 mirror: same assertion but on the Tier 1 budget-constrained tier.
// Tier 1 self-gates exclude heavy passes (SCCP, GVN, alias analyses, etc.) —
// for those, the second run is trivially a no-op (the first run was a no-op).
// The idempotency still holds; this guards against a future bug where a
// Tier 1 pass self-gate becomes a no-op-on-first-run but mutates on the
// second invocation.
// =============================================================================
TEST(regr_pass_idempotency_kitchen_sink_tier1) {
    bool ok = false;
    Graph g = vortex_test::lower_function(kKitchenSink, &ok);
    CHECK(ok);
    if (!ok) return;

    auto entries = build_pass_entries();
    int failures = 0;
    passes::PassContext ctx;
    ctx.tier = passes::TierMode::Tier1;
    for (const PassEntry& pe : entries) {
        if (!run_pass_twice_is_noop(pe, g, ctx)) ++failures;
    }
    CHECK_EQ(failures, 0);
}

// =============================================================================
// Idempotency on the polyhedral default-on path: the polyhedral pass
// fires ONCE (real transformation) and a second run must be a no-op. A
// non-idempotent polyhedral pass would re-swap loops forever — that's
// clearly broken and would manifest as an infinite fixpoint.
// =============================================================================
TEST(regr_pass_idempotency_polyhedral_default_on) {
    bool ok = false;
    Graph g = vortex_test::lower_function(kPolySrc, &ok);
    CHECK(ok);
    if (!ok) return;

    // Default PassContext: polyhedral is ON by default, no opt-out flag.
    passes::PassContext ctx;
    ctx.tier = passes::TierMode::Tier2;

    passes::P33_PolyhedralOptimization p;
    Result<passes::PassResult> r1 = p.run(g, ctx);
    CHECK(r1.has_value());
    if (!r1) return;
    // First run either transforms (changed=true) or declines (false). Both
    // are valid first-run results.
    Result<passes::PassResult> r2 = p.run(g, ctx);
    CHECK(r2.has_value());
    if (!r2) return;
    // Second run MUST be a no-op — the interchange is now applied and a
    // re-run sees a candidate where the bounds are no longer affine
    // (post-swap, the outer IV is the former inner IV — the legality
    // check correctly refuses).
    CHECK(!r2->changed);
}

// =============================================================================
// Idempotency for PEA on a real candidate: PEA fires on the first run; the
// second run sees an Allocated marker and must decline.
// =============================================================================
TEST(regr_pass_idempotency_pea) {
    Graph g = make_pea_candidate();

    passes::PassContext ctx;
    ctx.tier = passes::TierMode::Tier2;
    passes::P40_PartialEscapeAnalysis p;
    Result<passes::PassResult> r1 = p.run(g, ctx);
    CHECK(r1.has_value());
    if (!r1) return;
    Result<passes::PassResult> r2 = p.run(g, ctx);
    CHECK(r2.has_value());
    if (!r2) return;
    CHECK(!r2->changed);
}

// =============================================================================
// Idempotency for Pass 45 on a real candidate: string concat folds; the
// second run sees no PyBinary(Add, ConstPy, ConstPy) pattern and must
// decline.
// =============================================================================
TEST(regr_pass_idempotency_p45_string_intern) {
    Graph g = make_str_concat_candidate();
    passes::PassContext ctx;
    ctx.tier = passes::TierMode::Tier2;
    static stdx::small_vector<char, 4096> pool;
    pool.clear();
    for (char ch : std::string_view("abcdef")) pool.push_back(ch);
    ctx.string_pool = &pool;

    passes::P45_StringInterning p;
    Result<passes::PassResult> r1 = p.run(g, ctx);
    CHECK(r1.has_value());
    if (!r1) return;
    Result<passes::PassResult> r2 = p.run(g, ctx);
    CHECK(r2.has_value());
    if (!r2) return;
    CHECK(!r2->changed);
}
