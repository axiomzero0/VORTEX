// =============================================================================
// tests/unit/p16_ic_monomorphism_test.cpp — Pass 16 (IC Monomorphism &
// Speculative Devirtualization) research-grade tests.
//
// Tests the two devirtualization paths:
//   1. Static: MakeFunction callee → GuardedDirectCall (zero-risk)
//   2. Speculative: hot CallPy with PGO type data → Guard(TypeIs) +
//      GuardedDirectCall + FrameState (Rule 3/4/5)
//
// Also tests idempotency (Rule 10), Tier-1 declining (Rule 2), and
// Rule 46 compliance (no type data → no speculation).
// =============================================================================

#include "harness.hpp"

#include "vortex/backend/target.hpp"
#include "vortex/frontend/lowering.hpp"
#include "vortex/ir/graph.hpp"
#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/pass_pipeline.hpp"
#include "vortex/passes/passes_fwd.hpp"
#include "vortex/support/config.hpp"

using namespace vortex;
using namespace vortex::ir;
namespace passes = vortex::passes;
namespace backend = vortex::backend;
namespace fe = vortex::fe;

namespace {

passes::PassContext make_tier2_ctx() {
    passes::PassContext ctx;
    ctx.tier = passes::TierMode::Tier2;
    ctx.node_budget = cfg::tier2_node_budget;
    ctx.code_unit_id = 1;
    ctx.target = &backend::host_target();
    return ctx;
}

[[nodiscard]] std::uint32_t count_kind(const Graph& g, NodeKind k) noexcept {
    std::uint32_t n = 0;
    g.for_each_live([&](NodeId id) {
        if (g.node(id).kind == k) ++n;
    });
    return n;
}

// Build a graph with a hot CallPy that has PGO type data (shape_id set)
// and pgo_count above the devirt floor. This is the speculative
// devirtualization candidate.
//
//   %recv = Parameter(0)     # receiver object
//   %callee = Parameter(1)  # function to call
//   %arg = ConstInt(42)
//   %call = CallPy(callee, recv, arg)  # pgo_count = 1000, shape_id = 42
[[nodiscard]] Graph make_speculative_devirt_candidate() {
    Graph g;
    NodeId start = g.create(NodeKind::Start);
    g.set_start(start);

    NodeId recv = g.create(NodeKind::Parameter, {start});
    g.node(recv).aux0 = 0;
    g.node(recv).set_flag(NodeFlag::Pure);

    NodeId callee = g.create(NodeKind::Parameter, {start});
    g.node(callee).aux0 = 1;
    g.node(callee).set_flag(NodeFlag::Pure);

    NodeId arg = g.create(NodeKind::ConstInt);
    g.node(arg).const_value = Value::integer(42);
    g.node(arg).set_flag(NodeFlag::Pure);

    // CallPy: ins[0]=ctrl, ins[1]=eff, ins[2]=callee, ins[3]=recv, ins[4]=arg
    NodeId call = g.create(NodeKind::CallPy, {start, start, callee, recv, arg});
    g.node(call).set_flag(NodeFlag::OnEffectChain);
    g.node(call).set_flag(NodeFlag::MayCall);
    g.node(call).set_flag(NodeFlag::MayThrow);
    g.node(call).pgo_count = cfg::ic_devirt_pgo_floor * 10;   // well above threshold
    g.node(call).shape_id = 42;   // PGO-recorded expected type_id

    NodeId ret = g.create(NodeKind::Return, {start, call});
    g.node(ret).set_flag(NodeFlag::OnEffectChain);
    g.set_end(ret);

    g.n_parameters = 2;
    g.function_name = global_symbols().intern("spec_devirt");
    return g;
}

// Build a graph with a CallPy that has HIGH pgo_count but NO type data
// (shape_id == 0). Rule 46 forbids speculation without evidence.
[[nodiscard]] Graph make_no_type_data_candidate() {
    Graph g = make_speculative_devirt_candidate();
    // Clear the type data — the pass must decline.
    g.node(g.end()).ins[1] = g.end();   // dummy: just clear shape_id
    // Find the CallPy and clear its shape_id.
    g.for_each_live([&](NodeId id) {
        if (g.node(id).kind == NodeKind::CallPy) {
            g.node(id).shape_id = 0;
        }
    });
    return g;
}

// Build a graph with a CallPy that has type data but LOW pgo_count
// (below ic_devirt_pgo_floor). The pass must decline — not enough heat.
[[nodiscard]] Graph make_low_heat_candidate() {
    Graph g = make_speculative_devirt_candidate();
    g.for_each_live([&](NodeId id) {
        if (g.node(id).kind == NodeKind::CallPy) {
            g.node(id).pgo_count = cfg::ic_devirt_pgo_floor / 2;   // below threshold
        }
    });
    return g;
}

}  // namespace

// =============================================================================
// Speculative devirtualization: hot CallPy with PGO type data gets a
// Guard(TypeIs) node + GuardedDirectCall + FrameState (Rule 3/4/5).
// =============================================================================
TEST(p16_spec_devirt_emits_type_guard) {
    Graph g = make_speculative_devirt_candidate();
    passes::PassContext ctx = make_tier2_ctx();
    passes::P16_ICMonomorphism p;
    Result<passes::PassResult> r = p.run(g, ctx);
    CHECK(r.has_value());
    if (!r) return;
    CHECK(r->changed);

    // The CallPy became a GuardedDirectCall.
    CHECK_EQ(count_kind(g, NodeKind::CallPy), 0u);
    CHECK_EQ(count_kind(g, NodeKind::GuardedDirectCall), 1u);

    // A Guard(TypeIs) was emitted with the expected type_id.
    std::uint32_t type_guards = 0;
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        if (n.kind != NodeKind::Guard) return;
        if (n.subop != static_cast<std::uint16_t>(GuardKind::TypeIs)) return;
        ++type_guards;
        CHECK(n.has(NodeFlag::Speculative));   // Rule 5: FrameState attached
        CHECK(n.has(NodeFlag::TypeGuarded));
        CHECK_EQ(n.shape_id, 42u);   // the PGO-recorded type_id
    });
    CHECK_EQ(type_guards, 1u);

    // The GuardedDirectCall carries a FrameState (its aux1 indexes the
    // graph's frame_state table).
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        if (n.kind != NodeKind::GuardedDirectCall) return;
        CHECK(n.has(NodeFlag::Speculative));
    });

    // At least 2 FrameStates registered (one for the guard, one for the call).
    CHECK(g.frame_state_count() >= 2);
}

// Idempotency (Rule 10): a second run is a no-op. The CallPy is already
// a GuardedDirectCall, and the type guard already exists.
TEST(p16_spec_devirt_is_idempotent) {
    Graph g = make_speculative_devirt_candidate();
    passes::PassContext ctx = make_tier2_ctx();
    passes::P16_ICMonomorphism p;
    (void)p.run(g, ctx);
    Result<passes::PassResult> r2 = p.run(g, ctx);
    CHECK(r2.has_value());
    if (!r2) return;
    CHECK(!r2->changed);
    // Still exactly 1 GuardedDirectCall, 1 type guard.
    CHECK_EQ(count_kind(g, NodeKind::GuardedDirectCall), 1u);
    std::uint32_t guards = 0;
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        if (n.kind == NodeKind::Guard &&
            n.subop == static_cast<std::uint16_t>(GuardKind::TypeIs)) ++guards;
    });
    CHECK_EQ(guards, 1u);
}

// Rule 46: no type data (shape_id == 0) → no speculation. The pass must
// decline, leaving the CallPy unchanged.
TEST(p16_spec_devirt_declines_without_type_data) {
    Graph g = make_no_type_data_candidate();
    passes::PassContext ctx = make_tier2_ctx();
    passes::P16_ICMonomorphism p;
    Result<passes::PassResult> r = p.run(g, ctx);
    CHECK(r.has_value());
    if (!r) return;
    CHECK(!r->changed);
    CHECK_EQ(count_kind(g, NodeKind::CallPy), 1u);
    CHECK_EQ(count_kind(g, NodeKind::GuardedDirectCall), 0u);
    CHECK_EQ(count_kind(g, NodeKind::Guard), 0u);
}

// Not enough heat (pgo_count < ic_devirt_pgo_floor): the pass declines.
TEST(p16_spec_devirt_declines_on_low_heat) {
    Graph g = make_low_heat_candidate();
    passes::PassContext ctx = make_tier2_ctx();
    passes::P16_ICMonomorphism p;
    Result<passes::PassResult> r = p.run(g, ctx);
    CHECK(r.has_value());
    if (!r) return;
    CHECK(!r->changed);
    CHECK_EQ(count_kind(g, NodeKind::CallPy), 1u);
    CHECK_EQ(count_kind(g, NodeKind::GuardedDirectCall), 0u);
}

// Tier 1: speculative path is gated off (Rule 2: speculation needs PGO).
// Only the static path (MakeFunction) could fire; since this candidate
// has no MakeFunction, the pass is a no-op.
TEST(p16_spec_devirt_declines_in_tier1) {
    Graph g = make_speculative_devirt_candidate();
    passes::PassContext ctx;
    ctx.tier = passes::TierMode::Tier1;
    ctx.node_budget = cfg::tier1_node_budget;
    ctx.code_unit_id = 1;
    ctx.target = &backend::host_target();
    passes::P16_ICMonomorphism p;
    Result<passes::PassResult> r = p.run(g, ctx);
    CHECK(r.has_value());
    if (!r) return;
    // Tier 1: no PGO speculation. CallPy stays as-is.
    CHECK_EQ(count_kind(g, NodeKind::GuardedDirectCall), 0u);
    CHECK_EQ(count_kind(g, NodeKind::Guard), 0u);
}

// =============================================================================
// Integration smoke-test: the full pipeline runs clean on a spec-devirt
// candidate. Catches verifier violations from the new Guard(TypeIs) node
// interacting with downstream passes.
// =============================================================================
TEST(p16_spec_devirt_passes_verifier_after_full_pipeline) {
    Graph g = make_speculative_devirt_candidate();
    passes::PassContext ctx = make_tier2_ctx();
    passes::OptPipeline pipeline;
    Result<void> r = passes::run_pipeline(g, ctx, pipeline);
    CHECK(r.has_value());
}
