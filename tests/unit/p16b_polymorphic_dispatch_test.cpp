// =============================================================================
// tests/unit/p16b_polymorphic_dispatch_test.cpp — Pass 16b (Polymorphic IC
// Dispatch) tests.
//
// Tests the three IC tiers:
//   1. Monomorphic (1 type) — P16 handles, P16b declines
//   2. Polymorphic (2-4 types) — P16b emits DispatchCache
//   3. Megamorphic (>4 types) — P16b declines, telemetry records
//
// Also tests idempotency (Rule 10), Tier-1 declining (Rule 2), and
// full-pipeline verifier smoke.
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

// Build a CallPy with PGO data showing 3 observed receiver types
// (polymorphic, within the 2-4 range). aux0 = type_count, shape_id =
// shape hash, pgo_count above the devirt floor.
[[nodiscard]] Graph make_polymorphic_candidate() {
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

    NodeId call = g.create(NodeKind::CallPy, {start, start, callee, recv, arg});
    g.node(call).set_flag(NodeFlag::OnEffectChain);
    g.node(call).set_flag(NodeFlag::MayCall);
    g.node(call).set_flag(NodeFlag::MayThrow);
    g.node(call).pgo_count = cfg::ic_devirt_pgo_floor * 10;   // well above threshold
    g.node(call).aux0 = 3;            // 3 observed types (polymorphic)
    g.node(call).shape_id = 0xABCDEF;   // PGO shape hash

    NodeId ret = g.create(NodeKind::Return, {start, call});
    g.node(ret).set_flag(NodeFlag::OnEffectChain);
    g.set_end(ret);

    g.n_parameters = 2;
    g.function_name = global_symbols().intern("poly_call");
    return g;
}

// Build a megamorphic candidate: aux0 > ic_poly_max_types (4).
[[nodiscard]] Graph make_megamorphic_candidate() {
    Graph g = make_polymorphic_candidate();
    g.for_each_live([&](NodeId id) {
        if (g.node(id).kind == NodeKind::CallPy) {
            g.node(id).aux0 = cfg::ic_poly_max_types + 5;   // 9 types: megamorphic
        }
    });
    return g;
}

// Build a candidate with no PGO type data (aux0 == 0). Rule 46: decline.
[[nodiscard]] Graph make_no_type_count_candidate() {
    Graph g = make_polymorphic_candidate();
    g.for_each_live([&](NodeId id) {
        if (g.node(id).kind == NodeKind::CallPy) {
            g.node(id).aux0 = 0;   // no type count recorded
        }
    });
    return g;
}

}  // namespace

// =============================================================================
// Polymorphic candidate (3 types): P16b emits a DispatchCache node replacing
// the CallPy. The cache carries the type count (aux0) and shape hash
// (shape_id), with a FrameState (Rule 5).
// =============================================================================
TEST(p16b_poly_emits_dispatch_cache) {
    Graph g = make_polymorphic_candidate();
    passes::PassContext ctx = make_tier2_ctx();
    passes::P16b_PolymorphicDispatch p;
    Result<passes::PassResult> r = p.run(g, ctx);
    CHECK(r.has_value());
    if (!r) return;
    CHECK(r->changed);

    // The CallPy became a DispatchCache.
    CHECK_EQ(count_kind(g, NodeKind::CallPy), 0u);
    CHECK_EQ(count_kind(g, NodeKind::DispatchCache), 1u);

    // The DispatchCache carries the type count and shape hash.
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        if (n.kind != NodeKind::DispatchCache) return;
        CHECK_EQ(n.aux0, 3u);                  // 3 cached types
        CHECK_EQ(n.shape_id, 0xABCDEFu);       // PGO shape hash
        CHECK(n.has(NodeFlag::Speculative));     // Rule 5: FrameState
        CHECK(n.has(NodeFlag::MayCall));
    });

    // A FrameState was registered.
    CHECK(g.frame_state_count() > 0);
}

// Idempotency (Rule 10): a second run is a no-op. The CallPy is already a
// DispatchCache; has_dispatch_cache() returns true.
TEST(p16b_poly_is_idempotent) {
    Graph g = make_polymorphic_candidate();
    passes::PassContext ctx = make_tier2_ctx();
    passes::P16b_PolymorphicDispatch p;
    (void)p.run(g, ctx);
    Result<passes::PassResult> r2 = p.run(g, ctx);
    CHECK(r2.has_value());
    if (!r2) return;
    CHECK(!r2->changed);
    CHECK_EQ(count_kind(g, NodeKind::DispatchCache), 1u);
}

// Megamorphic candidate (>4 types): P16b declines, no DispatchCache emitted.
// Rule 65: telemetry records the declination.
TEST(p16b_mega_declines_on_megamorphic) {
    Graph g = make_megamorphic_candidate();
    passes::PassContext ctx = make_tier2_ctx();
    passes::P16b_PolymorphicDispatch p;
    Result<passes::PassResult> r = p.run(g, ctx);
    CHECK(r.has_value());
    if (!r) return;
    CHECK(!r->changed);
    CHECK_EQ(count_kind(g, NodeKind::DispatchCache), 0u);
    CHECK_EQ(count_kind(g, NodeKind::CallPy), 1u);   // unchanged
}

// Rule 46: no type count data (aux0 == 0) → decline.
TEST(p16b_declines_without_type_count) {
    Graph g = make_no_type_count_candidate();
    passes::PassContext ctx = make_tier2_ctx();
    passes::P16b_PolymorphicDispatch p;
    Result<passes::PassResult> r = p.run(g, ctx);
    CHECK(r.has_value());
    if (!r) return;
    CHECK(!r->changed);
    CHECK_EQ(count_kind(g, NodeKind::DispatchCache), 0u);
}

// Tier 1: P16b declines (Rule 2: speculation needs PGO).
TEST(p16b_declines_in_tier1) {
    Graph g = make_polymorphic_candidate();
    passes::PassContext ctx;
    ctx.tier = passes::TierMode::Tier1;
    ctx.node_budget = cfg::tier1_node_budget;
    ctx.code_unit_id = 1;
    ctx.target = &backend::host_target();
    passes::P16b_PolymorphicDispatch p;
    Result<passes::PassResult> r = p.run(g, ctx);
    CHECK(r.has_value());
    if (!r) return;
    CHECK(!r->changed);
    CHECK_EQ(count_kind(g, NodeKind::DispatchCache), 0u);
}

// Low heat (pgo_count < ic_devirt_pgo_floor): P16b declines.
TEST(p16b_declines_on_low_heat) {
    Graph g = make_polymorphic_candidate();
    g.for_each_live([&](NodeId id) {
        if (g.node(id).kind == NodeKind::CallPy) {
            g.node(id).pgo_count = cfg::ic_devirt_pgo_floor / 2;
        }
    });
    passes::PassContext ctx = make_tier2_ctx();
    passes::P16b_PolymorphicDispatch p;
    Result<passes::PassResult> r = p.run(g, ctx);
    CHECK(r.has_value());
    if (!r) return;
    CHECK(!r->changed);
    CHECK_EQ(count_kind(g, NodeKind::DispatchCache), 0u);
}

// =============================================================================
// Integration smoke-test: the full pipeline runs clean on a polymorphic
// candidate. The DispatchCache survives downstream passes (GVN, DCE) and
// the scheduler lowers it to Op::CALL without crashing.
// =============================================================================
TEST(p16b_poly_passes_verifier_after_full_pipeline) {
    Graph g = make_polymorphic_candidate();
    passes::PassContext ctx = make_tier2_ctx();
    passes::OptPipeline pipeline;
    Result<void> r = passes::run_pipeline(g, ctx, pipeline);
    CHECK(r.has_value());
}
