// =============================================================================
// tests/unit/p31_slp_test.cpp — Pass 31 (SLP Vectorization) research-grade
// tests for the four sub-passes: 31a (unbox mark), 31b (cost-model packetize),
// 31c (speculative AliasDisjoint guard emission), 31d (gather/scatter fallback).
//
// These tests pin the new research-grade behaviors described in the VORTEX
// design doc: cost-model rejection (Rule 45), speculative SLP with hardware
// guards (Rules 3/4/5), and the gather/scatter fallback for pointer-array
// LoadIndex groups. Each test constructs a synthetic IR graph that exercises
// one specific sub-pass, runs P31 in isolation, and asserts the IR-level
// observable.
// =============================================================================

#include "harness.hpp"

#include "vortex/backend/target.hpp"
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

// Build a Tier-2 PassContext with the host's real target descriptor —
// the pass queries simd_width_bytes and feature flags from it.
passes::PassContext make_tier2_ctx() {
    passes::PassContext ctx;
    ctx.tier = passes::TierMode::Tier2;
    ctx.node_budget = cfg::tier2_node_budget;
    ctx.code_unit_id = 1;
    ctx.target = &backend::host_target();
    return ctx;
}

// Count nodes of a specific kind in the graph.
[[nodiscard]] std::uint32_t count_kind(const Graph& g, NodeKind k) noexcept {
    std::uint32_t n = 0;
    g.for_each_live([&](NodeId id) {
        if (g.node(id).kind == k) ++n;
    });
    return n;
}

// Count nodes that have a specific flag.
[[nodiscard]] std::uint32_t count_flag(const Graph& g, NodeFlag f) noexcept {
    std::uint32_t n = 0;
    g.for_each_live([&](NodeId id) {
        if (g.node(id).has(f)) ++n;
    });
    return n;
}

// Build a graph with two isomorphic adjacent PyBinary(Mul) chains whose
// operands are ConstInt (unboxable). Mul has latency 3 (CostClass::Mul,
// queried from TargetDescriptor), so vector_pays(2, 3) = (3*2 > 2+2) =
// (6 > 4) = TRUE — the cost model (Rule 45) accepts this packet.
//
// Chain A:  a1 = c1 * c2
// Chain B:  b1 = c3 * c4
// Packet:  VecOp { a1, b1 }
[[nodiscard]] Graph make_slp_candidate() {
    Graph g;
    NodeId start = g.create(NodeKind::Start);
    g.set_start(start);

    auto mk_const = [&](std::int64_t v) {
        NodeId c = g.create(NodeKind::ConstInt);
        g.node(c).const_value = Value::integer(v);
        g.node(c).set_flag(NodeFlag::Pure);
        return c;
    };
    auto mk_mul = [&](NodeId lhs, NodeId rhs) {
        NodeId n = g.create(NodeKind::PyBinary, {start, start, lhs, rhs});
        g.node(n).subop = static_cast<std::uint16_t>(BinOpKind::Mul);
        g.node(n).set_flag(NodeFlag::Pure);
        return n;
    };

    NodeId c1 = mk_const(1), c2 = mk_const(2);
    NodeId c3 = mk_const(3), c4 = mk_const(4);

    NodeId a1 = mk_mul(c1, c2);
    NodeId b1 = mk_mul(c3, c4);

    NodeId ret = g.create(NodeKind::Return, {start, a1});
    g.node(ret).set_flag(NodeFlag::OnEffectChain);
    g.set_end(ret);
    g.add_input(ret, b1);

    g.function_name = global_symbols().intern("slp_candidate");
    return g;
}

// Build a graph with a TRIVIAL 2-candidate SLP packet: just one PyBinary
// per chain, two chains. The cost model (Rule 45) must REJECT this —
// the insert/extract overhead (2 moves) equals the scalar cost (2 ops),
// so SLP doesn't pay. vector_pays(2, 1) returns false on every target.
[[nodiscard]] Graph make_trivial_slp_candidate() {
    Graph g;
    NodeId start = g.create(NodeKind::Start);
    g.set_start(start);

    NodeId c1 = g.create(NodeKind::ConstInt);
    g.node(c1).const_value = Value::integer(1);
    g.node(c1).set_flag(NodeFlag::Pure);
    NodeId c2 = g.create(NodeKind::ConstInt);
    g.node(c2).const_value = Value::integer(2);
    g.node(c2).set_flag(NodeFlag::Pure);
    NodeId c3 = g.create(NodeKind::ConstInt);
    g.node(c3).const_value = Value::integer(3);
    g.node(c3).set_flag(NodeFlag::Pure);
    NodeId c4 = g.create(NodeKind::ConstInt);
    g.node(c4).const_value = Value::integer(4);
    g.node(c4).set_flag(NodeFlag::Pure);

    NodeId a = g.create(NodeKind::PyBinary, {start, start, c1, c2});
    g.node(a).subop = static_cast<std::uint16_t>(BinOpKind::Add);
    g.node(a).set_flag(NodeFlag::Pure);
    NodeId b = g.create(NodeKind::PyBinary, {start, start, c3, c4});
    g.node(b).subop = static_cast<std::uint16_t>(BinOpKind::Add);
    g.node(b).set_flag(NodeFlag::Pure);

    NodeId ret = g.create(NodeKind::Return, {start, a});
    g.node(ret).set_flag(NodeFlag::OnEffectChain);
    g.set_end(ret);
    g.add_input(ret, b);

    g.function_name = global_symbols().intern("trivial_slp");
    return g;
}

// Build a graph with two hot LoadIndex nodes whose bases differ — the
// 31c speculative-SLP candidate. Both loads carry pgo_count above the
// Tier2 IC threshold.
[[nodiscard]] Graph make_speculative_slp_candidate() {
    Graph g;
    NodeId start = g.create(NodeKind::Start);
    g.set_start(start);

    NodeId list_a = g.create(NodeKind::NewList, {start});
    g.node(list_a).set_flag(NodeFlag::OnEffectChain);
    NodeId list_b = g.create(NodeKind::NewList, {start});
    g.node(list_b).set_flag(NodeFlag::OnEffectChain);

    NodeId idx0 = g.create(NodeKind::ConstInt);
    g.node(idx0).const_value = Value::integer(0);
    g.node(idx0).set_flag(NodeFlag::Pure);
    NodeId idx1 = g.create(NodeKind::ConstInt);
    g.node(idx1).const_value = Value::integer(1);
    g.node(idx1).set_flag(NodeFlag::Pure);

    // LoadIndex: ins[0]=control, ins[1]=effect, ins[2]=base, ins[3]=index
    NodeId load_a = g.create(NodeKind::LoadIndex, {start, list_a, list_a, idx0});
    g.node(load_a).set_flag(NodeFlag::OnEffectChain);
    g.node(load_a).set_flag(NodeFlag::MayThrow);
    g.node(load_a).pgo_count = cfg::slp_alias_guard_pgo_floor * 10;   // well above threshold

    NodeId load_b = g.create(NodeKind::LoadIndex, {start, list_b, list_b, idx1});
    g.node(load_b).set_flag(NodeFlag::OnEffectChain);
    g.node(load_b).set_flag(NodeFlag::MayThrow);
    g.node(load_b).pgo_count = cfg::slp_alias_guard_pgo_floor * 10;   // well above threshold

    NodeId ret = g.create(NodeKind::Return, {start, load_a});
    g.node(ret).set_flag(NodeFlag::OnEffectChain);
    g.set_end(ret);
    g.add_input(ret, load_b);

    g.function_name = global_symbols().intern("spec_slp");
    return g;
}

// Build a graph with 3 LoadIndex nodes sharing the same base — the 31d
// gather/scatter candidate. Base is a NewList (no Unboxed flag set).
[[nodiscard]] Graph make_gather_candidate() {
    Graph g;
    NodeId start = g.create(NodeKind::Start);
    g.set_start(start);

    NodeId list_a = g.create(NodeKind::NewList, {start});
    g.node(list_a).set_flag(NodeFlag::OnEffectChain);

    NodeId idx0 = g.create(NodeKind::ConstInt);
    g.node(idx0).const_value = Value::integer(0);
    g.node(idx0).set_flag(NodeFlag::Pure);
    NodeId idx1 = g.create(NodeKind::ConstInt);
    g.node(idx1).const_value = Value::integer(1);
    g.node(idx1).set_flag(NodeFlag::Pure);
    NodeId idx2 = g.create(NodeKind::ConstInt);
    g.node(idx2).const_value = Value::integer(2);
    g.node(idx2).set_flag(NodeFlag::Pure);

    NodeId l0 = g.create(NodeKind::LoadIndex, {start, list_a, list_a, idx0});
    g.node(l0).set_flag(NodeFlag::OnEffectChain);
    g.node(l0).set_flag(NodeFlag::MayThrow);
    NodeId l1 = g.create(NodeKind::LoadIndex, {start, list_a, list_a, idx1});
    g.node(l1).set_flag(NodeFlag::OnEffectChain);
    g.node(l1).set_flag(NodeFlag::MayThrow);
    NodeId l2 = g.create(NodeKind::LoadIndex, {start, list_a, list_a, idx2});
    g.node(l2).set_flag(NodeFlag::OnEffectChain);
    g.node(l2).set_flag(NodeFlag::MayThrow);

    NodeId ret = g.create(NodeKind::Return, {start, l0});
    g.node(ret).set_flag(NodeFlag::OnEffectChain);
    g.set_end(ret);
    g.add_input(ret, l1);
    g.add_input(ret, l2);

    g.function_name = global_symbols().intern("gather");
    return g;
}

}  // namespace

// =============================================================================
// 31b: cost-model packetization fires on isomorphic adjacent PyBinary(Add)
// chains with unboxable operands. The VecOp node is emitted, the members'
// Vectorizable flag is cleared (idempotency). This is the happy path that
// proves the new code preserves the original behavior.
// =============================================================================
TEST(p31_slp_packetizes_isomorphic_pybinary_chain) {
    Graph g = make_slp_candidate();
    passes::PassContext ctx = make_tier2_ctx();
    passes::P31_SLPVectorization p;
    Result<passes::PassResult> r = p.run(g, ctx);
    CHECK(r.has_value());
    if (!r) return;
    // The host target has at least SSE2/ASIMD (16-byte SIMD width) — the
    // packet (2 lanes) is below the cost-model gate.
    CHECK_EQ(count_kind(g, NodeKind::VecOp), 1u);
    CHECK(r->changed);
}

// Idempotency (Rule 10): a second run is a no-op because the candidates'
// Vectorizable flags were cleared.
TEST(p31_slp_packetize_is_idempotent) {
    Graph g = make_slp_candidate();
    passes::PassContext ctx = make_tier2_ctx();
    passes::P31_SLPVectorization p;
    (void)p.run(g, ctx);
    Result<passes::PassResult> r2 = p.run(g, ctx);
    CHECK(r2.has_value());
    if (!r2) return;
    CHECK(!r2->changed);
    CHECK_EQ(count_kind(g, NodeKind::VecOp), 1u);   // still exactly one packet
}

// =============================================================================
// 31b cost-model rejection: a TRIVIAL 2-lane packet with 1 op per chain
// does NOT pay on any target (vector_pays(2,1) returns false because the
// 2 lane moves equal the 2 scalar ops). The pass must decline to emit a
// VecOp — proving the cost model is actually consulted, not bypassed.
// =============================================================================
TEST(p31_slp_cost_model_rejects_trivial_packet) {
    Graph g = make_trivial_slp_candidate();
    passes::PassContext ctx = make_tier2_ctx();
    passes::P31_SLPVectorization p;
    Result<passes::PassResult> r = p.run(g, ctx);
    CHECK(r.has_value());
    if (!r) return;
    // Cost model rejected: no VecOp emitted, no change reported.
    CHECK(!r->changed);
    CHECK_EQ(count_kind(g, NodeKind::VecOp), 0u);
}

// =============================================================================
// 31c: Speculative SLP emits a Guard(AliasDisjoint) node when two hot
// LoadIndex nodes have different bases and no static alias proof.
// Tier 2 only (Rule 2: speculation needs PGO).
// =============================================================================
TEST(p31_slp_emits_alias_disjoint_guard_for_hot_load_pair) {
    Graph g = make_speculative_slp_candidate();
    passes::PassContext ctx = make_tier2_ctx();
    passes::P31_SLPVectorization p;
    Result<passes::PassResult> r = p.run(g, ctx);
    CHECK(r.has_value());
    if (!r) return;
    CHECK(r->changed);

    // The Guard node exists with subop == AliasDisjoint and a Speculative
    // flag (Rule 5: FrameState attached).
    std::uint32_t guards = 0;
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        if (n.kind != NodeKind::Guard) return;
        if (n.subop == static_cast<std::uint16_t>(GuardKind::AliasDisjoint)) {
            ++guards;
            CHECK(n.has(NodeFlag::Speculative));
        }
    });
    CHECK_EQ(guards, 1u);

    // Both loads are now TypeGuarded (the guard's "proof" is the runtime
    // check — after the guard fires, the loads cannot alias).
    CHECK_EQ(count_flag(g, NodeFlag::TypeGuarded), 2u);

    // A FrameState was registered with the graph (Rule 5).
    CHECK(g.frame_state_count() > 0);
}

// 31c is idempotent: a second run sees both loads TypeGuarded and does not
// emit a second guard.
TEST(p31_slp_speculative_guard_is_idempotent) {
    Graph g = make_speculative_slp_candidate();
    passes::PassContext ctx = make_tier2_ctx();
    passes::P31_SLPVectorization p;
    (void)p.run(g, ctx);
    Result<passes::PassResult> r2 = p.run(g, ctx);
    CHECK(r2.has_value());
    if (!r2) return;
    CHECK(!r2->changed);

    std::uint32_t guards = 0;
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        if (n.kind == NodeKind::Guard &&
            n.subop == static_cast<std::uint16_t>(GuardKind::AliasDisjoint)) {
            ++guards;
        }
    });
    CHECK_EQ(guards, 1u);
}

// 31c declines in Tier 1 (no PGO, no speculation per Rule 2).
TEST(p31_slp_speculative_guard_declines_in_tier1) {
    Graph g = make_speculative_slp_candidate();
    passes::PassContext ctx;
    ctx.tier = passes::TierMode::Tier1;
    ctx.node_budget = cfg::tier1_node_budget;
    ctx.code_unit_id = 1;
    ctx.target = &backend::host_target();
    passes::P31_SLPVectorization p;
    Result<passes::PassResult> r = p.run(g, ctx);
    CHECK(r.has_value());
    if (!r) return;
    // Tier 1 self-gates: the entire pass returns early. No guards, no
    // packets, no gathers.
    CHECK(!r->changed);
    CHECK_EQ(count_kind(g, NodeKind::Guard), 0u);
}

// =============================================================================
// 31d: gather/scatter fallback for pointer-array LoadIndex groups.
// When 3+ LoadIndex nodes share the same base (a boxed list — no Unboxed
// flag on the base), a Gather node is emitted carrying the lane count,
// and each LoadIndex is replaced by a VecExtract(Gather, lane_id).
//
// Gated on the target having AVX2 (x86) or ASIMD (aarch64). The host
// target always has at least the baseline (ASIMD on arm64, SSE2 on x86-
// 64 — the SSE2 baseline does NOT include AVX2). This test gracefully
// skips on x86 hosts without AVX2 rather than failing.
// =============================================================================
TEST(p31_slp_emits_gather_for_pointer_array_loads) {
    Graph g = make_gather_candidate();
    passes::PassContext ctx = make_tier2_ctx();

    // Skip on x86-64 hosts without AVX2 — the cost model correctly
    // declines to emit gather (Rule 45), and the test cannot assert
    // what it's testing. On aarch64, ASIMD is always present.
    const bool fast_gather_supported =
        ctx.target->has(backend::TargetFeature::AVX2) ||
        ctx.target->has(backend::TargetFeature::ASIMD);
    if (!fast_gather_supported) {
        // The pass still runs without faulting — it just declines.
        passes::P31_SLPVectorization p;
        Result<passes::PassResult> r = p.run(g, ctx);
        CHECK(r.has_value());
        return;
    }

    passes::P31_SLPVectorization p;
    Result<passes::PassResult> r = p.run(g, ctx);
    CHECK(r.has_value());
    if (!r) return;
    CHECK(r->changed);

    // One Gather node for the (list_a, [l0,l1,l2]) group.
    CHECK_EQ(count_kind(g, NodeKind::Gather), 1u);
    // Three VecExtract projections (one per lane).
    CHECK_EQ(count_kind(g, NodeKind::VecExtract), 3u);
}

// 31d is idempotent: a second run sees the existing Gather node (via
// has_gather_for_base) and declines to emit a duplicate.
TEST(p31_slp_gather_is_idempotent) {
    Graph g = make_gather_candidate();
    passes::PassContext ctx = make_tier2_ctx();
    const bool fast_gather_supported =
        ctx.target->has(backend::TargetFeature::AVX2) ||
        ctx.target->has(backend::TargetFeature::ASIMD);
    if (!fast_gather_supported) return;   // nothing to idempotency-check

    passes::P31_SLPVectorization p;
    (void)p.run(g, ctx);
    Result<passes::PassResult> r2 = p.run(g, ctx);
    CHECK(r2.has_value());
    if (!r2) return;
    CHECK(!r2->changed);
    CHECK_EQ(count_kind(g, NodeKind::Gather), 1u);   // still exactly one
}

// =============================================================================
// 31b cost-model: when the target's vector width is too narrow for the
// packet to pay, the pass declines. We simulate this by constructing a
// TargetDescriptor with simd_width_bytes=0 — the pass returns early
// before any packet is emitted.
// =============================================================================
TEST(p31_slp_declines_when_target_has_no_simd) {
    Graph g = make_slp_candidate();
    passes::PassContext ctx = make_tier2_ctx();

    // Construct a deliberately-useless descriptor: zero SIMD width.
    backend::TargetDescriptor no_simd{};
    no_simd.architecture = backend::compiled_arch();
    no_simd.simd_width_bytes = 0;   // forces early return
    ctx.target = &no_simd;

    passes::P31_SLPVectorization p;
    Result<passes::PassResult> r = p.run(g, ctx);
    CHECK(r.has_value());
    if (!r) return;
    CHECK(!r->changed);
    CHECK_EQ(count_kind(g, NodeKind::VecOp), 0u);
    CHECK_EQ(count_kind(g, NodeKind::Guard), 0u);
    CHECK_EQ(count_kind(g, NodeKind::Gather), 0u);
}

// =============================================================================
// Integration smoke-test: the full pipeline runs clean on a graph that
// exercises P31's new behaviors. Catches verifier violations introduced
// by the new node shapes (Guard with FrameState, Gather with effect-chain).
// =============================================================================
TEST(p31_slp_passes_verifier_after_full_pipeline) {
    Graph g = make_speculative_slp_candidate();
    passes::PassContext ctx = make_tier2_ctx();
    passes::OptPipeline pipeline;
    Result<void> r = passes::run_pipeline(g, ctx, pipeline);
    CHECK(r.has_value());
    // No crashes, no diagnostics — the new Guard/Gather nodes survive
    // the rest of the pipeline (LateGVN, GlobalDCE, etc.) without violating
    // the verifier (Rule 40).
}
