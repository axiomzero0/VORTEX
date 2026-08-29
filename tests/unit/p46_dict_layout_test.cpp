// =============================================================================
// tests/unit/p46_dict_layout_test.cpp — Pass 46 (Dict-to-Struct) research-grade
// tests for D2S layout specialization: shape guard emission, store rewrite,
// load rewrite, construction store rewrite, idempotency, and blocking on
// iteration/len/call operations.
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

// Build a graph with a NewDict, 3 constant-key construction stores, and
// 2 constant-key loads. This is the canonical D2S candidate: 3 keys
// (above cfg::dict_layout_min_fields=2), no iteration/len/call.
//
//   d = {}
//   d["a"] = 1   (construction store 1)
//   d["b"] = 2   (construction store 2)
//   d["c"] = 3   (construction store 3)
//   x = d["a"]   (load 1)
//   y = d["c"]   (load 2)
[[nodiscard]] Graph make_d2s_candidate() {
    Graph g;
    NodeId start = g.create(NodeKind::Start);
    g.set_start(start);

    NodeId d = g.create(NodeKind::NewDict, {start});
    g.node(d).set_flag(NodeFlag::OnEffectChain);

    auto mk_const_str = [&](std::uint32_t sym) {
        NodeId c = g.create(NodeKind::ConstPy);
        g.node(c).const_value.tag = Tag::None;
        g.node(c).symbol = sym;
        g.node(c).set_flag(NodeFlag::Pure);
        return c;
    };
    auto mk_const_int = [&](std::int64_t v) {
        NodeId c = g.create(NodeKind::ConstInt);
        g.node(c).const_value = Value::integer(v);
        g.node(c).set_flag(NodeFlag::Pure);
        return c;
    };

    NodeId key_a = mk_const_str(100);
    NodeId key_b = mk_const_str(200);
    NodeId key_c = mk_const_str(300);
    NodeId val1 = mk_const_int(1);
    NodeId val2 = mk_const_int(2);
    NodeId val3 = mk_const_int(3);

    // StoreIndex: ins[0]=ctrl, ins[1]=eff, ins[2]=base, ins[3]=key, ins[4]=val
    NodeId s1 = g.create(NodeKind::StoreIndex, {start, d, d, key_a, val1});
    g.node(s1).set_flag(NodeFlag::OnEffectChain);
    g.node(s1).set_flag(NodeFlag::MayThrow);
    NodeId s2 = g.create(NodeKind::StoreIndex, {start, s1, d, key_b, val2});
    g.node(s2).set_flag(NodeFlag::OnEffectChain);
    g.node(s2).set_flag(NodeFlag::MayThrow);
    NodeId s3 = g.create(NodeKind::StoreIndex, {start, s2, d, key_c, val3});
    g.node(s3).set_flag(NodeFlag::OnEffectChain);
    g.node(s3).set_flag(NodeFlag::MayThrow);

    // LoadIndex: ins[0]=ctrl, ins[1]=eff, ins[2]=base, ins[3]=key
    NodeId l1 = g.create(NodeKind::LoadIndex, {start, s3, d, key_a});
    g.node(l1).set_flag(NodeFlag::OnEffectChain);
    g.node(l1).set_flag(NodeFlag::MayThrow);
    NodeId l2 = g.create(NodeKind::LoadIndex, {start, l1, d, key_c});
    g.node(l2).set_flag(NodeFlag::OnEffectChain);
    g.node(l2).set_flag(NodeFlag::MayThrow);

    NodeId ret = g.create(NodeKind::Return, {start, l2});
    g.node(ret).set_flag(NodeFlag::OnEffectChain);
    g.set_end(ret);
    g.add_input(ret, l1);   // keep l1 alive

    g.function_name = global_symbols().intern("d2s_candidate");
    return g;
}

// Build a graph where the dict is ITERATED — D2S must NOT fire.
//   d = {}
//   d["a"] = 1
//   d["b"] = 2
//   it = iter(d)   <- this blocks specialization
[[nodiscard]] Graph make_iterated_dict() {
    Graph g;
    NodeId start = g.create(NodeKind::Start);
    g.set_start(start);

    NodeId d = g.create(NodeKind::NewDict, {start});
    g.node(d).set_flag(NodeFlag::OnEffectChain);

    auto mk_const_str = [&](std::uint32_t sym) {
        NodeId c = g.create(NodeKind::ConstPy);
        g.node(c).const_value.tag = Tag::None;
        g.node(c).symbol = sym;
        g.node(c).set_flag(NodeFlag::Pure);
        return c;
    };
    auto mk_const_int = [&](std::int64_t v) {
        NodeId c = g.create(NodeKind::ConstInt);
        g.node(c).const_value = Value::integer(v);
        g.node(c).set_flag(NodeFlag::Pure);
        return c;
    };

    NodeId key_a = mk_const_str(100);
    NodeId key_b = mk_const_str(200);
    NodeId val1 = mk_const_int(1);
    NodeId val2 = mk_const_int(2);

    NodeId s1 = g.create(NodeKind::StoreIndex, {start, d, d, key_a, val1});
    g.node(s1).set_flag(NodeFlag::OnEffectChain);
    NodeId s2 = g.create(NodeKind::StoreIndex, {start, s1, d, key_b, val2});
    g.node(s2).set_flag(NodeFlag::OnEffectChain);

    // Iter on the dict — this blocks D2S (iteration needs hash-table form).
    NodeId it = g.create(NodeKind::Iter, {start, s2, d});
    g.node(it).set_flag(NodeFlag::OnEffectChain);
    g.node(it).set_flag(NodeFlag::MayThrow);

    NodeId ret = g.create(NodeKind::Return, {start, it});
    g.node(ret).set_flag(NodeFlag::OnEffectChain);
    g.set_end(ret);

    g.function_name = global_symbols().intern("iter_dict");
    return g;
}

}  // namespace

// =============================================================================
// D2S fires on a 3-key record-like dict: shape guard emitted, all stores
// become StoreField, all loads become LoadField.
// =============================================================================
TEST(p46_d2s_specializes_record_like_dict) {
    Graph g = make_d2s_candidate();
    passes::PassContext ctx = make_tier2_ctx();
    passes::P46_DictLayoutSpecialization p;
    Result<passes::PassResult> r = p.run(g, ctx);
    CHECK(r.has_value());
    if (!r) return;
    CHECK(r->changed);

    // Shape guard emitted with ShapeIs subop and Speculative flag (Rule 5).
    std::uint32_t guards = 0;
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        if (n.kind != NodeKind::Guard) return;
        if (n.subop == static_cast<std::uint16_t>(GuardKind::ShapeIs)) {
            ++guards;
            CHECK(n.has(NodeFlag::Speculative));
            CHECK(n.has(NodeFlag::ShapeGuarded));
            CHECK(n.shape_id != 0);
        }
    });
    CHECK_EQ(guards, 1u);

    // 3 stores became StoreField, 2 loads became LoadField.
    CHECK_EQ(count_kind(g, NodeKind::StoreField), 3u);
    CHECK_EQ(count_kind(g, NodeKind::LoadField), 2u);
    // No StoreIndex/LoadIndex remain (all rewritten).
    CHECK_EQ(count_kind(g, NodeKind::StoreIndex), 0u);
    CHECK_EQ(count_kind(g, NodeKind::LoadIndex), 0u);
}

// Idempotency (Rule 10): a second run is a no-op. The shape guard already
// exists (has_shape_guard returns true), and no StoreIndex/LoadIndex remain.
TEST(p46_d2s_is_idempotent) {
    Graph g = make_d2s_candidate();
    passes::PassContext ctx = make_tier2_ctx();
    passes::P46_DictLayoutSpecialization p;
    (void)p.run(g, ctx);
    Result<passes::PassResult> r2 = p.run(g, ctx);
    CHECK(r2.has_value());
    if (!r2) return;
    CHECK(!r2->changed);
    // Still exactly 1 guard, 3 StoreField, 2 LoadField.
    CHECK_EQ(count_kind(g, NodeKind::Guard), 1u);
    CHECK_EQ(count_kind(g, NodeKind::StoreField), 3u);
    CHECK_EQ(count_kind(g, NodeKind::LoadField), 2u);
}

// D2S declines in Tier 1 (budget: Tier 2/3 only).
TEST(p46_d2s_declines_in_tier1) {
    Graph g = make_d2s_candidate();
    passes::PassContext ctx;
    ctx.tier = passes::TierMode::Tier1;
    ctx.node_budget = cfg::tier1_node_budget;
    ctx.code_unit_id = 1;
    ctx.target = &backend::host_target();
    passes::P46_DictLayoutSpecialization p;
    Result<passes::PassResult> r = p.run(g, ctx);
    CHECK(r.has_value());
    if (!r) return;
    CHECK(!r->changed);
    CHECK_EQ(count_kind(g, NodeKind::Guard), 0u);
    CHECK_EQ(count_kind(g, NodeKind::StoreField), 0u);
    CHECK_EQ(count_kind(g, NodeKind::LoadField), 0u);
}

// =============================================================================
// D2S must NOT fire when the dict is iterated — the iteration protocol
// needs the hash-table form. This is the "no key-dependent iteration"
// rule from the spec.
// =============================================================================
TEST(p46_d2s_blocks_on_iterated_dict) {
    Graph g = make_iterated_dict();
    passes::PassContext ctx = make_tier2_ctx();
    passes::P46_DictLayoutSpecialization p;
    Result<passes::PassResult> r = p.run(g, ctx);
    CHECK(r.has_value());
    if (!r) return;
    CHECK(!r->changed);
    // No specialization: no guard, no StoreField/LoadField.
    CHECK_EQ(count_kind(g, NodeKind::Guard), 0u);
    CHECK_EQ(count_kind(g, NodeKind::StoreField), 0u);
    CHECK_EQ(count_kind(g, NodeKind::LoadField), 0u);
    // Original StoreIndex nodes remain.
    CHECK_EQ(count_kind(g, NodeKind::StoreIndex), 2u);
}

// =============================================================================
// D2S must NOT fire when the dict has fewer than cfg::dict_layout_min_fields
// keys — the shape guard overhead exceeds the single-probe hash cost.
// =============================================================================
TEST(p46_d2s_declines_on_single_key_dict) {
    Graph g;
    NodeId start = g.create(NodeKind::Start);
    g.set_start(start);
    NodeId d = g.create(NodeKind::NewDict, {start});
    g.node(d).set_flag(NodeFlag::OnEffectChain);

    NodeId key = g.create(NodeKind::ConstPy);
    g.node(key).const_value.tag = Tag::None;
    g.node(key).symbol = 100;
    g.node(key).set_flag(NodeFlag::Pure);
    NodeId val = g.create(NodeKind::ConstInt);
    g.node(val).const_value = Value::integer(1);
    g.node(val).set_flag(NodeFlag::Pure);

    NodeId s = g.create(NodeKind::StoreIndex, {start, d, d, key, val});
    g.node(s).set_flag(NodeFlag::OnEffectChain);
    NodeId ret = g.create(NodeKind::Return, {start, s});
    g.node(ret).set_flag(NodeFlag::OnEffectChain);
    g.set_end(ret);

    passes::PassContext ctx = make_tier2_ctx();
    passes::P46_DictLayoutSpecialization p;
    Result<passes::PassResult> r = p.run(g, ctx);
    CHECK(r.has_value());
    if (!r) return;
    // 1 key < cfg::dict_layout_min_fields (2): must decline.
    CHECK(!r->changed);
    CHECK_EQ(count_kind(g, NodeKind::StoreField), 0u);
}

// =============================================================================
// Integration smoke-test: the full pipeline runs clean on a D2S candidate.
// Catches verifier violations introduced by the new Guard/StoreField/LoadField
// nodes interacting with downstream passes (GVN, DCE, etc.).
// =============================================================================
TEST(p46_d2s_passes_verifier_after_full_pipeline) {
    Graph g = make_d2s_candidate();
    passes::PassContext ctx = make_tier2_ctx();
    passes::OptPipeline pipeline;
    Result<void> r = passes::run_pipeline(g, ctx, pipeline);
    CHECK(r.has_value());
}
