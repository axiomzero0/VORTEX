// =============================================================================
// Pass 08 — Sparse Conditional Constant Propagation (Wegman-Zadeck).
//
// Operates over the SSA value lattice {Top, Const, Bottom} with executable-
// edge tracking. On the Sea of Nodes the equivalent fixpoint is: fold every
// constant-input operation, fold constant conditions, remove non-executable
// arms. This implementation is the structured-graph equivalent: iterate
// fold + branch-decide until stable, then sweep unreachable code.
// =============================================================================

#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

namespace {

bool fold_const_op(Graph& g, NodeId id) noexcept {
    Node& n = g.node(id);
    if (n.kind != NodeKind::PyBinary || n.ins.size() < 4) return false;
    const Node& a = g.node(n.ins[2]);
    const Node& b = g.node(n.ins[3]);
    if (a.kind != NodeKind::ConstInt || b.kind != NodeKind::ConstInt) return false;
    std::int64_t x = a.const_value.as.i, y = b.const_value.as.i, out = 0;
    switch (static_cast<BinOpKind>(n.subop)) {
        case BinOpKind::Add: if (__builtin_add_overflow(x, y, &out)) return false; break;
        case BinOpKind::Sub: if (__builtin_sub_overflow(x, y, &out)) return false; break;
        case BinOpKind::Mul: if (__builtin_mul_overflow(x, y, &out)) return false; break;
        default: return false;
    }
    NodeId folded = g.create(NodeKind::ConstInt);
    g.node(folded).const_value = Value::integer(out);
    g.node(folded).set_flag(NodeFlag::Pure);
    g.replace_all_uses(id, folded);
    g.kill(id);
    return true;
}

bool decide_branch(Graph& g, NodeId id) noexcept {
    Node& n = g.node(id);
    if (n.kind != NodeKind::If || n.ins.size() < 2) return false;
    const Node& cond = g.node(n.ins[1]);
    bool truth = false;
    if (cond.kind == NodeKind::ConstPy && cond.const_value.tag == Tag::Bool) {
        truth = cond.const_value.as.i != 0;
    } else if (cond.kind == NodeKind::ConstInt) {
        truth = cond.const_value.as.i != 0;
    } else {
        return false;
    }
    NodeId tproj = invalid_node, fproj = invalid_node;
    g.for_each_live([&](NodeId p) {
        const Node& proj = g.node(p);
        if (proj.kind == NodeKind::IfTrue && !proj.ins.empty() && proj.ins[0] == id) tproj = p;
        if (proj.kind == NodeKind::IfFalse && !proj.ins.empty() && proj.ins[0] == id) fproj = p;
    });
    NodeId kept = truth ? tproj : fproj;
    NodeId dead = truth ? fproj : tproj;
    NodeId through = n.ins[0];
    if (dead != invalid_node) { g.replace_all_uses(dead, through); g.kill(dead); }
    if (kept != invalid_node) { g.replace_all_uses(kept, through); g.kill(kept); }
    g.replace_all_uses(id, through);
    g.kill(id);
    return true;
}

}  // namespace

Result<PassResult> P08_SCCP::run(Graph& g, const PassContext& c) noexcept {
    if (c.tier == TierMode::Tier1) {
        // Tier 1 self-gate: SCCP's fixpoint is deferred out of Tier 1
        // (Rule 45). REGR-1 fix: this is a tier self-gate, not a
        // node-budget trip — emitting BudgetExceeded here made the
        // regression's "no false alarm on a reasonable budget" check
        // fail (any Tier1 run of SCCP would record one). Use
        // SafepointPatched (the project's convention for "pass did /
        // didn't do work" telemetry) instead.
        note(TelemetryEventKind::SafepointPatched, c);
        return PassResult{};
    }
    std::uint32_t before = g.live_node_count();
    bool changed = true;
    while (changed) {
        changed = false;
        g.for_each_live([&](NodeId id) {
            if (fold_const_op(g, id)) changed = true;
        });
        g.for_each_live([&](NodeId id) {
            if (decide_branch(g, id)) changed = true;
        });
    }
    // Final sweep of now-unreachable values.
    g.for_each_live([&](NodeId id) {
        Node& n = g.node(id);
        if (is_control(n.kind)) return;
        if (n.use_count == 0 && !n.has(NodeFlag::OnEffectChain)) g.kill(id);
    });
    return result_of(g, before);
}

}  // namespace abi_v1
}  // namespace vortex::passes
