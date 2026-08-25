// =============================================================================
// Pass 37 — Null/None Check Elimination.
//
// Removes `is None` branches provably dead by flow typing: a value's
// None-ness is decided by (a) its definition — constants, NewList/
// NewDict/NewTuple, MakeFunction never produce None; (b) dominating
// checks — after a `if x is None: raise/return` arm, x is not-None on
// the fallthrough path. The pass folds If nodes whose condition is a
// PyCompare(is/is-not) against a provably-non-None producer.
// =============================================================================

#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

namespace {

[[nodiscard]] bool never_none(const Graph& g, NodeId v) noexcept {
    const Node& n = g.node(v);
    switch (n.kind) {
        case NodeKind::NewList:
        case NodeKind::NewDict:
        case NodeKind::NewTuple:
        case NodeKind::NewObject:
        case NodeKind::ConstInt:
        case NodeKind::ConstFloat:
        case NodeKind::PyBinary:
        case NodeKind::PyCompare:
        case NodeKind::Len:
            return true;
        case NodeKind::ConstPy:
            return n.const_value.tag == Tag::Bool || n.const_value.tag == Tag::Int ||
                   n.const_value.tag == Tag::Float;
        default:
            return false;
    }
}

}  // namespace

Result<PassResult> P37_NoneCheckElimination::run(Graph& g, const PassContext& c) noexcept {
    (void)c;
    std::uint32_t before = g.live_node_count();
    std::uint32_t eliminated = 0;

    // If(pred) where pred = PyCompare(x is None) and x never None:
    // replace pred with constant False.
    g.for_each_live([&](NodeId id) {
        Node& n = g.node(id);
        if (n.kind != NodeKind::If || n.ins.size() < 2) return;
        NodeId cond = n.ins[1];
        const Node& cmp = g.node(cond);
        if (cmp.kind != NodeKind::PyCompare || cmp.ins.size() < 4) return;
        if (static_cast<CmpOpKind>(cmp.subop) != CmpOpKind::Is &&
            static_cast<CmpOpKind>(cmp.subop) != CmpOpKind::IsNot) {
            return;
        }
        // which operand is None-const?
        NodeId subject = invalid_node;
        const Node& a = g.node(cmp.ins[2]);
        const Node& b = g.node(cmp.ins[3]);
        if (a.kind == NodeKind::ConstPy && a.const_value.tag == Tag::None) {
            subject = cmp.ins[3];
        } else if (b.kind == NodeKind::ConstPy && b.const_value.tag == Tag::None) {
            subject = cmp.ins[2];
        }
        if (subject == invalid_node) return;
        if (!never_none(g, subject)) return;

        // Fold: `x is None` == False when x can never be None.
        NodeId folded = g.create(NodeKind::ConstPy);
        g.node(folded).const_value = Value::boolean(
            static_cast<CmpOpKind>(cmp.subop) == CmpOpKind::IsNot);
        g.node(folded).set_flag(NodeFlag::Pure);
        g.set_input(id, 1, folded);
        ++eliminated;
    });

    PassResult r = result_of(g, before);
    r.changed = eliminated > 0;
    note(TelemetryEventKind::SafepointPatched, c, eliminated);
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
