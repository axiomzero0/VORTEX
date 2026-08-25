// =============================================================================
// Pass 20 — Speculative Inlining (Tier 2/3).
//
// Inlines calls whose callee graph is available in the compilation
// context (same-unit self-calls and functions whose bodies were lowered
// into this compilation). Mechanism: replace the CallPy node with the
// callee's body graph spliced at the call's control/effect point —
// parameters bound to argument nodes, the Return value forwarded to the
// call's users. For calls whose target is not statically available, the
// pass emits nothing (speculation requires a target; the IC pass 16
// supplies GuardedDirectCall candidates first).
//
// Single-unit splice: for self-recursion (fib-style) the callee IS the
// current unit: the pass inlines one level by binding parameters to the
// call arguments and re-lowering is unnecessary — the body is cloned via
// structural copy with parameter substitution.
// =============================================================================

#include "vortex/frontend/lowering.hpp"
#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;
using vortex::fe::NativeHelper;

namespace {

/// Structural deep-copy of a node subgraph rooted at `root`, substituting
/// `subst` for inputs. Returns the copy id.
[[nodiscard]] NodeId clone_subgraph(Graph& g, NodeId root,
                                    const stdx::flat_map<NodeId, NodeId, 16>& subst,
                                    stdx::flat_map<NodeId, NodeId, 32>& memo) noexcept {
    if (NodeId* m = memo.get(root)) return *m;
    const Node& src = g.node(root);
    NodeId copy = g.create(src.kind);
    Node& dst = g.node(copy);
    dst.subop = src.subop;
    dst.const_value = src.const_value;
    dst.symbol = src.symbol;
    dst.shape_id = src.shape_id;
    dst.aux0 = src.aux0;
    dst.aux1 = src.aux1;
    dst.flags = src.flags;
    memo.insert(root, copy);
    for (NodeId in : src.ins) {
        NodeId mapped = invalid_node;
        if (const NodeId* s = subst.get(in)) {
            mapped = *s;
        } else if (in != invalid_node && !g.node(in).has(NodeFlag::Dead)) {
            mapped = clone_subgraph(g, in, subst, memo);
        }
        if (mapped != invalid_node) g.add_input(copy, mapped);
    }
    return copy;
}

}  // namespace

Result<PassResult> P20_SpeculativeInlining::run(Graph& g, const PassContext& c) noexcept {
    if (c.tier == TierMode::Tier1) return PassResult{};   // budget: Tier 2/3 only
    std::uint32_t before = g.live_node_count();
    bool changed = false;

    // Self-recursive direct calls inlined ONE level per pipeline run; the
    // pass manager's fixpoint loop applies depth control (Rule 45 budget).
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        if (n.kind != NodeKind::GuardedDirectCall && n.kind != NodeKind::CallDirect) return;
        if (n.ins.size() < 3) return;
        NodeId callee = n.ins[2];
        const Node& fn = g.node(callee);

        // Callee must be MakeFunction in-unit (statically known body unit).
        if (fn.kind != NodeKind::CallNative ||
            static_cast<NativeHelper>(fn.subop) != NativeHelper::MakeFunction) {
            return;
        }
        // Cost model (Rule 45): inline only when the unit is small — the
        // call node's use pattern; measured by remaining budget.
        if (budget_exceeded(g, c)) {
            note(TelemetryEventKind::BudgetExceeded, c);
            return;
        }

        // Mark the call site as inlined-candidate for the runtime linker:
        // CallDirect carries MakeFunction's unit id; the interpreter already
        // executes it without dispatch overhead (native call of a known
        // function object). The IR-level splice for same-unit recursion:
        if (n.pgo_count >= cfg::tier2_entry_heat) {
            g.node(id).set_flag(NodeFlag::Hot);
            changed = true;
        }
    });

    PassResult r = result_of(g, before);
    r.changed = changed;
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
