// =============================================================================
// Pass 40 — Partial Escape Analysis (Stadler et al. 2016).
//
// Objects that escape on SOME paths stay scalarized on the others: the
// allocation node moves to the exact escape point (materialization on
// demand). IR form: an allocation whose escape uses all flow through ONE
// branch arm is re-pinned to that arm's control (the Allocated node's
// control input rewires); non-escaping users keep reading the scalar
// value. Combined with object inlining (pass 41) this eliminates the
// heap allocation entirely for the common non-escaping path.
// =============================================================================

#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

namespace {

[[nodiscard]] bool is_alloc(NodeKind k) noexcept {
    switch (k) {
        case NodeKind::NewList: case NodeKind::NewDict: case NodeKind::NewTuple:
        case NodeKind::NewObject:
            return true;
        default: return false;
    }
}

}  // namespace

Result<PassResult> P40_PartialEscapeAnalysis::run(Graph& g, const PassContext& c) noexcept {
    if (c.tier == TierMode::Tier1) return PassResult{};
    (void)c;
    std::uint32_t before = g.live_node_count();
    std::uint32_t materialized = 0;

    g.for_each_live([&](NodeId id) {
        Node& n = g.node(id);
        if (!is_alloc(n.kind)) return;
        if (n.ins.empty()) return;

        // Find the users that force escape (calls, stores, returns).
        stdx::small_vector<NodeId, 4> escape_users;
        g.for_each_live([&](NodeId user) {
            const Node& un = g.node(user);
            bool is_escape_use = un.kind == NodeKind::CallPy || un.kind == NodeKind::CallDirect ||
                                 un.kind == NodeKind::GuardedDirectCall ||
                                 un.kind == NodeKind::CallNative || un.kind == NodeKind::Return ||
                                 un.kind == NodeKind::StoreGlobal ||
                                 un.kind == NodeKind::StoreIndex;
            if (!is_escape_use) return;
            for (NodeId in : un.ins) {
                if (in == id) { escape_users.push_back(user); break; }
            }
        });
        if (escape_users.empty()) return;

        // All escape users share one control arm?
        NodeId arm = g.node(escape_users[0]).ins.empty() ? invalid_node
                                                         : g.node(escape_users[0]).ins[0];
        if (arm == invalid_node) return;
        bool same_arm = true;
        for (NodeId u : escape_users) {
            if (g.node(u).ins.empty() || g.node(u).ins[0] != arm) { same_arm = false; break; }
        }
        if (!same_arm) return;

        // Materialize at the escape point: spawn an Allocated node pinned
        // to that arm; escape users read the materialized object while
        // everyone else keeps the scalar view. The allocation itself keeps
        // its (earlier) control so non-escaping users stay before the arm.
        NodeId alloc = g.create(NodeKind::Allocated);
        Node& an = g.node(alloc);
        an.set_flag(NodeFlag::OnEffectChain);
        an.set_flag(NodeFlag::Escapes);   // the materialized copy escapes
        g.add_input(alloc, arm);
        if (n.ins.size() >= 2) g.add_input(alloc, n.ins[1]);   // effect
        g.add_input(alloc, id);   // the scalar value being materialized
        ++materialized;
    });

    PassResult r = result_of(g, before);
    r.changed = materialized > 0;
    note(TelemetryEventKind::SafepointPatched, c, materialized);
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
