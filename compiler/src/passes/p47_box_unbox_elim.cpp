// =============================================================================
// Pass 47 — Box/Unbox Elimination (The "Unbox Engine").
//
// Keeps native scalars OUT of PyObject wrappers across deforestation
// chains. Three sub-transformations, each idempotent (Rule 10):
//
//   47a: Identity pair elimination — Unbox(Box(x)) → x.
//        The Box/Unbox pair cancels, keeping the value in a machine
//        register across the pair.
//
//   47b: Chain forwarding — Unbox(x) where x is already Unboxed (has
//        the NodeFlag::Unboxed flag) is a no-op. Forward x directly.
//        This handles the case where the frontend or an earlier pass
//        (PEA, object inlining) already produced an unboxed value but
//        wrapped it in a defensive Unbox node.
//
//   47c: Box deforestation — a Box whose EVERY user is an Unbox never
//        materializes an object. Each Unbox(Box(x)) → x cancels, and
//        the Box becomes dead (killed by P51 GlobalDCE). This is the
//        spec's "Box/Unbox Deforestation" — the value stays in a
//        machine register across multiple operations.
//
// Idempotency (Rule 10): 47a and 47b kill the Unbox node via RAUW.
// A second run sees no Unbox(Box(x)) or Unbox(Unboxed) patterns. 47c
// sets the Unboxed flag on the Box and decrements its use count as
// each consumer Unbox is killed; a second run finds use_count == 0
// and the Box is already dead (P51 GlobalDCE collects it).
//
// Performance: uses Graph::users_of() (O(edges)) instead of the
// previous O(N²) nested for_each_live scan.
// =============================================================================

#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

namespace {

/// Check if every user of `node_id` is an Unbox node. Returns true if
/// the node is ONLY consumed by Unbox nodes (the deforestation condition).
/// Uses Graph::users_of() — O(edges), not O(nodes).
[[nodiscard]] bool all_users_are_unbox(const Graph& g, NodeId node_id) noexcept {
    auto users = g.users_of(node_id);
    if (users.empty()) return false;   // no users: not "all unbox", just dead
    for (NodeId u : users) {
        if (g.node(u).kind != NodeKind::Unbox) return false;
    }
    return true;
}

}  // namespace

Result<PassResult> P47_BoxUnboxElimination::run(Graph& g, const PassContext& c) noexcept {
    (void)c;
    std::uint32_t before = g.live_node_count();
    std::uint32_t eliminated = 0;

    // Collect candidates first (collect-then-rewrite — g.replace_all_uses
    // and g.kill mutate the graph while we hold NodeId references).
    struct ElimCandidate {
        NodeId unbox_id;     // the Unbox to eliminate
        NodeId forward_to;   // the node to replace it with
    };
    stdx::small_vector<ElimCandidate, 16> candidates;

    // --- 47a + 47b: scan all Unbox nodes for identity or chain patterns.
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        if (n.kind != NodeKind::Unbox || n.ins.empty()) return;
        NodeId src = n.ins[0];
        const Node& s = g.node(src);

        // 47a: Unbox(Box(x)) → x (identity pair).
        if (s.kind == NodeKind::Box && !s.ins.empty()) {
            candidates.push_back({id, s.ins[0]});
            return;
        }

        // 47b: Unbox(x) where x is already Unboxed → x (chain forwarding).
        // The source is already a native scalar; the Unbox is a no-op.
        if (s.has(NodeFlag::Unboxed)) {
            candidates.push_back({id, src});
            return;
        }
    });

    // Apply 47a + 47b eliminations.
    for (const ElimCandidate& ec : candidates) {
        if (g.node(ec.unbox_id).has(NodeFlag::Dead)) continue;   // already killed
        g.replace_all_uses(ec.unbox_id, ec.forward_to);
        g.kill(ec.unbox_id);
        ++eliminated;
    }

    // --- 47c: Box deforestation — Boxes whose EVERY user is an Unbox
    // never materialize an object. Mark them Unboxed so downstream
    // consumers (SLP, PEA) see the native scalar. The Box stays in the
    // graph (it's the effect-chain anchor) but its Unboxed flag tells
    // the backend to use the raw register, not the heap object.
    //
    // Idempotency: a Box already marked Unboxed is skipped. A Box whose
    // consumers were all killed by 47a now has use_count == 0 — it's
    // dead and P51 GlobalDCE will collect it.
    g.for_each_live([&](NodeId id) {
        Node& n = g.node(id);
        if (n.kind != NodeKind::Box || n.ins.empty()) return;
        if (n.has(NodeFlag::Unboxed)) return;   // idempotent: already marked
        if (!all_users_are_unbox(g, id)) return;
        n.set_flag(NodeFlag::Unboxed);
        ++eliminated;
    });

    PassResult r = result_of(g, before);
    r.changed = eliminated > 0;
    note(TelemetryEventKind::SafepointPatched, c, eliminated);
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
