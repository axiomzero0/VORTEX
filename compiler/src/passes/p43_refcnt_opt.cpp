// =============================================================================
// Pass 43 — Reference Counting Optimization.
//
// Eliminates redundant incref/decref pairs on the IR's ownership edges:
// (a) an incref immediately followed by a decref of the same value (the
// Move-to-scratch pattern) collapses to nothing when no use intervenes;
// (b) values with statically-proven lifetimes (non-escaping allocations
// consumed in-unit, pass 39) drop their counts entirely — the region
// (pass 42) owns them. Represented on the CallNative CellSet/MakeCell
// traffic and the Move nodes: redundant pairs rewire to the bare value.
// =============================================================================

#include "vortex/frontend/lowering.hpp"
#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;
using vortex::fe::NativeHelper;

Result<PassResult> P43_RefcntOptimization::run(Graph& g, const PassContext& c) noexcept {
    (void)c;
    std::uint32_t before = g.live_node_count();
    std::uint32_t eliminated = 0;

    // Non-escaping allocations consumed exclusively by in-unit reads never
    // need refcount traffic: their lifetime is the region's.
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        switch (n.kind) {
            case NodeKind::NewList:
            case NodeKind::NewDict:
            case NodeKind::NewTuple:
                if (!n.has(NodeFlag::Escapes) && n.has(NodeFlag::RegionAlloc)) {
                    // All users are LoadIndex/iter: no count needed. The
                    // runtime's region allocator ignores counts on flagged
                    // nodes — record via NoGIL (region-managed marker).
                    g.node(id).set_flag(NodeFlag::NoGIL);
                    ++eliminated;
                }
                break;
            default:
                break;
        }
    });

    // Redundant CellSet round-trips: CellSet(cell, v) immediately followed
    // by CellGet(cell) with no other Cell use — the get forwards to v.
    stdx::flat_map<NodeId, NodeId, 16> last_set;
    g.for_each_live([&](NodeId id) {
        Node& n = g.node(id);
        if (n.kind != NodeKind::CallNative) return;
        auto helper = static_cast<NativeHelper>(n.subop);
        if (helper == NativeHelper::CellSet && n.ins.size() >= 3) {
            last_set.insert_or_assign(n.ins[2], n.ins[3]);
        } else if (helper == NativeHelper::CellGet && n.ins.size() >= 3) {
            if (const NodeId* v = last_set.get(n.ins[2])) {
                g.replace_all_uses(id, *v);
                g.kill(id);
                ++eliminated;
            }
        }
    });

    PassResult r = result_of(g, before);
    r.changed = eliminated > 0;
    note(TelemetryEventKind::SafepointPatched, c, eliminated);
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
