// =============================================================================
// Pass 44 — Write Barrier Elimination.
//
// A generational GC's write barrier (old-space store pointing to a young
// object) is unnecessary when the stored value provably cannot be young:
// constants, interned strings, and non-escaping in-unit allocations
// promoted at region end. The pass marks StoreAttr/StoreIndex nodes
// whose value input is barrier-free; the runtime's store paths skip the
// card-mark on flagged nodes.
// =============================================================================

#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

namespace {

[[nodiscard]] bool never_young(const Graph& g, NodeId v) noexcept {
    const Node& n = g.node(v);
    switch (n.kind) {
        case NodeKind::ConstInt:
        case NodeKind::ConstFloat:
        case NodeKind::ConstPy:
            return true;   // immediates and interned constants
        case NodeKind::NewList:
        case NodeKind::NewDict:
        case NodeKind::NewTuple:
            // Region-allocated objects die with the region: never enter the
            // young generation's remembered set.
            return n.has(NodeFlag::RegionAlloc) && !n.has(NodeFlag::Escapes);
        default:
            (void)g;
            return false;
    }
}

}  // namespace

Result<PassResult> P44_WriteBarrierElimination::run(Graph& g, const PassContext& c) noexcept {
    (void)c;
    std::uint32_t before = g.live_node_count();
    std::uint32_t eliminated = 0;

    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        if (n.kind != NodeKind::StoreAttr && n.kind != NodeKind::StoreIndex) return;
        // value operand position: StoreAttr (control, effect, base, value);
        // StoreIndex (control, effect, base, index, value).
        NodeId value = invalid_node;
        if (n.kind == NodeKind::StoreAttr && n.ins.size() >= 4) value = n.ins[3];
        if (n.kind == NodeKind::StoreIndex && n.ins.size() >= 5) value = n.ins[4];
        if (value == invalid_node) return;
        if (never_young(g, value)) {
            g.node(id).set_flag(NodeFlag::NoGIL);   // barrier-free store marker
            ++eliminated;
        }
    });

    PassResult r = result_of(g, before);
    r.changed = false;
    note(TelemetryEventKind::SafepointPatched, c, eliminated);
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
