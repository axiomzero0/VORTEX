// =============================================================================
// Pass 41 — Object Inlining (Scalar Replacement of Aggregates).
//
// A non-escaping object (pass 39 marker clear) shatters: its LoadIndex/
// LoadAttr reads become direct references to the stored values when the
// store is visible in-unit. For NewList with constant/known elements,
// every LoadIndex(list, const_k) forwards to element k's definition —
// the object header, the allocation, and the container all disappear.
// =============================================================================

#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

namespace {

[[nodiscard]] bool is_alloc(NodeKind k) noexcept {
    switch (k) {
        case NodeKind::NewList: case NodeKind::NewTuple:
            return true;
        default: return false;
    }
}

}  // namespace

Result<PassResult> P41_ObjectInlining::run(Graph& g, const PassContext& c) noexcept {
    (void)c;
    std::uint32_t before = g.live_node_count();
    std::uint32_t replaced = 0;

    // Map (alloc node, const index) -> value node.
    stdx::flat_map<std::uint64_t, NodeId, 32> elem;
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        if (!is_alloc(n.kind)) return;
        if (n.has(NodeFlag::Escapes)) return;   // escape analysis veto
        // Elements are inputs 2.. (control, effect, e0, e1, ...)
        for (std::uint32_t i = 2; i < n.ins.size(); ++i) {
            std::uint64_t key = (std::uint64_t(id) << 32) | (i - 2);
            elem.insert(key, n.ins[i]);
        }
    });

    // Forward constant-index loads on those allocations.
    g.for_each_live([&](NodeId id) {
        Node& n = g.node(id);
        if (n.kind != NodeKind::LoadIndex || n.ins.size() < 4) return;
        NodeId base = n.ins[2];
        if (!is_alloc(g.node(base).kind)) return;
        if (g.node(base).has(NodeFlag::Escapes)) return;
        const Node& idx = g.node(n.ins[3]);
        if (idx.kind != NodeKind::ConstInt) return;
        std::uint64_t key = (std::uint64_t(base) << 32) |
                            static_cast<std::uint32_t>(idx.const_value.as_i());
        if (const NodeId* val = elem.get(key)) {
            g.replace_all_uses(id, *val);
            g.kill(id);
            ++replaced;
        }
    });

    PassResult r = result_of(g, before);
    r.changed = replaced > 0;
    note(TelemetryEventKind::SafepointPatched, c, replaced);
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
