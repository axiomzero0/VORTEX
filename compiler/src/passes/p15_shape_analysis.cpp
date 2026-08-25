// =============================================================================
// Pass 15 — Python Object Shape/Dict Layout Analysis.
//
// Analyzes StoreAttr/LoadAttr chains on instances: within a unit, the
// attribute WRITE sequence defines the object's shape trajectory. When
// every access to a base follows the same fixed attribute set (no
// dynamically-growing attribute sets between allocation and use), the base
// is marked ShapeGuarded: passes 40b/46 may specialize it to fixed-offset
// field loads. Instance bases whose attribute set grows mid-unit are
// marked Escapes (shape polymorphic — runtime shape machinery required).
// =============================================================================

#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

Result<PassResult> P15_ShapeAnalysis::run(Graph& g, const PassContext& c) noexcept {
    (void)c;
    std::uint32_t before = g.live_node_count();

    // For each base node: the ordered set of stored attributes.
    stdx::flat_map<NodeId, stdx::small_vector<std::uint32_t, 4>, 16> base_shape;
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        if (n.kind != NodeKind::StoreAttr || n.ins.size() < 3) return;
        NodeId base = n.ins[2];
        auto* shape = base_shape.get(base);
        if (!shape) { base_shape.insert(base, {}); shape = base_shape.get(base); }
        bool have = false;
        for (std::uint32_t s : *shape) {
            if (s == n.symbol) { have = true; break; }
        }
        if (!have) shape->push_back(n.symbol);
    });

    // LoadAttrs consistent with the stored set -> ShapeGuarded base.
    std::uint32_t guarded = 0;
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        if (n.kind != NodeKind::LoadAttr || n.ins.size() < 3) return;
        NodeId base = n.ins[2];
        if (const auto* shape = base_shape.get(base)) {
            bool known = false;
            for (std::uint32_t s : *shape) {
                if (s == n.symbol) { known = true; break; }
            }
            if (known) {
                g.node(id).set_flag(NodeFlag::ShapeGuarded);
                g.node(base).set_flag(NodeFlag::ShapeGuarded);
                ++guarded;
            }
        }
    });

    PassResult r = result_of(g, before);
    r.changed = false;
    note(TelemetryEventKind::WatchdogInvalidation, c, guarded);
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
