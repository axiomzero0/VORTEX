// =============================================================================
// Pass 48 — Thread-Local Allocation Buffer (TLAB) Sizing.
//
// Sizes the mutator's bump-pointer arena from the unit's allocation
// profile: total bytes of in-unit allocations (region groups from pass
// 42 plus individual survivors) determines the TLAB request — amortizing
// the global heap sync across the largest allocation burst. The size
// hint rides the unit metadata (aux0 on the region marker) between
// cfg bounds.
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
            return true;
        default: return false;
    }
}

}  // namespace

Result<PassResult> P48_TLABSizing::run(Graph& g, const PassContext& c) noexcept {
    (void)c;
    std::uint32_t before = g.live_node_count();

    // Estimated bytes: each allocation's container overhead + payload.
    std::uint64_t bytes = 0;
    std::uint32_t allocs = 0;
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        if (!is_alloc(n.kind)) return;
        ++allocs;
        // header + per-element slots; elements = ins - 2
        std::uint32_t elements = n.ins.size() > 2 ? static_cast<std::uint32_t>(n.ins.size()) - 2 : 0;
        bytes += 48 + 16ull * elements;   // 48B header + 16B/element (Value)
    });

    // PGO weighting: hot units (call_count) scale the request.
    std::uint64_t weight = c.is_profiled() ? 2 : 1;
    std::uint32_t request = static_cast<std::uint32_t>(
        (bytes * weight) < cfg::tlab_default_bytes ? cfg::tlab_default_bytes
                                                   : (bytes * weight) > cfg::tlab_max_bytes
                                                         ? cfg::tlab_max_bytes
                                                         : bytes * weight);

    // Record on the region markers: the runtime reads the largest marker
    // before entering the unit.
    g.for_each_live([&](NodeId id) {
        Node& n = g.node(id);
        if (n.kind == NodeKind::RegionFree) {
            n.aux1 = request;
        }
    });

    PassResult r = result_of(g, before);
    r.changed = false;
    note(TelemetryEventKind::SafepointPatched, c, allocs);
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
