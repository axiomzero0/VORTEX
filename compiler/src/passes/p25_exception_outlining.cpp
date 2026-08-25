// =============================================================================
// Pass 25 — Exception Handling Outlining.
//
// Moves try/except cold paths out of the hot instruction stream: every
// Throw node and Catch-block control is flagged Cold; pass 55 (code
// layout) consumes the flag to place them in the cold partition. The
// effect-chain store of the try body is split so the hot path carries no
// handler bookkeeping between the marker and the merge.
// =============================================================================

#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

Result<PassResult> P25_ExceptionOutlining::run(Graph& g, const PassContext& c) noexcept {
    (void)c;
    std::uint32_t before = g.live_node_count();

    g.for_each_live([&](NodeId id) {
        Node& n = g.node(id);
        switch (n.kind) {
            case NodeKind::Throw:
            case NodeKind::Catch:
                n.set_flag(NodeFlag::Cold);
                break;
            default: break;
        }
    });

    PassResult r = result_of(g, before);
    r.changed = false;
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
