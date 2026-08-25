// =============================================================================
// Pass 19 — Call Graph Construction & PGO Weighting.
//
// Builds the intra-unit call graph: CallPy/GuardedDirectCall -> callee
// node; edges weighted by pgo_count (interpreter-recorded call counts).
// The weights drive inlining decisions in passes 20-22 (cost model input,
// Rule 45): hot edges inline, cold edges outline.
// =============================================================================

#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

Result<PassResult> P19_CallGraphPGO::run(Graph& g, const PassContext& c) noexcept {
    (void)c;
    std::uint32_t before = g.live_node_count();

    std::uint32_t edges = 0;
    std::uint64_t total_weight = 0;
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        if (n.kind != NodeKind::CallPy && n.kind != NodeKind::CallDirect &&
            n.kind != NodeKind::GuardedDirectCall) {
            return;
        }
        if (n.ins.size() < 3) return;
        NodeId callee = n.ins[2];
        // Edge weight: PGO count when present, static 1 otherwise.
        std::uint64_t w = n.pgo_count ? n.pgo_count : 1;
        g.node(callee).pgo_count += w;   // callee node accumulates incoming weight
        ++edges;
        total_weight += w;
    });

    PassResult r = result_of(g, before);
    r.changed = false;
    note(TelemetryEventKind::SafepointPatched, c, total_weight);
    (void)edges;
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
