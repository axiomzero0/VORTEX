// =============================================================================
// Pass 39 — Basic Escape Analysis.
//
// An object ESCAPES when a reference to it flows anywhere the analysis
// cannot see: a call argument, a global store, a container store, a
// return value. Non-escaping objects are stack-allocatable. This pass
// computes the conservative escape set by reachability from escape
// points, then marks non-escaping allocations with Escapes=0 (clear) so
// passes 40/41 can scalarize them.
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

[[nodiscard]] bool is_escape_sink(const Graph& g, NodeId id) noexcept {
    const Node& n = g.node(id);
    switch (n.kind) {
        case NodeKind::Return:
        case NodeKind::StoreGlobal:
        case NodeKind::StoreAttr:
        case NodeKind::StoreIndex:
        case NodeKind::CallPy:
        case NodeKind::CallDirect:
        case NodeKind::GuardedDirectCall:
        case NodeKind::CallNative:
        case NodeKind::Yield:
        case NodeKind::Throw:
            return true;
        default:
            (void)g;
            return false;
    }
}

}  // namespace

Result<PassResult> P39_EscapeAnalysis::run(Graph& g, const PassContext& c) noexcept {
    (void)c;
    std::uint32_t before = g.live_node_count();

    // Backward reachability from escape sinks through data edges: every
    // allocation reachable from a sink escapes.
    stdx::flat_map<NodeId, bool, 32> escapes;
    stdx::small_vector<NodeId, 64> work;
    g.for_each_live([&](NodeId id) {
        if (!is_escape_sink(g, id)) return;
        for (std::uint32_t i = 0; i < g.node(id).ins.size(); ++i) {
            work.push_back(g.node(id).ins[i]);
        }
    });
    while (!work.empty()) {
        NodeId v = work.back();
        work.pop_back();
        if (escapes.contains(v)) continue;
        escapes.insert(v, true);
        for (NodeId in : g.node(v).ins) {
            work.push_back(in);
        }
    }

    // Mark allocations.
    std::uint32_t non_escaping = 0;
    g.for_each_live([&](NodeId id) {
        Node& n = g.node(id);
        if (!is_alloc(n.kind)) return;
        if (escapes.contains(id)) {
            n.set_flag(NodeFlag::Escapes);
        } else {
            n.clear_flag(NodeFlag::Escapes);
            ++non_escaping;
        }
    });

    PassResult r = result_of(g, before);
    r.changed = false;
    note(TelemetryEventKind::SafepointPatched, c, non_escaping);
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
