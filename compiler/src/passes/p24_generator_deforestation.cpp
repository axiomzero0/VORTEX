// =============================================================================
// Pass 24 — Generator/Coroutine State Machine Deforestation.
//
// Fully-consumed generator pipelines (sum(x for x in gen(n)) lowered
// eagerly by the frontend) carry no residual suspension: when no YIELD is
// reachable, the iterator protocol over in-unit materialized lists is a
// pure indirection and the pass marks those Iter nodes NoGIL (pure data
// iteration) for the backend's fast path.
// =============================================================================

#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

Result<PassResult> P24_GeneratorDeforestation::run(Graph& g, const PassContext& c) noexcept {
    (void)c;
    std::uint32_t before = g.live_node_count();

    bool has_yield = false;
    g.for_each_live([&](NodeId id) {
        if (g.node(id).kind == NodeKind::Yield) has_yield = true;
    });
    if (has_yield) return PassResult{};

    bool changed = false;
    g.for_each_live([&](NodeId id) {
        Node& n = g.node(id);
        if (n.kind != NodeKind::Iter || n.ins.size() < 3) return;
        NodeId base = n.ins[2];
        const Node& b = g.node(base);
        if (b.kind == NodeKind::NewList || b.kind == NodeKind::NewTuple) {
            n.set_flag(NodeFlag::NoGIL);
            changed = true;
        }
    });

    PassResult r = result_of(g, before);
    r.changed = changed;
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
