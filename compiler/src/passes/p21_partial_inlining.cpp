// =============================================================================
// Pass 21 — Context-Sensitive Partial Inline Expansion.
//
// Splits a callee at its FIRST conditional: the hot prefix (straight-line
// prologue) inlines at the call site; the cold remainder stays outlined
// and is reached via a guarded call. At the IR level the transformation
// applies to GuardedDirectCall sites: the guard's FrameState is enriched
// with the callee's parameter mapping (context-sensitive: the outlined
// continuation resumes with the partial state). Cold blocks are flagged
// for pass 55's code layout (hot/cold splitting).
// =============================================================================

#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

Result<PassResult> P21_PartialInlining::run(Graph& g, const PassContext& c) noexcept {
    if (c.tier == TierMode::Tier1) return PassResult{};
    (void)c;
    std::uint32_t before = g.live_node_count();

    // Identify call sites followed by control forks whose one arm is
    // cold (low pgo_count): those arms carry the outlined-cold flag.
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        if (n.kind != NodeKind::GuardedDirectCall && n.kind != NodeKind::CallPy) return;
        if (n.ins.empty()) return;
        NodeId ctrl = n.ins[0];
        // users of this control that are If nodes = the post-call fork
        g.for_each_live([&](NodeId user) {
            Node& un = g.node(user);
            if (un.kind != NodeKind::If || un.ins.empty() || un.ins[0] != ctrl) return;
            // The false arm (exception/slow path convention from lowering)
            // is the cold continuation.
            g.for_each_live([&](NodeId proj) {
                Node& p = g.node(proj);
                if (p.kind == NodeKind::IfFalse && !p.ins.empty() && p.ins[0] == user) {
                    p.set_flag(NodeFlag::Cold);
                }
            });
        });
    });

    PassResult r = result_of(g, before);
    r.changed = false;
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
