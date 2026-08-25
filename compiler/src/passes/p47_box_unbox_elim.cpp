// =============================================================================
// Pass 47 — Box/Unbox Elimination.
//
// Keeps native scalars OUT of PyObject wrappers across deforestation
// chains: an Unbox(Box(x)) is the identity; a Box whose only users are
// Unbox nodes forwards its raw input; arithmetic on unboxed values stays
// in registers (the Unboxed flag) instead of re-entering the object
// world. Box/Unbox Deforestation: a Box immediately consumed by a Unbox
// cancels, keeping the value in a machine register across the pair.
// =============================================================================

#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

Result<PassResult> P47_BoxUnboxElimination::run(Graph& g, const PassContext& c) noexcept {
    (void)c;
    std::uint32_t before = g.live_node_count();
    std::uint32_t eliminated = 0;

    // Identity pairs: Unbox(Box(x)) -> x.
    g.for_each_live([&](NodeId id) {
        Node& n = g.node(id);
        if (n.kind != NodeKind::Unbox || n.ins.empty()) return;
        NodeId boxed = n.ins[0];
        const Node& b = g.node(boxed);
        if (b.kind != NodeKind::Box) return;
        if (b.ins.empty()) return;
        g.replace_all_uses(id, b.ins[0]);
        g.kill(id);
        ++eliminated;
    });

    // Boxes whose EVERY user is an Unbox never materialize an object.
    g.for_each_live([&](NodeId id) {
        Node& n = g.node(id);
        if (n.kind != NodeKind::Box || n.ins.empty()) return;
        bool all_unbox = n.use_count > 0;
        g.for_each_live([&](NodeId user) {
            const Node& un = g.node(user);
            bool uses_it = false;
            for (NodeId in : un.ins) {
                if (in == id) { uses_it = true; break; }
            }
            if (uses_it && un.kind != NodeKind::Unbox) all_unbox = false;
        });
        if (all_unbox) {
            n.set_flag(NodeFlag::Unboxed);   // no object: raw register
            ++eliminated;
        }
    });

    PassResult r = result_of(g, before);
    r.changed = eliminated > 0;
    note(TelemetryEventKind::SafepointPatched, c, eliminated);
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
