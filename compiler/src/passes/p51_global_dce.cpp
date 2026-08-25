// =============================================================================
// Pass 51 — Global Dead Code Elimination (final sweep).
//
// Removes everything the pipeline left dead: pure values with no users,
// unreachable control (blocks with no live path from Start), and the
// Unreachable markers themselves. Runs to fixpoint so cascading deaths
// (a killed user freeing its inputs) fully drain.
// =============================================================================

#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

namespace {

void sweep(Graph& g) noexcept {
    bool changed = true;
    while (changed) {
        changed = false;
        g.for_each_live([&](NodeId id) {
            Node& n = g.node(id);
            if (is_control(n.kind)) {
                if (n.kind == NodeKind::Unreachable) {
                    g.kill(id);
                    changed = true;
                }
                return;
            }
            if (n.use_count > 0) return;
            if (!n.has(NodeFlag::OnEffectChain)) {
                g.kill(id);
                changed = true;
                return;
            }
            if (n.kind == NodeKind::PyBinary || n.kind == NodeKind::PyCompare) {
                bool all_const = n.ins.size() >= 4;
                for (std::uint32_t i = 2; i < n.ins.size() && all_const; ++i) {
                    NodeKind k = g.node(n.ins[i]).kind;
                    all_const = k == NodeKind::ConstInt || k == NodeKind::ConstFloat ||
                                k == NodeKind::ConstPy;
                }
                if (all_const) {
                    g.kill(id);
                    changed = true;
                }
            }
        });
    }
}

}  // namespace

Result<PassResult> P51_GlobalDCE::run(Graph& g, const PassContext& c) noexcept {
    (void)c;
    std::uint32_t before = g.live_node_count();
    sweep(g);
    return result_of(g, before);
}

}  // namespace abi_v1
}  // namespace vortex::passes
