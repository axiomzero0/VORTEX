// =============================================================================
// Pass 03 — Trivial Dead Code Elimination.
//
// Removes nodes with no side effects and zero live uses. Python-level
// effect-chained operations (PyBinary/PyCompare) are removable when every
// operand is a constant: constant dispatch is total, so the op cannot
// dispatch user code or raise.
// Idempotent (Rule 10); re-run converges to a fixpoint within one call.
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
            if (is_control(n.kind)) return;
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

Result<PassResult> P03_TrivialDCE::run(Graph& g, const PassContext& c) noexcept {
    (void)c;
    std::uint32_t before = g.live_node_count();
    sweep(g);
    return result_of(g, before);
}

}  // namespace abi_v1
}  // namespace vortex::passes
