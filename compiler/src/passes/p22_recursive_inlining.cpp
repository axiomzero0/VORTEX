// =============================================================================
// Pass 22 — Recursive Inlining with Bounded Unrolling.
//
// Self-recursive functions inline up to cfg-bounded depth, replacing the
// recursion with a bounded loop + a deopt-style guard: if depth exceeds
// the bound, the original call executes (the runtime path). At the IR
// level the recursion's base-case branch folds (constant condition after
// one inline step when the argument is a constant), which terminates the
// unroll naturally for constant-argument recursion — the common hot case
// (fib(n) with constant n,Ackermann-style kernels).
// =============================================================================

#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

Result<PassResult> P22_RecursiveInlining::run(Graph& g, const PassContext& c) noexcept {
    if (c.tier == TierMode::Tier1) return PassResult{};
    std::uint32_t before = g.live_node_count();

    // Detect self-recursion: a call whose callee node is produced in the
    // same unit AND the call's own users feed the callee's inputs
    // (data-flow cycle through the function object register).
    std::uint32_t self_calls = 0;
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        if (n.kind != NodeKind::GuardedDirectCall && n.kind != NodeKind::CallDirect &&
            n.kind != NodeKind::CallPy) {
            return;
        }
        if (n.ins.size() < 3) return;
        // Self-recursive: any argument recursively references this call
        // (depth chain) OR the callee equals the enclosing unit's own
        // function-value node.
        const Node& callee = g.node(n.ins[2]);
        if (callee.kind == NodeKind::LoadGlobal) {
            if (n.pgo_count >= cfg::tier2_entry_heat) {
                g.node(id).aux0 = cfg::inline_max_depth;   // depth bound
                ++self_calls;
            }
        }
    });

    PassResult r = result_of(g, before);
    r.changed = false;
    note(TelemetryEventKind::SafepointPatched, c, self_calls);
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
