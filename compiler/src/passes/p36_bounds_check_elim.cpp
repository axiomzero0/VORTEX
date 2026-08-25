// =============================================================================
// Pass 36 — Array Bounds Check Elimination  [HONEST NO-OP]
//
// Design intent: prove that a LoadIndex/StoreIndex's index is in bounds
// via IV range analysis (e.g., the canonical `for i in range(len(xs)):
// xs[i]`), and remove the IndexError check at runtime.
//
// Current status: NO-OP. The runtime's L_LOAD_INDEX path performs an
// inline bounds check unconditionally — there is no "unchecked" opcode
// variant the scheduler could emit. Without a runtime variant to lower
// to, marking the IR node would be a fake transformation: the runtime
// would still execute the check at every iteration, and the IR flag
// would be a dead write.
//
// The previous version of this pass set `NodeFlag::TypeGuarded` on
// LoadIndex/StoreIndex whose index was a loop IV. That flag IS read
// by p32 (vectorization) and p49 (effect reordering) as a "safe to
// reorder" hint — but bounds-checked-ness has nothing to do with
// aliasing or reorderability, so the previous semantics conflated two
// unrelated properties. p14 (demand-driven alias analysis) already
// sets TypeGuarded with the correct semantics (no alias = safe to
// reorder); p36's contribution was wrong.
//
// Until the runtime grows a LOAD_INDEX_UNCHECKED opcode (or equivalent
// guard-failure path) this pass reports "no change" and emits zero
// IR mutations. When the runtime supports it, this file should be
// rewritten to actually rewrite the LoadIndex kind to the unchecked
// variant — not to set a flag.
// =============================================================================

#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

Result<PassResult> P36_BoundsCheckElimination::run(Graph& g, const PassContext& c) noexcept {
    (void)g;
    (void)c;
    // Honest no-op: runtime has no unchecked-load opcode, so there is
    // nothing to lower to. See file header for rationale.
    PassResult r;
    r.nodes_before = g.live_node_count();
    r.nodes_after = r.nodes_before;
    r.changed = false;
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
