// =============================================================================
// Pass 44 — Write Barrier Elimination  [HONEST NO-OP]
//
// Design intent: a generational GC's write barrier (old-space store
// pointing to a young object) can be skipped when the stored value is
// provably not young (constants, interned strings, non-escaping region
// allocations). The pass should mark stores whose value is barrier-free,
// and the runtime's store paths should skip the card-mark on flagged
// nodes.
//
// Current status: NO-OP. The runtime has NO generational GC and NO write
// barrier — store paths in interp.cpp/objects.cpp perform unconditional
// writes with no card-mark, no remembered-set, no generational
// bookkeeping. Running this pass would set a NodeFlag::NoGIL marker on
// stores that no consumer reads, advertising a barrier skip that never
// happens. That's a fake transformation.
//
// Until a generational GC is introduced, this pass reports "no change"
// and emits zero IR mutations. When a write barrier exists, this file
// should be rewritten to emit the actual barrier-skip op (or omit the
// barrier op) at the store site — not to set a flag.
// =============================================================================

#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

Result<PassResult> P44_WriteBarrierElimination::run(Graph& g, const PassContext& c) noexcept {
    (void)g;
    (void)c;
    // Honest no-op: runtime has no write barrier, so there is nothing to
    // eliminate. See file header for rationale.
    PassResult r;
    r.nodes_before = g.live_node_count();
    r.nodes_after = r.nodes_before;
    r.changed = false;
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
