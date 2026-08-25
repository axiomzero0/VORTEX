// =============================================================================
// Pass 38 — GIL Hoisting & Batching  [HONEST NO-OP]
//
// Design intent: identify regions of pure native computation and emit a
// single acquire/release of the interpreter lock at the region boundary
// instead of per-op acquisition.
//
// Current status: NO-OP. The runtime does not have a Global Interpreter
// Lock (no `acquire_lock`/`release_lock` calls, no GIL state in the
// Vm). There is nothing to hoist or batch. Running this pass against the
// current runtime would be a fake transformation — the IR would carry a
// NoGIL flag that no consumer reads, advertising an optimization that
// never executes.
//
// Until the runtime grows a GIL or an equivalent serialization point,
// this pass reports "no change" and emits zero IR mutations. When a GIL
// is introduced, this file should be rewritten to emit the acquire/release
// region markers — not to set a flag for someone else to interpret.
// =============================================================================

#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

Result<PassResult> P38_GILHoisting::run(Graph& g, const PassContext& c) noexcept {
    (void)g;
    (void)c;
    // Honest no-op: runtime has no GIL, so there is nothing to hoist.
    // Report no change. See file header for rationale.
    PassResult r;
    r.nodes_before = g.live_node_count();
    r.nodes_after = r.nodes_before;
    r.changed = false;
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
