// =============================================================================
// Pass 48 — Thread-Local Allocation Buffer (TLAB) Sizing  [HONEST NO-OP]
//
// Design intent: size the mutator's bump-pointer arena from the unit's
// allocation profile (total bytes of in-unit allocations, PGO-weighted),
// amortizing the global heap sync across the largest allocation burst.
// The size hint rides the unit metadata, and the runtime reads it
// before entering the unit to pre-size its TLAB.
//
// Current status: NO-OP. The runtime has NO TLAB: allocations go through
// Runtime::heap_alloc() unconditionally, no thread-local buffers, no
// bump-pointer regions, no amortized reclamation. The RegionFree node
// this pass used to write to is never lowered by the backend and never
// seen by the runtime. Setting `aux1 = request` on RegionFree was a
// dead write — no consumer.
//
// Until a TLAB exists in the runtime, this pass reports "no change" and
// emits zero IR mutations. When a TLAB is introduced, this file should
// emit the actual TLAB-size metadata (probably a unit-level field, not
// a per-RegionFree aux) consumed by the unit-enter path — not a flag.
// =============================================================================

#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

Result<PassResult> P48_TLABSizing::run(Graph& g, const PassContext& c) noexcept {
    (void)g;
    (void)c;
    // Honest no-op: runtime has no TLAB, so there is nothing to size.
    // See file header for rationale.
    PassResult r;
    r.nodes_before = g.live_node_count();
    r.nodes_after = r.nodes_before;
    r.changed = false;
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
