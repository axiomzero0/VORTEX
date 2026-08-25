// =============================================================================
// Pass 30 — Loop Unrolling  [HONEST NO-OP]
//
// Design intent: unroll loops with provably-constant trip counts
// (range-loops with constant bounds) by a factor capped by
// cfg::unroll_max_factor, the trip count, and the node budget.
//
// Current status: NO-OP. The previous version set `header.aux0 = factor`
// and `NodeFlag::Hot` on the loop header, claiming the scheduler would
// do "bytecode-level body duplication" — but the scheduler does NOT
// read aux0 on Loop headers (audited: scheduler.cpp reads aux0 only
// for ConstPy string-pool offsets, Parameter slots, CallPy argc/kw/cls).
// No body duplication ever happened. That was a fake transformation.
//
// Real IR-level full unrolling requires:
//   1. Cloning the body nodes N times.
//   2. Substituting each clone's IV phi with ConstInt(k).
//   3. Chaining the effect/memory state across clones.
//   4. Maintaining phi-rewiring for variables updated in the body.
//   5. Killing the original Loop / GetIterCheck / backedge.
// That's intricate IR surgery, and the for-in-range pattern lowered to
// a complex IR shape (GetIterCheck, Iter, IterNext) makes it harder
// still. The previous "easy route" of just tagging the header was a
// stub; this honest no-op documents that the real surgery has not
// been done yet.
//
// Until the IR-rewriting version is implemented, this pass reports
// "no change" and emits zero IR mutations.
// =============================================================================

#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

Result<PassResult> P30_LoopUnrolling::run(Graph& g, const PassContext& c) noexcept {
    (void)g;
    (void)c;
    // Honest no-op: real IR-level unrolling is not implemented. See
    // file header for rationale.
    PassResult r;
    r.nodes_before = g.live_node_count();
    r.nodes_after = r.nodes_before;
    r.changed = false;
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
