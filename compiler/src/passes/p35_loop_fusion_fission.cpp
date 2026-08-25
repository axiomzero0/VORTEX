// =============================================================================
// Pass 35 — Loop Fusion & Fission  [HONEST NO-OP]
//
// Design intent:
//   Fusion: two ADJACENT loops (same bound, no intervening effects, no
//   carried dependences) merge into one — halves loop overhead and
//   improves cache locality.
//   Fission: a loop whose body exceeds the register-pressure budget
//   splits back apart.
//
// Current status: NO-OP. The previous version ran real analyses
// (same_range_bound, no-carried-dependence, register-pressure proxy)
// but then wrote `shape_id = 0xF00D` (fusion) or `0x5F1F` (fission)
// on the loop headers — markers no consumer reads. The scheduler does
// NOT merge or split loops based on shape_id; nothing downstream
// interprets those magic numbers. That was a fake transformation.
//
// Real loop fusion requires:
//   1. IR-level merge: stitch the two loop bodies into one, rewiring
//      effect-chain, IV phis, and exit projections of both loops.
//   2. Liveness analysis to verify register pressure doesn't exceed
//      the target's budget (queried from TargetDescriptor, not a
//      cfg constant).
//   3. Effect-dependence proof: no memory location written by A is
//      read by B (real alias analysis, not a heuristic).
// None of that was implemented.
//
// Until the real IR-rewriting version is implemented, this pass
// reports "no change" and emits zero IR mutations.
// =============================================================================

#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

Result<PassResult> P35_LoopFusionFission::run(Graph& g, const PassContext& c) noexcept {
    (void)g;
    (void)c;
    // Honest no-op: real IR-level fusion/fission not implemented. See
    // file header for rationale.
    PassResult r;
    r.nodes_before = g.live_node_count();
    r.nodes_after = r.nodes_before;
    r.changed = false;
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
