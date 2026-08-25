// =============================================================================
// Pass 34 — Software Pipelining  [HONEST NO-OP]
//
// Design intent: overlap loop iterations to hide latency — long-latency
// producers (calls, loads) issue EARLY, their consumers read LATE,
// iteration i+1's producers overlap iteration i's consumers. At the IR
// level, the pass re-pins control-dependent effect ops inside the body
// and records a modulo-scheduling initiation-interval hint.
//
// Current status: NO-OP. The previous version computed an II estimate
// with HARDCODED latencies (calls=4, loads=2, arith=1) and wrote it to
// `header.aux0`. The hardcoded latencies violated Rule 27 (machine
// facts must be queried via TargetDescriptor, not assumed). Worse, the
// scheduler does NOT read aux0 on Loop headers — no modulo-scheduling
// ever happened. That was a fake transformation with two defects.
//
// Real software pipelining requires:
//   1. A latency model queried from TargetDescriptor::latency(CostClass).
//   2. Dependence-graph construction over the loop body.
//   3. Modulo-scheduling: assignment of ops to stage+cycle within II.
//   4. Generation of prologue/epilogue/kernel stages at the IR level
//      (real node duplication, register renaming for stage k+1 vs k).
//   5. The scheduler must lower this structure to bytecode without
//      collapsing it back to sequential form.
// None of that was implemented; the previous "II hint" was a dead
// write. This honest no-op reports "no change" until the real
// implementation lands.
// =============================================================================

#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

Result<PassResult> P34_SoftwarePipelining::run(Graph& g, const PassContext& c) noexcept {
    (void)g;
    (void)c;
    // Honest no-op: real modulo-scheduling not implemented. See file
    // header for rationale.
    PassResult r;
    r.nodes_before = g.live_node_count();
    r.nodes_after = r.nodes_before;
    r.changed = false;
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
