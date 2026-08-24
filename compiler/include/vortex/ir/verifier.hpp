// =============================================================================
// vortex/ir/verifier.hpp — Graph Verifier (Rule 40)
//
// Runs in debug builds after EVERY pass. Checks:
//   1. No dangling NodeIds (inputs point to live nodes).
//   2. Effect-chain continuity (memory ops reachable from start's effect).
//   3. Control dominance of data uses (simplified: control inputs are live
//      and every node's control input, if any, precedes it in dominator
//      order — full check runs the dominator analysis).
//   4. Use-def consistency (use counts match actual input occurrences).
//   5. FrameState attached to every speculative guard (Rule 5).
//
// The verifier NEVER repairs; it reports. Failures are Fatal diagnostics.
// =============================================================================

#pragma once

#include "vortex/ir/graph.hpp"

namespace vortex::ir {

inline namespace abi_v1 {

/// Returns a list of diagnostics (empty == valid).
[[nodiscard]] stdx::small_vector<Diagnostic, 4> verify_graph(const Graph& g) noexcept;

/// Convenience: true iff valid. On failure, reports to stderr when
/// VORTEX_DEBUG (so CI debug builds surface the exact pass that broke).
[[nodiscard]] bool verify_or_report(const Graph& g, const char* pass_name) noexcept;

}  // namespace abi_v1
}  // namespace vortex::ir
