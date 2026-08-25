// =============================================================================
// vortex/passes/pass_common.hpp — shared pass utilities (Rule 23/26/57).
//
// One helper set, consumed by all 51 pass files: result construction,
// telemetry emission, verifier invocation, fixpoint driver.
// =============================================================================

#pragma once

#include "vortex/ir/graph.hpp"
#include "vortex/ir/verifier.hpp"
#include "vortex/passes/pass_manager.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

/// Standard PassResult construction (nodes before/after + changed bit).
[[nodiscard]] inline PassResult result_of(const ir::Graph& g,
                                          std::uint32_t before) noexcept {
    PassResult r;
    r.nodes_before = before;
    r.nodes_after = g.live_node_count();
    r.changed = r.nodes_before != r.nodes_after;
    return r;
}

/// Record a pass effect in telemetry (Rule 26: no silent fallbacks).
inline void note(TelemetryEventKind kind, const PassContext& c, std::uint64_t aux = 0) noexcept {
    if (c.telemetry) c.telemetry->record(kind, c.code_unit_id, 0, aux);
}

/// Run the verifier after a pass in debug (Rule 40).
inline void verify_after(const ir::Graph& g, const char* name) noexcept {
    if (g_verify_after_each_pass) {
        g_verify_after_each_pass(g, name);
    }
}

/// Budget gate shared by every pass (Rule 10/45).
[[nodiscard]] inline bool budget_exceeded(const ir::Graph& g,
                                          const PassContext& c) noexcept {
    return g.live_node_count() > c.node_budget;
}

}  // namespace abi_v1
}  // namespace vortex::passes
