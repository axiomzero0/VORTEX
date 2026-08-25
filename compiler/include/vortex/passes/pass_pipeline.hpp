// =============================================================================
// vortex/passes/pass_pipeline.hpp — the compile-time pass pipeline (Rule 51).
//
// A zero-allocation, zero-virtual-dispatch pipeline (Rule 9): the sequence is
// a std::tuple of pass instances iterated with fold expressions. Adding a
// pass is a one-line change (Rule 51's "automated refactoring" contract).
// =============================================================================

#pragma once

#include "vortex/passes/passes_fwd.hpp"
#include "vortex/ir/verifier.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

/// The full optimizing sequence (identical for every tier — Rule 1; tier
/// differences live in PassContext, not the sequence).
using OptPipeline = std::tuple<
    P03_TrivialDCE,
    P04_LocalConstantFolding,
    P05_AlgebraicSimplification,
    P06_ControlFlowSimplification,
    P07_LocalCSE,
    P08_SCCP,
    P09_RedundantStoreElimination,
    P10_EarlyGVN,
    P18_SideEffectAnalysis,
    P51_GlobalDCE>;

/// Budget-aware tier filter: Tier 1 runs only the linear-time subset.
struct TierFilter {
    TierMode tier{TierMode::Tier1};
    [[nodiscard]] bool include(const char* pass_name) const noexcept {
        if (tier != TierMode::Tier1) return true;
        // Tier 1: skip the fixpoint-heavy analyses (budget).
        std::string_view n(pass_name);
        return n != "08_sccp" && n != "10_early_gvn";
    }
};

/// Run the pipeline over one graph. Verifier (Rule 40) runs after every
/// pass in debug; telemetry records budgets (Rule 26).
template <typename Pipeline = OptPipeline>
[[nodiscard]] Result<void> run_pipeline(ir::Graph& g, const PassContext& ctx,
                                        Pipeline& pipeline) noexcept {
    TierFilter filter{ctx.tier};
    std::apply(
        [&](auto&... pass) {
            (
                [&]() {
                    if (!filter.include(pass.name)) return;
                    if (g.live_node_count() > ctx.node_budget) {
                        if (ctx.telemetry) {
                            ctx.telemetry->record(TelemetryEventKind::BudgetExceeded,
                                                  ctx.code_unit_id, 0, g.live_node_count());
                        }
                        return;   // budget guard: stop processing
                    }
                    auto r = pass.run(g, ctx);
                    if (!r) return;   // diagnostics propagate via graph state
                    if (g_verify_after_each_pass) {
                        g_verify_after_each_pass(g, pass.name);   // Rule 40
                    }
                }(),
                ...);
        },
        pipeline);
    return {};
}

/// Convenience: default-constructed pipeline.
[[nodiscard]] Result<void> optimize(ir::Graph& g, const PassContext& ctx) noexcept;

}  // namespace abi_v1
}  // namespace vortex::passes
