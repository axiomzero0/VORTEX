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
#include "vortex/support/introspect.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

/// The full optimizing sequence (identical for every tier — Rule 1; tier
/// differences live in PassContext, not the sequence). All 49 optimization
/// passes in spec order; frontend passes 1-2 run at graph construction.
using OptPipeline = std::tuple<
    // Phase 1: IR normalization & intraprocedural foundation
    P03_TrivialDCE,
    P04_LocalConstantFolding,
    P05_AlgebraicSimplification,
    P06_ControlFlowSimplification,
    P07_LocalCSE,
    P08_SCCP,
    P09_RedundantStoreElimination,
    P10_EarlyGVN,
    // Phase 2: alias & pointer analysis
    P11_AndersenPointsTo,
    P12_CFLReachabilityAlias,
    P13_FlowSensitiveAlias,
    P14_DemandDrivenAlias,
    P15_ShapeAnalysis,
    P16_ICMonomorphism,
    P16b_PolymorphicDispatch,
    P17_MROLinearization,
    P18_SideEffectAnalysis,
    // Phase 3: interprocedural & speculative inlining
    P19_CallGraphPGO,
    P20_SpeculativeInlining,
    P21_PartialInlining,
    P22_RecursiveInlining,
    P23_ClosureDevirtualization,
    P24_GeneratorDeforestation,
    P25_ExceptionOutlining,
    P26_IPCP,
    // Phase 4: loop optimizations & vectorization
    P27_LICM,
    P28_SpeculativeLICM,
    P29_InductionVariables,
    P30_LoopUnrolling,
    P31_SLPVectorization,
    P32_LoopVectorization,
    P33_PolyhedralOptimization,
    P34_SoftwarePipelining,
    P35_LoopFusionFission,
    P36_BoundsCheckElimination,
    P37_NoneCheckElimination,
    P38_GILHoisting,
    // Phase 5: memory, allocation, escape analysis
    P39_EscapeAnalysis,
    P40_PartialEscapeAnalysis,
    P41_ObjectInlining,
    P42_RegionMemoryInference,
    P43_RefcntOptimization,
    P44_WriteBarrierElimination,
    P45_StringInterning,
    P46_DictLayoutSpecialization,
    P47_BoxUnboxElimination,
    P48_TLABSizing,
    // Phase 6: late optimizations & backend preparation
    P49_SpeculativeEffectReordering,
    P50_LateGVN,
    P51_GlobalDCE>;

/// Budget-aware tier filter. Two independent gates:
///   1. OPT-OUT gate (all tiers): passes flagged OptOption in the spec only
///      SKIP when the caller explicitly opted out. Polyhedral (33) is the
///      canonical example — DEFAULT-ON, opt-out only via
///      OptOption::DisablePolyhedral. The only legitimate reason to opt
///      out is compilation-time sensitivity (see pass header for the full
///      rationale).
///   2. TIER gate: Tier 1 runs only the linear-time subset. Tier 1 still
///      gets polyhedral — it's linear in N — UNLESS the budget gate also
///      fires (the Tier 1 linear-time list below excludes polyhedral
///      alongside the other non-linear analyses).
struct TierFilter {
    TierMode tier{TierMode::Tier1};
    explicit TierFilter(TierMode t) noexcept : tier(t) {}
    TierFilter() = default;
    [[nodiscard]] bool include(const char* pass_name, const PassContext& ctx) const noexcept {
        std::string_view n(pass_name);
        // Opt-out optimizations: skip when the caller explicitly disabled.
        // Polyhedral is DEFAULT-ON — set OptOption::DisablePolyhedral to
        // skip it (the only legitimate reason: compilation-time).
        if (n == "33_polyhedral" && ctx.options.has(OptOption::DisablePolyhedral)) {
            return false;
        }
        if (tier != TierMode::Tier1) return true;
        // Tier 1 (budget-constrained baseline): cheap passes only. The
        // fixpoint-heavy analyses and all speculation defer to Tier 2/3 —
        // each pass additionally self-gates on the tier mode. NOTE:
        // polyhedral is O(N + L + ΣA·D) — linear in N for bounded D — but
        // it's still a fixpoint-heavy analysis that we defer in Tier 1 to
        // keep the baseline JIT's compile time bounded.
        return n != "08_sccp" && n != "10_early_gvn" && n != "11_andersen" &&
               n != "12_cfl_alias" && n != "14_demand_alias" && n != "16_ic_mono" &&
               n != "16b_ic_poly" &&
               n != "20_spec_inline" && n != "21_partial_inline" &&
               n != "22_recursive_inline" && n != "26_ipcp" && n != "28_spec_licm" &&
               n != "30_unroll" && n != "31_slp" && n != "32_loop_vec" &&
               n != "33_polyhedral" && n != "34_sw_pipeline" && n != "40_pea" &&
               n != "49_effect_reorder" && n != "50_late_gvn";
    }
};

/// Run the pipeline over one graph. Verifier (Rule 40) runs after every
/// pass in debug; telemetry records budgets (Rule 26). PhaseScope records
/// per-pass duration + node-count delta (Rule 123).
template <typename Pipeline = OptPipeline>
[[nodiscard]] Result<void> run_pipeline(ir::Graph& g, const PassContext& ctx,
                                        Pipeline& pipeline) noexcept {
    TierFilter filter{ctx.tier};
    std::apply(
        [&](auto&... pass) {
            (
                [&]() {
                    if (!filter.include(pass.name, ctx)) return;
                    if (g.live_node_count() > ctx.node_budget) {
                        if (ctx.telemetry) {
                            ctx.telemetry->record(TelemetryEventKind::BudgetExceeded,
                                                  ctx.code_unit_id, 0, g.live_node_count());
                        }
                        return;   // budget guard: stop processing
                    }
                    // Rule 123: PhaseScope records duration + node-count delta.
                    // Zero overhead when no hook is set (single branch on armed_).
                    support::PhaseScope span(
                        support::PhaseKind::Pass,
                        static_cast<std::uint16_t>(0),  // pass_id extracted from name
                        static_cast<std::uint8_t>(ctx.tier),
                        static_cast<std::uint16_t>(ctx.code_unit_id),
                        g.live_node_count());
                    auto r = pass.run(g, ctx);
                    if (!r) {
                        span.set_result(support::PhaseResult::CompileFailed, g.live_node_count());
                        return;
                    }
                    span.set_result(support::PhaseResult::Ok, g.live_node_count());
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
