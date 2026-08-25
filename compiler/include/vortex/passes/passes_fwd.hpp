// =============================================================================
// vortex/passes/passes_fwd.hpp — Pass declarations (55 passes).
//
// Every pass follows the unified interface (Rule 1):
//   struct PassNN_Name {
//       static constexpr const char* name = "NN_name";
//       [[nodiscard]] Result<PassResult> run(ir::Graph&, const PassContext&) noexcept;
//   };
//
// Grouping mirrors the spec:
//   Phase 1 (1-10):   IR normalization & intraprocedural foundation
//   Phase 2 (11-18):  Alias & pointer analysis
//   Phase 3 (19-26):  Interprocedural & speculative inlining
//   Phase 4 (27-38):  Loop optimizations & vectorization
//   Phase 5 (39-48):  Memory, allocation, escape analysis
//   Phase 6 (49-55):  Late optimizations & backend preparation
// =============================================================================

#pragma once

#include "vortex/passes/pass_manager.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

// --- Phase 1: IR normalization ------------------------------------------------
struct P01_FrontendLowering {
    static constexpr const char* name = "01_frontend_lowering";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P02_SeaOfNodesConstruction {
    static constexpr const char* name = "02_son_construction";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P03_TrivialDCE {
    static constexpr const char* name = "03_trivial_dce";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P04_LocalConstantFolding {
    static constexpr const char* name = "04_local_constfold";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P05_AlgebraicSimplification {
    static constexpr const char* name = "05_algebraic_simplify";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P06_ControlFlowSimplification {
    static constexpr const char* name = "06_cf_simplify";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P07_LocalCSE {
    static constexpr const char* name = "07_local_cse";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P08_SCCP {
    static constexpr const char* name = "08_sccp";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P09_RedundantStoreElimination {
    static constexpr const char* name = "09_redundant_stores";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P10_EarlyGVN {
    static constexpr const char* name = "10_early_gvn";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};

// --- Phase 2: Alias & pointer analysis -------------------------------------------
struct P11_AndersenPointsTo {
    static constexpr const char* name = "11_andersen";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P12_CFLReachabilityAlias {
    static constexpr const char* name = "12_cfl_alias";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P13_FlowSensitiveAlias {
    static constexpr const char* name = "13_flow_alias";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P14_DemandDrivenAlias {
    static constexpr const char* name = "14_demand_alias";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P15_ShapeAnalysis {
    static constexpr const char* name = "15_shape_analysis";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P16_ICMonomorphism {
    static constexpr const char* name = "16_ic_mono";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P17_MROLinearization {
    static constexpr const char* name = "17_mro";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P18_SideEffectAnalysis {
    static constexpr const char* name = "18_effects";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};

// --- Phase 3: Interprocedural & speculative inlining ------------------------------
struct P19_CallGraphPGO {
    static constexpr const char* name = "19_callgraph_pgo";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P20_SpeculativeInlining {
    static constexpr const char* name = "20_spec_inline";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P21_PartialInlining {
    static constexpr const char* name = "21_partial_inline";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P22_RecursiveInlining {
    static constexpr const char* name = "22_recursive_inline";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P23_ClosureDevirtualization {
    static constexpr const char* name = "23_closure_devirt";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P24_GeneratorDeforestation {
    static constexpr const char* name = "24_gen_deforest";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P25_ExceptionOutlining {
    static constexpr const char* name = "25_exc_outline";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P26_IPCP {
    static constexpr const char* name = "26_ipcp";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};

// --- Phase 4: Loops & vectorization ------------------------------------------------
struct P27_LICM {
    static constexpr const char* name = "27_licm";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P28_SpeculativeLICM {
    static constexpr const char* name = "28_spec_licm";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P29_InductionVariables {
    static constexpr const char* name = "29_iv_strength";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P30_LoopUnrolling {
    static constexpr const char* name = "30_unroll";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P31_SLPVectorization {
    static constexpr const char* name = "31_slp";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P32_LoopVectorization {
    static constexpr const char* name = "32_loop_vec";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P33_PolyhedralOptimization {
    static constexpr const char* name = "33_polyhedral";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P34_SoftwarePipelining {
    static constexpr const char* name = "34_sw_pipeline";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P35_LoopFusionFission {
    static constexpr const char* name = "35_fusion_fission";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P36_BoundsCheckElimination {
    static constexpr const char* name = "36_bce";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P37_NoneCheckElimination {
    static constexpr const char* name = "37_nce";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P38_GILHoisting {
    static constexpr const char* name = "38_gil_hoist";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};

// --- Phase 5: Memory, allocation, escape analysis -----------------------------------
struct P39_EscapeAnalysis {
    static constexpr const char* name = "39_escape";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P40_PartialEscapeAnalysis {
    static constexpr const char* name = "40_pea";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P41_ObjectInlining {
    static constexpr const char* name = "41_obj_inline";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P42_RegionMemoryInference {
    static constexpr const char* name = "42_regions";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P43_RefcntOptimization {
    static constexpr const char* name = "43_refcnt";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P44_WriteBarrierElimination {
    static constexpr const char* name = "44_write_barriers";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P45_StringInterning {
    static constexpr const char* name = "45_string_intern";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P46_DictLayoutSpecialization {
    static constexpr const char* name = "46_dict_layout";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P47_BoxUnboxElimination {
    static constexpr const char* name = "47_unboxing";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P48_TLABSizing {
    static constexpr const char* name = "48_tlab";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};

// --- Phase 6: Late optimizations ---------------------------------------------------
struct P49_SpeculativeEffectReordering {
    static constexpr const char* name = "49_effect_reorder";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P50_LateGVN {
    static constexpr const char* name = "50_late_gvn";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
struct P51_GlobalDCE {
    static constexpr const char* name = "51_global_dce";
    [[nodiscard]] Result<PassResult> run(ir::Graph& g, const PassContext& c) noexcept;
};
// Passes 52-55 are backend passes (instruction selection, regalloc,
// peephole, layout) — they live in vortex/backend/.

}  // namespace abi_v1
}  // namespace vortex::passes
