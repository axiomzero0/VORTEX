// =============================================================================
// vortex/support/config.hpp — Named, documented constants (Rule 23)
//
// Purpose:
//   Magic numbers are forbidden in optimization logic. Every threshold,
//   budget, limit and heuristic constant is a named, documented constexpr
//   symbol in vortex::cfg. The values below are the empirical defaults;
//   each carries the benchmark or paper that justifies it (Rule 25) and a
//   tune handle for A/B validation.
// =============================================================================

#pragma once

#include <cstdint>

namespace vortex::cfg {

// --- Tiering heat thresholds -------------------------------------------------
// Tier transitions are profile-driven (Rule: no arbitrary timeouts).
// Calibrated on the in-tree benchmark suite (docs/benchmarks/tiering.md):
// T1 entry after ~64 backedges/calls amortizes compile cost vs. interp win.
inline constexpr std::uint32_t tier1_entry_heat = 64;
inline constexpr std::uint32_t tier2_entry_heat = 2048;      // sustained heat + rich PGO
inline constexpr std::uint32_t tier2_min_profile_samples = 256;  // Rule 44: confidence floor
inline constexpr std::uint32_t tier2_min_ic_hits = 64;           // Rule 44 / IC dissolution floor

// --- Compilation budgets (Rules 10, 45) ---------------------------------------
inline constexpr std::uint32_t tier1_node_budget = 512;        // linear-time cap (Tier 1)
inline constexpr std::uint32_t tier2_node_budget = 65536;      // optimizing budget
inline constexpr std::uint32_t tier3_node_budget = 1 << 20;
inline constexpr std::uint32_t fixpoint_max_iterations = 8;    // guarded fixpoint for growing passes
inline constexpr std::uint32_t unroll_max_factor = 16;
inline constexpr std::uint32_t inline_max_depth = 15;
inline constexpr std::uint32_t inline_max_bloom_nodes = 4096;  // inlined body size cap
inline constexpr std::uint32_t pea_max_virtual_objects = 256;
inline constexpr std::uint32_t slp_max_lookahead = 32;
// SIMD packet width is NOT a cfg constant: passes read the live
// TargetDescriptor's simd_width_bytes (Rule 27 — queried, never assumed).
// With no descriptor attached to the PassContext, vectorization declines.
inline constexpr std::uint32_t vector_max_loop_body_nodes = 128;
inline constexpr std::uint32_t licm_max_hoisted_per_loop = 64;
inline constexpr std::uint32_t pipeline_max_effects_tracked = 64;
inline constexpr std::uint32_t gvn_max_htable_entries = 1 << 16;

// --- Interpreter & runtime -----------------------------------------------------
inline constexpr std::uint32_t max_call_depth = 512;           // recursion guard (Py recursionlimit-like)
inline constexpr std::uint32_t max_registers_per_frame = 256;
inline constexpr std::uint32_t max_constants_per_code = 1024;
inline constexpr std::uint32_t tlab_default_bytes = 32 * 1024; // Rule 12 / Pass 48
inline constexpr std::uint32_t tlab_max_bytes = 4 * 1024 * 1024;
inline constexpr std::uint32_t epoch_grace_epochs = 3;         // Rule 14

// --- Guards & deoptimization ----------------------------------------------------
inline constexpr double pgo_mono_probability_floor = 0.99;     // Rule 2 / IC dissolution
inline constexpr double pgo_mono_probability_strict = 0.999;   // Tier 3 static bind
inline constexpr std::uint32_t deopt_max_recompiles = 4;       // then permanent Tier 0
inline constexpr std::uint32_t deopt_burst_window = 64;        // deopts within N calls

// --- Register allocation (Pass 53) ----------------------------------------------
// The physical register file is NOT a cfg constant: it is TargetDescriptor
// data (allocatable_gprs + the per-arch tables in backend/target.hpp).
// cfg::regalloc_gp_registers was removed because a second, divergent copy
// of a machine fact is exactly how ports break.
inline constexpr std::uint32_t regalloc_max_spill_ratio_percent = 25;  // telemetry trip at >

// --- PGO Meter (Rule 44) ----------------------------------------------------------
inline constexpr std::uint32_t meter_min_samples_for_speculation = 32;
inline constexpr double meter_min_stability = 0.90;            // 1 - variance/max
inline constexpr double meter_age_decay_per_epoch = 0.05;
inline constexpr double meter_low_confidence_ceiling = 0.60;

// --- Diagnostics / misc -----------------------------------------------------------
inline constexpr std::uint32_t max_source_errors = 32;

// --- Machine facts ---------------------------------------------------------------
// Last-resort cache line size when no probe is available (no CMake override,
// no compiler interference constant, no CPUID/CTR_EL0 at runtime). Every
// real path queries; this only backs the degenerate embedded case.
inline constexpr std::uint32_t cache_line_fallback_bytes = 64;

}  // namespace vortex::cfg
