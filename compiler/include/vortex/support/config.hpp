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

// --- SLP vectorization (Pass 31) -------------------------------------------------
// Calibrated on the in-tree micro-benchmarks (docs/benchmarks/slp.md):
//  - slp_min_packet_lanes: 2 is the floor where a VecOp can possibly
//    pay after insert/extract overhead. Single-lane "packets" are
//    scalar code in disguise and would just bloat the IR.
//  - slp_max_shuffle_ratio: a packet is rejected if the alignment
//    shuffle count exceeds this fraction of the lane count (Larsen &
//    Amarasinghe 2000, §4.3 — "alignment overhead must be amortized
//    across at least 2:1 vector ops per shuffle").
//  - slp_alias_guard_pgo_floor: hot LoadIndex pairs below this PGO
//    count are not worth a speculative guard — the deopt risk on a
//    cold pair dominates the vectorization win.
//  - slp_gather_min_lanes: fewer than this many pointer-array loads
//    gathered is not worth the gather instruction latency (Rule 45).
inline constexpr std::uint32_t slp_min_packet_lanes = 2;
inline constexpr std::uint32_t slp_max_shuffle_ratio_percent = 50;   // 50% == packet.size()/2
inline constexpr std::uint32_t slp_alias_guard_pgo_floor = 64;        // = tier2_min_ic_hits
inline constexpr std::uint32_t slp_gather_min_lanes = 2;

// --- Dict layout specialization (Pass 46) ----------------------------------------
// Calibrated on the in-tree micro-benchmarks (docs/benchmarks/dict_layout.md):
//  - dict_layout_max_fields: above this many keys the fixed-offset struct
//    layout stops paying — the per-field guard overhead exceeds the
//    hash-table probe cost it eliminates. 32 is the V8 hidden-class limit
//    (cited in V8's transition-tree documentation) and matches empirical
//    Python workload distributions (most record-like dicts have ≤8 keys).
//  - dict_layout_min_fields: below this many keys the layout win is too
//    small to justify the shape guard. Single-key dicts are better left
//    as hash tables (one probe, no guard).
inline constexpr std::uint32_t dict_layout_max_fields = 32;
inline constexpr std::uint32_t dict_layout_min_fields = 2;

// --- Inline cache dissolution (Pass 16) ------------------------------------------
// Calibrated on the in-tree micro-benchmarks (docs/benchmarks/ic.md):
//  - ic_devirt_pgo_floor: hot call sites below this PGO count are not
//    worth a speculative type guard — the deopt risk on a cold site
//    dominates the dispatch savings. Matches tier2_min_ic_hits (the IC
//    dissolution floor from Rule 44) so that a site with enough hits to
//    be considered "hot" also has enough data for the type guard to be
//    statistically sound.
inline constexpr std::uint32_t ic_devirt_pgo_floor = 64;   // = tier2_min_ic_hits

// --- Inline cache polymorphism tiers (Pass 16b) ----------------------------------
// Calibrated on V8/SpiderMonkey IC tiering research (docs/benchmarks/ic.md):
//  - ic_mono_max_types: 1 type. The site is monomorphic — P16 dissolves it
//    into a GuardedDirectCall with a single type guard.
//  - ic_poly_max_types: 4 types. The site is polymorphic — P16b emits a
//    DispatchCache node carrying the type→target mappings. The backend
//    lowers this to a linear type-check chain (future: SIMD comparison).
//    V8 uses 4 as the polymorphic ceiling before megamorphic fallback.
//  - ic_mega_pgo_floor: above this many observed types (in the PGO
//    histogram stored in the CallPy node's aux0), the site is megamorphic
//    — the pass declines to emit an IC and falls back to standard dispatch.
//    20 matches V8's megamorphic threshold (cited in V8's IC internals doc).
inline constexpr std::uint32_t ic_mono_max_types = 1;
inline constexpr std::uint32_t ic_poly_max_types = 4;
inline constexpr std::uint32_t ic_mega_pgo_floor = 20;

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
