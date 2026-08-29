// =============================================================================
// vortex/passes/pass_manager.hpp — Unified pass pipeline (Rules 1, 2, 10, 45)
//
// Purpose:
//   One pipeline, multiple inputs (Rule 1): every pass runs in every tier,
//   gated by a TierMode that describes what evidence it may use:
//     Static  — IR only (Tier 1 budget-constrained, Tier 3 proof-required)
//     Profile — IR + PGO data (Tier 2, guard-emitting)
//   PGO never changes WHAT a pass does, only HOW AGGRESSIVELY (Rule 2).
//
// Pass contract (Rules 10, 40, 26):
//   - run(Graph&, const PassContext&) returns Result<PassResult>:
//     PassResult reports nodes-before/after and whether the graph changed.
//   - Idempotence (Rule 10): a second run must be a no-op; the manager
//     verifies this in debug builds by double-running every pass.
//   - The verifier (Rule 40) runs after EVERY pass in debug builds.
//   - Budget breaches are TELEMETRY events (Rule 26), never silent.
//
// No std::function, no virtual dispatch in the hot loop (Rule 9): the
// pipeline is a compile-time tuple iterated with fold expressions.
// =============================================================================

#pragma once

#include <cstdint>

#include "vortex/backend/target.hpp"
#include "vortex/ir/graph.hpp"
#include "vortex/support/config.hpp"
#include "vortex/support/flags.hpp"
#include "vortex/support/telemetry.hpp"

namespace vortex::passes {

inline namespace abi_v1 {

enum class TierMode : std::uint8_t {
    Tier1,   // budget-constrained baseline JIT: cheap passes only
    Tier2,   // optimizing JIT: PGO-driven, guard-emitting
    Tier3,   // AOT/static: proofs only, zero guards
};

/// Opt-out optimization switches (Rule 23: named, no magic booleans).
///
/// These gate transforms whose compile-time cost is high enough that the
/// caller may want to skip them in compile-time-sensitive contexts. The
/// pass is ON BY DEFAULT — setting the corresponding flag here opts OUT.
/// A flag set here is advisory to the pipeline filter AND self-checked by
/// the pass itself (defense in depth: direct invocations are gated
/// identically).
enum class OptOption : std::uint8_t {
    /// Pass 33 — polyhedral loop transforms (interchange/skew/tiling).
    /// DEFAULT-ON. The only legitimate reason to set this flag is
    /// compilation-time sensitivity on hot-loop-heavy code (the analysis
    /// is O(N + L + ΣA·D) — linear for bounded D, non-trivial in
    /// absolute terms). Maps to CompileOptions::disable_polyhedral in
    /// the runtime driver.
    DisablePolyhedral = 1 << 0,
};

/// Input evidence bundle per compilation (Rule 1's "multiple inputs").
struct PassContext {
    TierMode tier{TierMode::Tier1};
    std::uint32_t node_budget{cfg::tier1_node_budget};
    std::uint32_t code_unit_id{0};
    Telemetry* telemetry{nullptr};

    /// Opt-out optimization switches — empty by default (Rule 23).
    /// Empty means "run all default-on passes". Setting a flag here
    /// opts the corresponding pass OUT (the inverse of opt-in).
    Flags<OptOption> options{};

    /// Pointer to the module's cooked string-pool (Rule 5: pool bytes are
    /// the only source for string-literal contents). Null when the pass
    /// pipeline runs outside a module context (e.g., direct unit tests of
    /// a pass with a synthetic graph). Passes that need to read string
    /// literals (Pass 45 constant folding of str+str) MUST null-check
    /// before indexing into the pool.
    const stdx::small_vector<char, 4096>* string_pool{nullptr};

    /// Target the graph is being compiled for. Machine facts (SIMD width,
    /// register count, cache line) are QUERIED from here — never assumed
    /// from cfg constants (Rule 27/24). Null means "no target knowledge":
    /// target-sensitive passes (vectorization width) must decline.
    const backend::TargetDescriptor* target{nullptr};

    // PGO inputs (valid when tier == Tier2):
    //   - call-site type histograms
    //   - loop trip counts
    //   - IC histories
    // (Feeder API implemented by the profiler; passes read-only.)
    const void* pgo{nullptr};

    [[nodiscard]] bool is_profiled() const noexcept { return tier == TierMode::Tier2; }
    [[nodiscard]] bool requires_proofs() const noexcept { return tier == TierMode::Tier3; }
};

struct PassResult {
    std::uint32_t nodes_before{0};
    std::uint32_t nodes_after{0};
    bool changed{false};
    [[nodiscard]] std::int64_t delta() const noexcept {
        return static_cast<std::int64_t>(nodes_after) -
               static_cast<std::int64_t>(nodes_before);
    }
};

/// Budget guard result (Rule 10: growing passes run in bounded fixpoints).
struct BudgetCheck {
    bool exceeded{false};
    std::uint32_t iterations_used{0};
};

/// A pass: pure function over the graph + context. Plain structs with a
/// static name and a run() — no allocation, no virtual dispatch (Rule 9).
struct PassBase {
    const char* name{""};
    std::uint16_t number{0};   // canonical pipeline number (1..55)
};

/// Monotonic + idempotent fixpoint driver for a single pass (Rule 10):
/// runs up to cfg::fixpoint_max_iterations while the graph keeps changing.
template <typename P>
[[nodiscard]] Result<PassResult> run_to_fixpoint(P& pass, ir::Graph& g,
                                                 const PassContext& ctx) noexcept;

/// Verifier hook (Rule 40): set in debug builds.
extern bool (*g_verify_after_each_pass)(const ir::Graph&, const char*);

}  // namespace abi_v1
}  // namespace vortex::passes

#include "vortex/passes/passes_fwd.hpp"
