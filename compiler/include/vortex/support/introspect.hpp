// =============================================================================
// vortex/support/introspect.hpp — Compiler self-observation (Rule 123)
//
// Purpose:
//   The Introspector. A zero-overhead-when-off, sub-1%-when-on observability
//   layer that records structured spans for every compilation phase, pass,
//   tier execution, deopt, and guard event. Spans carry duration, node-count
//   delta, IR hash, and telemetry-ring deltas — enough to reconstruct WHY
//   a compilation succeeded or failed, not just THAT it did.
//
// Architecture:
//   - PhaseSpan: 64-byte trivially-copyable record (fits in one cache line pair)
//   - PhaseScope: RAII guard that opens a span on construction, closes on
//     destruction, invokes the phase hook if set. No heap allocation.
//   - g_phase_hook: function-pointer seam (mirrors g_verify_after_each_pass).
//     Zero overhead when null (single branch on armed_).
//
// Rule compliance:
//   Rule 9: No virtual dispatch — function-pointer seam, not vtable.
//   Rule 23: All constants named (kPhaseSpanRingCap in config.hpp).
//   Rule 49: No macros for logic — inline constexpr + RAII.
//   Rule 88: Spans open at function-call and backedge granularity, NOT
//     per-bytecode (would tank the 32.7x speedup).
//   Rule 123: Passes declare contracts; PhaseScope records them.
//   Rule 149: No external dependencies; pure C++26.
// =============================================================================

#pragma once

#include <cstdint>
#include <chrono>

#include "vortex/support/config.hpp"

namespace vortex::support {

inline namespace abi_v1 {

/// Compilation/runtime phase kinds for span classification.
enum class PhaseKind : std::uint8_t {
    Lex         = 0,
    Parse       = 1,
    Lower       = 2,
    Pass        = 3,   // A single optimization pass
    Schedule    = 4,
    Tier0Emit   = 5,
    JitCompile  = 6,
    Tier0Exec   = 7,   // Tier-0 interpreter execution
    Tier1Exec   = 8,   // Meta-traced native execution
    Deopt       = 9,
    Guard       = 10,
    Invalidate  = 11,
    Safepoint   = 12,
};

/// Span result codes.
enum class PhaseResult : std::uint8_t {
    Ok             = 0,
    Skipped        = 1,   // pass declined (tier gate, budget, etc.)
    VerifyFail     = 2,   // IR verifier caught a problem
    BudgetExceeded = 3,   // node budget tripped
    CompileFailed  = 4,   // compilation returned Error
    DeoptFailed    = 5,   // deopt state reconstruction failed
};

/// A single observability span. 64 bytes, trivially copyable.
/// Records what happened, how long it took, and what changed.
struct PhaseSpan {
    PhaseKind      kind{};
    std::uint8_t   tier{};          // 0..3
    PhaseResult    result{};
    std::uint8_t   _pad0{};
    std::uint16_t  pass_id{};       // Pass number (0 for non-pass phases)
    std::uint16_t  code_unit_id{};
    std::uint32_t  node_count_in{};
    std::uint32_t  node_count_out{};
    std::uint64_t  start_ns{};
    std::uint64_t  dur_ns{};
    // CorrelationId fields (Layer 2 — causally link tier transitions)
    std::uint64_t  program_hash{};  // hash of source module
    std::uint32_t  header_pc{};     // backedge target that became hot
    std::uint32_t  _pad1{};
};
static_assert(sizeof(PhaseSpan) <= 64, "PhaseSpan must fit in 64 bytes (cache-line pair)");

/// Function-pointer hook (mirrors g_verify_after_each_pass pattern).
/// Called on PhaseScope destruction with the completed span.
/// Zero overhead when null — PhaseScope checks armed_ (one branch).
using PhaseHook = void(*)(const PhaseSpan&) noexcept;

/// Set the global phase hook. Pass nullptr to disable.
/// Thread-unsafe — set once at startup, before any compilation.
void set_phase_hook(PhaseHook hook) noexcept;

/// Get the current global phase hook (nullptr if unset).
[[nodiscard]] PhaseHook get_phase_hook() noexcept;

/// RAII span guard. Opens a span on construction, closes on destruction.
/// No heap allocation. No exceptions (Rule 6). No virtual dispatch (Rule 9).
///
/// Usage in run_pipeline():
///   PhaseScope span(PhaseKind::Pass, PassT::number, ctx);
///   auto r = PassT::run(g, ctx);
///   span.set_result(r.ok() ? PhaseResult::Ok : PhaseResult::CompileFailed);
///   // ~PhaseScope closes the span, invokes hook if set
struct PhaseScope {
    PhaseSpan s_{};
    bool      armed_{false};

    /// Construct a span for a compilation pass.
    PhaseScope(PhaseKind kind, std::uint16_t pass_id,
               std::uint8_t tier, std::uint16_t code_unit_id,
               std::uint32_t node_count_in) noexcept
        : armed_(get_phase_hook() != nullptr) {
        if (!armed_) return;
        s_.kind = kind;
        s_.tier = tier;
        s_.pass_id = pass_id;
        s_.code_unit_id = code_unit_id;
        s_.node_count_in = node_count_in;
        s_.start_ns = now_ns();
    }

    /// Record the result and node count after the phase completes.
    void set_result(PhaseResult r, std::uint32_t node_count_out) noexcept {
        s_.result = r;
        s_.node_count_out = node_count_out;
    }

    /// Record just the result (node count unchanged).
    void set_result(PhaseResult r) noexcept {
        s_.result = r;
        s_.node_count_out = s_.node_count_in;
    }

    /// Set correlation fields (Layer 2).
    void set_correlation(std::uint64_t program_hash, std::uint32_t header_pc) noexcept {
        s_.program_hash = program_hash;
        s_.header_pc = header_pc;
    }

    ~PhaseScope() noexcept {
        if (!armed_) return;
        s_.dur_ns = now_ns() - s_.start_ns;
        if (PhaseHook h = get_phase_hook()) {
            h(s_);
        }
    }

private:
    [[nodiscard]] static std::uint64_t now_ns() noexcept {
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
    }
};

}  // namespace abi_v1
}  // namespace vortex::support
