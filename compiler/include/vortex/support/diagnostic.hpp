// =============================================================================
// vortex/support/diagnostic.hpp — Actionable diagnostics (Rule 47)
//
// Purpose:
//   Every failure in the compiler/runtime carries a Diagnostic with:
//     1. exact source location (file, line, column, function name),
//     2. human-readable message,
//     3. expected-vs-actual state,
//     4. suggested fix.
//   Rule 47 bans opaque errors; the formatter guarantees all four fields.
//
// Invariants:
//   - Fixed-size storage (no heap) so constructing a Diagnostic on a failing
//     hot path never allocates (Rule 7).
//   - Severity is exhaustive; switch sites use VORTEX_UNREACHABLE (Rule 58).
// =============================================================================

#pragma once

#include <cstdint>
#include <cstdio>
#include <string_view>

namespace vortex {

inline namespace abi_v1 {

enum class Severity : std::uint8_t {
    Note,
    Warning,
    Error,
    Fatal,
};

struct SourceLocation {
    std::string_view file{};
    std::uint32_t line{0};
    std::uint32_t column{0};
    std::string_view function{};

    [[nodiscard]] static constexpr SourceLocation here(std::string_view f, std::uint32_t l,
                                                       std::uint32_t c,
                                                       std::string_view fn) noexcept {
        return SourceLocation{f, l, c, fn};
    }
};

/// Capture site cheaply at call sites.
#define VORTEX_SRC_LOC ::vortex::SourceLocation{__FILE__, __LINE__, 0, __func__}

/// Fixed-capacity diagnostic: message/expected/actual/fix each capped at
/// `diagnostic_text_capacity` bytes (truncation marked with '~').
inline constexpr std::size_t diagnostic_text_capacity = 160;

struct Diagnostic {
    Severity severity{Severity::Error};
    SourceLocation where{};
    std::string_view message{};
    std::string_view expected{};
    std::string_view actual{};
    std::string_view fix{};
    /// Stable machine code for telemetry aggregation (Rule 26) — see
    /// vortex/support/telemetry.hpp for the catalog.
    std::uint32_t code{0};

    [[nodiscard]] static Diagnostic note(std::string_view msg, std::uint32_t code = 0) noexcept {
        return Diagnostic{Severity::Note, {}, msg, {}, {}, {}, code};
    }
    [[nodiscard]] static Diagnostic error(std::string_view msg, std::uint32_t code = 0) noexcept {
        return Diagnostic{Severity::Error, {}, msg, {}, {}, {}, code};
    }
    [[nodiscard]] static Diagnostic fatal(std::string_view msg, std::uint32_t code = 0) noexcept {
        return Diagnostic{Severity::Fatal, {}, msg, {}, {}, {}, code};
    }

    /// Rule 47 formatter: emits location, severity, message, expected vs
    /// actual, and suggested fix in one deterministic, greppable line block.
    void report(std::FILE* out = stderr) const noexcept;

    [[nodiscard]] constexpr bool is_error() const noexcept {
        return severity == Severity::Error || severity == Severity::Fatal;
    }
};

// --- Machine-readable diagnostic codes (telemetry catalog, Rule 26) ----------
namespace diag_code {
inline constexpr std::uint32_t lex_invalid_char = 1;
inline constexpr std::uint32_t lex_unterminated_string = 2;
inline constexpr std::uint32_t parse_unexpected_token = 3;
inline constexpr std::uint32_t parse_unbalanced_delim = 4;
inline constexpr std::uint32_t name_undefined = 100;
inline constexpr std::uint32_t type_mismatch = 101;
inline constexpr std::uint32_t arity_mismatch = 102;
inline constexpr std::uint32_t guard_failed_deopt = 200;
inline constexpr std::uint32_t graph_verify_dangling_node = 300;
inline constexpr std::uint32_t graph_verify_effect_chain = 301;
inline constexpr std::uint32_t graph_verify_dominance = 302;
inline constexpr std::uint32_t graph_verify_framestate = 303;
inline constexpr std::uint32_t graph_verify_use_def = 304;
inline constexpr std::uint32_t budget_exceeded = 400;
inline constexpr std::uint32_t regalloc_spill_excess = 401;
inline constexpr std::uint32_t jit_bailout = 500;
inline constexpr std::uint32_t aot_artifact_version = 600;
inline constexpr std::uint32_t arena_exhausted = 700;
inline constexpr std::uint32_t runtime_type_error = 800;
inline constexpr std::uint32_t runtime_index_error = 801;
inline constexpr std::uint32_t runtime_key_error = 802;
inline constexpr std::uint32_t runtime_zero_division = 803;
inline constexpr std::uint32_t runtime_attribute_error = 804;
inline constexpr std::uint32_t runtime_value_error = 805;
inline constexpr std::uint32_t runtime_stop_iteration = 806;
inline constexpr std::uint32_t runtime_overflow = 807;
inline constexpr std::uint32_t runtime_not_implemented = 808;
inline constexpr std::uint32_t runtime_recursion = 809;
inline constexpr std::uint32_t runtime_memory = 810;
}  // namespace diag_code

}  // namespace abi_v1
}  // namespace vortex
