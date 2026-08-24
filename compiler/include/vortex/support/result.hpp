// =============================================================================
// vortex/support/result.hpp — Zero-cost error propagation (Rules 6, 22, 48)
//
// Purpose:
//   Result<T> = std::expected<T, Diagnostic>. The entire compiler and runtime
//   are compiled -fno-exceptions; every fallible operation returns Result.
//   TRY(...) is the sanctioned single-branch propagation macro (Rule 22).
//
// Invariants:
//   - All Result-returning functions are [[nodiscard]] (Rule 48): ignoring an
//     error is a compile failure via -Wunused-result.
//   - TRY has exactly one branch on the success path's flag register.
//   - Result<void> models fallible procedures.
//
// Edge cases:
//   - Moving a Result moves the value; Diagnostic is trivially copyable.
// =============================================================================

#pragma once

#include <expected>
#include <utility>

#include "vortex/support/diagnostic.hpp"

namespace vortex {

inline namespace abi_v1 {

template <typename T>
using Result = std::expected<T, Diagnostic>;

using Fail = std::unexpected<Diagnostic>;

/// Construct a failure inline: `return vortex::fail(diag_code::...)` or
/// `return vortex::fail_msg("...", code)`.
[[nodiscard]] inline Fail fail(Diagnostic d) noexcept { return Fail(std::move(d)); }
[[nodiscard]] inline Fail fail_msg(std::string_view msg, std::uint32_t code = 0) noexcept {
    return Fail(Diagnostic::error(msg, code));
}

}  // namespace abi_v1
}  // namespace vortex

// -----------------------------------------------------------------------------
// TRY — Rule 22 sanctioned macro. Compiles to a single branch; keeps the hot
// path's instruction cache pristine. Usage:
//     auto value = VORTEX_TRY(expr);       // Result<T>  -> unwraps to T
//     VORTEX_TRY_VOID(expr);               // Result<void> -> propagates only
// inside any function returning vortex::Result<...>.
// -----------------------------------------------------------------------------
#define VORTEX_TRY_VOID(...)                                \
    do {                                                    \
        auto _vx_try_result = (__VA_ARGS__);                \
        if (!_vx_try_result) [[unlikely]] {                 \
            return std::unexpected(_vx_try_result.error()); \
        }                                                   \
    } while (0)
#define VORTEX_TRY(...)                                     \
    ({                                                      \
        auto _vx_try_result = (__VA_ARGS__);                \
        if (!_vx_try_result) [[unlikely]] {                 \
            return std::unexpected(_vx_try_result.error()); \
        }                                                   \
        std::move(*_vx_try_result);                         \
    })
