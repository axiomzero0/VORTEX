// =============================================================================
// vortex/support/flags.hpp — Type-safe bitmask flags (Rule 32)
//
// Purpose:
//   All orthogonal boolean state on hot-path structures (NodeFlags,
//   EffectTags, ...) is a bitmask wrapped in Flags<E>. Raw integers are
//   forbidden for flag-like state.
//
// Invariants:
//   - Enum values must be distinct powers of two (static_asserted per user).
//   - Flags<E> is a trivially copyable wrapper around the underlying integer;
//     zero overhead vs raw bitmask, full type safety.
// =============================================================================

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "vortex/support/assume.hpp"

namespace vortex {

inline namespace abi_v1 {

/// Scoped-enum cardinality: Count sentinel -> std::size_t. Scoped enums
/// have no implicit integral conversion (the classic unary-plus trick only
/// works on unscoped enums), so array bounds and loop limits use this.
template <typename E>
    requires std::is_enum_v<E>
[[nodiscard]] consteval std::size_t enum_size(E count_sentinel) noexcept {
    return static_cast<std::size_t>(count_sentinel);
}

template <typename E>
    requires std::is_enum_v<E>
class Flags {
    static_assert(std::is_integral_v<std::underlying_type_t<E>>,
                  "flag enums need integral underlying types");
public:
    using underlying = std::underlying_type_t<E>;

    constexpr Flags() noexcept = default;
    constexpr Flags(E single) noexcept : bits_(static_cast<underlying>(single)) {}

    [[nodiscard]] constexpr underlying raw() const noexcept { return bits_; }

    [[nodiscard]] constexpr bool has(E flag) const noexcept {
        return (bits_ & static_cast<underlying>(flag)) != 0;
    }
    [[nodiscard]] constexpr bool has_all(Flags other) const noexcept {
        return (bits_ & other.bits_) == other.bits_;
    }
    [[nodiscard]] constexpr bool has_any(Flags other) const noexcept {
        return (bits_ & other.bits_) != 0;
    }
    [[nodiscard]] constexpr bool empty() const noexcept { return bits_ == 0; }

    constexpr Flags& set(E flag) noexcept {
        bits_ |= static_cast<underlying>(flag);
        return *this;
    }
    constexpr Flags& clear(E flag) noexcept {
        bits_ &= ~static_cast<underlying>(flag);
        return *this;
    }
    constexpr Flags& clear_all() noexcept { bits_ = 0; }

    [[nodiscard]] constexpr Flags operator|(Flags o) const noexcept { return Flags(bits_ | o.bits_); }
    [[nodiscard]] constexpr Flags operator&(Flags o) const noexcept { return Flags(bits_ & o.bits_); }
    constexpr Flags& operator|=(Flags o) noexcept {
        bits_ |= o.bits_;
        return *this;
    }
    [[nodiscard]] constexpr bool operator==(const Flags&) const noexcept = default;

    /// Iterate set bits — returns the E value at slot `i` (i < popcount).
    [[nodiscard]] constexpr E at(std::size_t i) const noexcept {
        underlying remaining = bits_;
        for (std::size_t seen = 0;; ++seen) {
            underlying low = remaining & (~remaining + 1);  // lowest set bit
            if (seen == i) return static_cast<E>(low);
            remaining ^= low;
        }
    }
    [[nodiscard]] constexpr std::size_t popcount() const noexcept {
        return static_cast<std::size_t>(__builtin_popcountll(bits_));
    }

private:
    constexpr explicit Flags(underlying bits) noexcept : bits_(bits) {}
    underlying bits_{0};
};

template <typename E>
[[nodiscard]] constexpr Flags<E> operator|(E a, E b) noexcept {
    return Flags<E>(a) | Flags<E>(b);
}

}  // namespace abi_v1
}  // namespace vortex
