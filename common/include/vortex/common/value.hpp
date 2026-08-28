// =============================================================================
// vortex/common/value.hpp — NaN-boxed runtime value (8 bytes, fits in one GPR)
//
// Purpose:
//   One 8-byte trivially-copyable value representation using NaN-boxing.
//   Halves memory bandwidth per register read/write vs the old 16-byte
//   Value (tag + padding + 8-byte union). On x86-64, an 8-byte value fits
//   in a single GPR — no partial register stalls, no extra memory traffic.
//
// NaN-boxing scheme (SpiderMonkey/LuaJIT-style):
//   IEEE 754 doubles have a "NaN space": any 64-bit pattern with bits
//   52-62 = 0x7FF (exponent all-ones) and non-zero mantissa is a NaN.
//   We use the top 16 bits (the NaN tag space) to encode non-double tags:
//
//   Bits 63-48 (top 16):
//     0x0000-0xFFEF: Double — the full 64 bits ARE the IEEE 754 double.
//                    Canonical NaN (0x7FF8...) has bit 63=0, safe.
//     0xFFF0:        None
//     0xFFF1:        Bool (bit 0 of payload = 0/1)
//     0xFFF2:        Int (bits 47-0 = int48, sign-extended)
//     0xFFF3:        Obj (bits 47-0 = pointer; x86-64 user-space ptrs ≤47 bits)
//
// Invariants:
//   - 8 bytes, trivially copyable, fits in a single GPR.
//   - Doubles are stored as-is (zero overhead for float arithmetic).
//   - Ints are 48-bit; int64 values that don't fit in 48 bits are promoted
//     to a bignum PyObj. This matches Python's small-int optimization.
//   - Pointers are 48-bit (all x86-64 user-space pointers fit in 47 bits).
//   - sizeof(Value) == 8, alignof(Value) == 8.
//
// Performance impact:
//   Every register read/write moves 8 bytes instead of 16. For a tight loop
//   doing 4 register reads + 2 writes per iteration, this saves 48 bytes
//   of memory traffic per iteration — ~2x throughput for compute-bound loops.
// =============================================================================

#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>

namespace vortex {

inline namespace abi_v1 {

struct PyObj;   // runtime object header — full definition in vortex/rt/object.hpp

enum class Tag : std::uint8_t {
    None = 0,
    Bool,
    Int,     // native int48 fast path (fits in 48 bits; bignum promotion otherwise)
    Float,   // native double (stored as IEEE 754, zero overhead)
    Obj,     // boxed PyObject* (str/list/dict/tuple/bignum/func/type/instance...)
};

/// NaN-boxed Value — 8 bytes total, fits in a single GPR.
///
/// The tag is encoded in the top 16 bits (the NaN space of IEEE 754 doubles).
/// The payload is in the bottom 48 bits. Doubles are stored as-is.
struct Value {
private:
    std::uint64_t raw_{0};

    // Tag constants (stored in bits 63-48)
    static constexpr std::uint64_t kTagMask   = 0xFFFF000000000000ULL;
    static constexpr std::uint64_t kPayloadMask = 0x0000FFFFFFFFFFFFULL;
    static constexpr std::uint64_t kTagNone   = 0xFFF0000000000000ULL;
    static constexpr std::uint64_t kTagBool   = 0xFFF1000000000000ULL;
    static constexpr std::uint64_t kTagInt    = 0xFFF2000000000000ULL;
    static constexpr std::uint64_t kTagObj    = 0xFFF3000000000000ULL;

    // Canonical NaN (quiet NaN, bit 63 = 0 — doesn't conflict with tags)
    static constexpr std::uint64_t kCanonicalNaN = 0x7FF8000000000000ULL;

    // Check if raw looks like a double (top 16 bits < 0xFFF0)
    [[nodiscard]] static constexpr bool is_double_raw(std::uint64_t r) noexcept {
        return (r & kTagMask) < kTagNone;
    }

public:
    // --- Default constructor: None ---
    constexpr Value() noexcept : raw_(kTagNone) {}

    // --- Construct from raw bits (for internal use) ---
    static constexpr Value from_raw(std::uint64_t r) noexcept {
        Value v; v.raw_ = r; return v;
    }

    // --- Factory methods ---
    [[nodiscard]] static constexpr Value none() noexcept {
        return from_raw(kTagNone);
    }
    [[nodiscard]] static constexpr Value boolean(bool b) noexcept {
        return from_raw(kTagBool | (b ? 1 : 0));
    }
    [[nodiscard]] static Value integer(std::int64_t x) noexcept {
        // Sign-extend to 48 bits, mask to 48 bits.
        // Values that don't fit in 48 bits should be promoted to bignum
        // before reaching here (the runtime's values_add checks overflow).
        std::uint64_t payload = static_cast<std::uint64_t>(x) & kPayloadMask;
        return from_raw(kTagInt | payload);
    }
    [[nodiscard]] static Value real(double x) noexcept {
        std::uint64_t bits;
        std::memcpy(&bits, &x, sizeof(bits));
        // Canonicalize NaN — map any NaN to our canonical pattern so
        // it doesn't collide with our tag space.
        if ((bits & 0x7FF0000000000000ULL) == 0x7FF0000000000000ULL &&
            (bits & 0x000FFFFFFFFFFFFFULL) != 0) {
            bits = kCanonicalNaN;
        }
        return from_raw(bits);
    }
    [[nodiscard]] static Value object(PyObj* o) noexcept {
        return from_raw(kTagObj | (reinterpret_cast<std::uint64_t>(o) & kPayloadMask));
    }

    // --- Tag access ---
    [[nodiscard]] constexpr Tag tag() const noexcept {
        if (is_double_raw(raw_)) return Tag::Float;
        std::uint64_t t = raw_ & kTagMask;
        if (t == kTagNone) return Tag::None;
        if (t == kTagBool) return Tag::Bool;
        if (t == kTagInt)  return Tag::Int;
        if (t == kTagObj)  return Tag::Obj;
        return Tag::None;  // shouldn't happen
    }

    // --- Fast tag predicates (single AND + compare, no branch chain) ---
    // These are the hot-path checks: `a.is_int()` replaces `a.tag() == Tag::Int`
    // with a single mask+compare instead of the 5-branch tag() decode.
    [[nodiscard]] constexpr bool is_int() const noexcept {
        return (raw_ & kTagMask) == kTagInt;
    }
    [[nodiscard]] constexpr bool is_float() const noexcept {
        return is_double_raw(raw_);
    }
    [[nodiscard]] constexpr bool is_obj() const noexcept {
        return (raw_ & kTagMask) == kTagObj;
    }
    [[nodiscard]] constexpr bool is_none() const noexcept {
        return raw_ == kTagNone;
    }

    // --- Payload accessors ---
    /// Get int64 payload. Sign-extends from 48 bits.
    /// For values known to be int (checked via is_int()), the sign-extension
    /// is needed because Value::integer() only keeps the low 48 bits.
    [[nodiscard]] std::int64_t as_i() const noexcept {
        // Sign-extend from 48 bits: if bit 47 is set, fill bits 63-48 with 1s.
        std::uint64_t payload = raw_ & kPayloadMask;
        if (payload & (1ULL << 47)) {
            payload |= kTagMask;  // set high bits
        }
        return static_cast<std::int64_t>(payload);
    }

    /// Fast int payload: no sign-extension (for values that fit in 47 bits,
    /// which covers all loop counters and most arithmetic). Use this in hot
    /// paths where the caller knows the value is a small int.
    [[nodiscard]] constexpr std::int64_t as_i_fast() const noexcept {
        return static_cast<std::int64_t>(raw_ & kPayloadMask);
    }

    /// Get double payload.
    [[nodiscard]] double as_f() const noexcept {
        double result;
        std::memcpy(&result, &raw_, sizeof(result));
        return result;
    }

    /// Get PyObject* payload.
    [[nodiscard]] PyObj* as_obj() const noexcept {
        return reinterpret_cast<PyObj*>(raw_ & kPayloadMask);
    }

    /// Get the raw 64-bit word (for JIT code that needs the bits directly).
    [[nodiscard]] constexpr std::uint64_t raw() const noexcept { return raw_; }

    // --- Tag predicates (backward-compatible with old .is(Tag) API) ---
    [[nodiscard]] constexpr bool is(Tag t) const noexcept {
        return tag() == t;
    }

    /// Fast truthiness for tags where it is statically decidable; object
    /// truthiness is delegated to the runtime (may call __bool__).
    [[nodiscard]] bool truthy_hint() const noexcept {
        switch (tag()) {
            case Tag::None: return false;
            case Tag::Bool: return as_i() != 0;
            case Tag::Int: return as_i() != 0;
            case Tag::Float: return as_f() != 0.0;
            case Tag::Obj: return true;   // conservative; runtime decides
        }
        return true;
    }

    // --- Equality (for structural comparison in IR nodes) ---
    [[nodiscard]] bool structurally_equal(const Value& o) const noexcept {
        return raw_ == o.raw_;
    }
};

static_assert(sizeof(Value) == 8, "Value must be 8 bytes (NaN-boxed, single GPR)");
static_assert(std::is_trivially_copyable_v<Value>, "Value must be trivially copyable");
static_assert(std::is_standard_layout_v<Value>, "Value must be standard layout");

}  // namespace abi_v1
}  // namespace vortex
