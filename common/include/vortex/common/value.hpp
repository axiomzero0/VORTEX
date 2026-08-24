// =============================================================================
// vortex/common/value.hpp — Tagged runtime value (shared compiler + runtime)
//
// Purpose:
//   One 16-byte trivially-copyable value representation used by:
//     - IR constants (ConstInt/ConstFloat/ConstPy),
//     - Tier-0 register file slots,
//     - Deoptimization state reconstruction (Rule 4),
//     - Golden-test expected values.
//
// Invariants:
//   - Trivially copyable & standard-layout: memcpy-safe, vectorizable.
//   - Int is int64_t (Python small-int fast path; bignum promotion happens
//     only at object boundaries, tagged Tag::Obj with PyObj*).
//   - Float is IEEE binary64.
//   - Obj is an unowned PyObject pointer: reference ownership is managed by
//     the runtime's refcount discipline, never by Value itself.
// =============================================================================

#pragma once

#include <cstdint>
#include <type_traits>

namespace vortex {

inline namespace abi_v1 {

struct PyObj;   // runtime object header — full definition in vortex/rt/object.hpp

enum class Tag : std::uint8_t {
    None = 0,
    Bool,
    Int,     // native int64 fast path (fits, non-bignum)
    Float,   // native double
    Obj,     // boxed PyObject* (str/list/dict/tuple/bignum/func/type/instance...)
};

struct Value {
    Tag tag{Tag::None};
    std::uint8_t pad_[7]{};
    union {
        std::int64_t i;
        double f;
        PyObj* obj;
    } as{0};

    [[nodiscard]] static constexpr Value none() noexcept {
        Value v; v.tag = Tag::None; v.as.i = 0; return v;
    }
    [[nodiscard]] static constexpr Value boolean(bool b) noexcept {
        Value v; v.tag = Tag::Bool; v.as.i = b ? 1 : 0; return v;
    }
    [[nodiscard]] static constexpr Value integer(std::int64_t x) noexcept {
        Value v; v.tag = Tag::Int; v.as.i = x; return v;
    }
    [[nodiscard]] static constexpr Value real(double x) noexcept {
        Value v; v.tag = Tag::Float; v.as.f = x; return v;
    }
    [[nodiscard]] static constexpr Value object(PyObj* o) noexcept {
        Value v; v.tag = Tag::Obj; v.as.obj = o; return v;
    }

    [[nodiscard]] constexpr bool is(Tag t) const noexcept { return tag == t; }
    [[nodiscard]] constexpr bool truthy_hint() const noexcept {
        // Fast truthiness for tags where it is statically decidable; object
        // truthiness is delegated to the runtime (may call __bool__).
        switch (tag) {
            case Tag::None: return false;
            case Tag::Bool: return as.i != 0;
            case Tag::Int: return as.i != 0;
            case Tag::Float: return as.f != 0.0;
            case Tag::Obj: return true;   // conservative; runtime decides
        }
        return true;
    }
};

static_assert(sizeof(Value) == 16, "Value must stay 16 bytes (SIMD-friendly, Rule 20)");
static_assert(std::is_trivially_copyable_v<Value>, "Value must be trivially copyable");

}  // namespace abi_v1
}  // namespace vortex
