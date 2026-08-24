// =============================================================================
// vortex/ir/ty.hpp — Static type lattice & shape model
//
// Purpose:
//   The type inference lattice feeding unboxing (Pass 47), devirtualization
//   (Passes 16/20/23), shape specialization (Passes 15/40a-c), and guard
//   emission. Types are interned in a TyTable; nodes carry a TyIndex.
//
// Lattice sketch (PyType models Python's dynamic universe):
//   Top ─ any value at all
//   ├─ Obj ─ some boxed PyObject
//   │   ├─ Str / List(elem) / Tuple / Dict(shape) / Func(sig) / Gen / Instance(cls,shape)
//   │   └─ Bignum (int that escaped int64)
//   ├─ Int (native int64, fits — unboxed)
//   ├─ Float (native double — unboxed)
//   ├─ Bool
//   └─ None
//   Bottom ─ unreachable
//
// Invariants:
//   - TyIndex 0 == Bottom, 1 == Top (fixed by TyTable constructor).
//   - join() is monotone & commutative; widening saturates at Top after
//     cfg::type_widen_rounds loop iterations (guarantees termination).
// =============================================================================

#pragma once

#include <cstdint>
#include <cstring>

#include "vortex/stdx/flat_map.hpp"
#include "vortex/stdx/small_vector.hpp"

namespace vortex::ir {

inline namespace abi_v1 {

using TyIndex = std::uint32_t;
inline constexpr TyIndex ty_bottom = 0;
inline constexpr TyIndex ty_top = 1;

enum class TyKind : std::uint8_t {
    Bottom = 0,
    Top,
    NoneTy,
    BoolTy,
    IntTy,        // unboxed int64
    FloatTy,      // unboxed double
    BignumTy,     // boxed arbitrary-precision int
    StrTy,
    ListTy,       // element type in elem
    TupleTy,      // element types in elems
    DictTy,       // shape in shape_id (key set), value type in elem
    FuncTy,       // function object; class_id = function id when known
    GenTy,        // generator
    IterTy,       // iterator
    ClassTy,      // the class object itself
    InstanceTy,   // instance of class_id with shape_id
    ObjTy,        // unknown boxed object
};

struct Ty {
    TyKind kind{TyKind::Top};
    std::uint8_t flags{0};
    std::uint16_t pad{};
    std::uint32_t class_id{0xFFFF'FFFF};   // InstanceTy/FuncTy/ClassTy
    std::uint32_t shape_id{0xFFFF'FFFF};   // InstanceTy/DictTy layout
    TyIndex elem{ty_top};                  // ListTy/DictTy value type
    stdx::small_vector<TyIndex, 4> elems{};  // TupleTy

    /// Structural equality — manual because small_vector has no operator==.
    [[nodiscard]] bool operator==(const Ty& o) const noexcept {
        if (kind != o.kind || class_id != o.class_id || shape_id != o.shape_id || elem != o.elem) {
            return false;
        }
        if (elems.size() != o.elems.size()) return false;
        return std::memcmp(elems.data(), o.elems.data(), elems.size() * sizeof(TyIndex)) == 0;
    }

    /// Unboxed-native predicate: value can live in a raw machine register.
    [[nodiscard]] bool is_unboxed() const noexcept {
        return kind == TyKind::IntTy || kind == TyKind::FloatTy || kind == TyKind::BoolTy;
    }
};

class TyTable {
public:
    TyTable() {
        // Canonical indices 0 and 1 (see header invariants).
        types_.push_back(Ty{.kind = TyKind::Bottom});
        types_.push_back(Ty{.kind = TyKind::Top});
    }

    [[nodiscard]] TyIndex intern(Ty t) {
        for (std::uint32_t i = 0; i < types_.size(); ++i) {
            if (types_[i] == t) return i;
        }
        types_.push_back(t);
        return static_cast<TyIndex>(types_.size() - 1);
    }

    [[nodiscard]] const Ty& operator[](TyIndex i) const noexcept {
        VORTEX_ASSUME(i < types_.size());
        return types_[i];
    }

    [[nodiscard]] TyIndex join(TyIndex a, TyIndex b) noexcept {
        if (a == b) return a;
        if (a == ty_bottom) return b;
        if (b == ty_bottom) return a;
        if (a == ty_top || b == ty_top) return ty_top;
        const Ty& ta = types_[a];
        const Ty& tb = types_[b];
        if (ta.kind == TyKind::IntTy && tb.kind == TyKind::BignumTy) return b;  // int -> bignum
        if (tb.kind == TyKind::IntTy && ta.kind == TyKind::BignumTy) return a;
        if (ta.kind == TyKind::IntTy && tb.kind == TyKind::FloatTy) return b;   // int -> float
        if (tb.kind == TyKind::IntTy && ta.kind == TyKind::FloatTy) return a;
        if (ta.kind != tb.kind) return ty_top;
        // Same kind: join payloads conservatively.
        Ty joined = ta;
        if (joined.class_id != tb.class_id) joined.class_id = 0xFFFF'FFFF;
        if (joined.shape_id != tb.shape_id) joined.shape_id = 0xFFFF'FFFF;
        if (joined.kind == TyKind::ListTy) joined.elem = join(joined.elem, tb.elem);
        if (joined.kind == TyKind::InstanceTy && joined.class_id == 0xFFFF'FFFF) {
            joined.kind = TyKind::ObjTy;   // unknown instance
        }
        return static_cast<TyIndex>(find_or_append(joined));
    }

    /// Numeric tower: can a value of `a` flow where `b` is expected without a
    /// runtime type error? (conservative; guards refine at runtime)
    [[nodiscard]] bool compatible(TyIndex a, TyIndex b) const noexcept {
        if (a == ty_top || b == ty_top || a == ty_bottom || b == ty_bottom) return true;
        const Ty& ta = types_[a];
        const Ty& tb = types_[b];
        if (ta.kind == tb.kind) return true;
        if (tb.kind == TyKind::ObjTy) return true;
        if ((ta.kind == TyKind::IntTy || ta.kind == TyKind::BignumTy) &&
            (tb.kind == TyKind::IntTy || tb.kind == TyKind::BignumTy)) {
            return true;
        }
        if ((ta.kind == TyKind::IntTy || ta.kind == TyKind::FloatTy ||
             ta.kind == TyKind::BignumTy) &&
            tb.kind == TyKind::FloatTy) {
            return true;
        }
        return false;
    }

    [[nodiscard]] std::uint32_t size() const noexcept { return types_.size(); }

    // Frequently used pre-interned indices (lazily stable within a table).
    [[nodiscard]] TyIndex int_ty() { return intern(Ty{.kind = TyKind::IntTy}); }
    [[nodiscard]] TyIndex float_ty() { return intern(Ty{.kind = TyKind::FloatTy}); }
    [[nodiscard]] TyIndex bool_ty() { return intern(Ty{.kind = TyKind::BoolTy}); }
    [[nodiscard]] TyIndex none_ty() { return intern(Ty{.kind = TyKind::NoneTy}); }
    [[nodiscard]] TyIndex str_ty() { return intern(Ty{.kind = TyKind::StrTy}); }
    [[nodiscard]] TyIndex list_ty(TyIndex elem = ty_top) {
        return intern(Ty{.kind = TyKind::ListTy, .elem = elem});
    }
    [[nodiscard]] TyIndex obj_ty() { return intern(Ty{.kind = TyKind::ObjTy}); }

private:
    [[nodiscard]] std::uint32_t find_or_append(const Ty& t) {
        for (std::uint32_t i = 0; i < types_.size(); ++i) {
            if (types_[i] == t) return i;
        }
        types_.push_back(t);
        return static_cast<std::uint32_t>(types_.size() - 1);
    }

    stdx::small_vector<Ty, 64> types_;
};

}  // namespace abi_v1
}  // namespace vortex::ir
