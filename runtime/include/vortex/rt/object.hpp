// =============================================================================
// vortex/rt/object.hpp — PyObject model (runtime)
//
// Purpose:
//   The heap object taxonomy for the VORTEX Python subset. Every boxed value
//   is a PyObj* with a tagged header; small ints / doubles / bool / none live
//   unboxed in the 16-byte Value register form (common/value.hpp) and are
//   boxed only when stored in containers or passed to generic code.
//
// Memory discipline (manual, CPython-style):
//   - refcount starts at 1 on allocation; incref on store-into-container,
//     register-write of an owned ref, and return; decref on overwrite and
//     teardown. No cycle collector in the subset (documented: programs in
//     scope are short-lived; see docs/adr/0005-memory-model.md).
//   - Allocation goes through the global heap with TLAB fast paths
//     (Rule 12); the interpreter's temporaries are stack Values.
//
// Shapes (Part I of the object-specialization spec):
//   PyInstance carries a ShapeId into a global shape tree; each shape node
//   adds one attribute at a fixed slot offset. Attribute access consults
//   the shape's cached flat map (attr symbol -> slot) — O(1) after the
//   first hit, and the JIT guards on shape_id to devirtualize (Pass 40b).
// =============================================================================

#pragma once

#include <cstdint>
#include <cstdio>
#include <string_view>

#include "vortex/common/value.hpp"
#include "vortex/stdx/small_vector.hpp"

namespace vortex::rt {

inline namespace abi_v1 {

using vortex::Value;
using vortex::Tag;

// --- Header -------------------------------------------------------------------
enum class ObjTag : std::uint8_t {
    Long = 1,       // int (int64 fast path; bignum payload when big)
    Float,
    Str,
    Bool,
    None,
    List,
    Tuple,
    Dict,
    Type,           // class object
    Instance,       // class instance (shape-based slots)
    Function,       // Python-level function (code unit + env)
    NativeFn,       // C++ builtin
    Cell,           // closure cell
    Generator,
    RangeIter,
    ListIter,
    StrIter,
    DictIter,
    GenIter,
    ExcIterSentinel,
    Module,         // native module namespace (math, ...)
    IterSentinel,
    BoundMethod,    // func + receiver pair
};

struct PyLongObj; struct PyFloatObj; struct PyStrObj; struct PyListObj;
struct PyTupleObj; struct PyDictObj; struct PyTypeObj; struct PyInstanceObj;
struct PyFuncObj; struct PyNativeFnObj; struct PyCellObj; struct PyGeneratorObj;
struct PyModuleObj; struct PyRangeIterObj;

}  // namespace abi_v1
}  // namespace vortex::rt

// The concrete definition of the forward-declared vortex::PyObj from
// common/value.hpp: every boxed Value points at one of these headers.
// (Defined in the SAME inline abi_v1 namespace as the forward declaration.)
namespace vortex {
inline namespace abi_v1 {
struct PyObj {
    rt::ObjTag tag{};
    std::uint8_t flags{};
    std::uint16_t pad{};
    std::uint32_t refcount{1};
};
}  // namespace abi_v1
}  // namespace vortex

namespace vortex::rt {
inline namespace abi_v1 {

// --- Integers ------------------------------------------------------------------
/// Arbitrary-precision fallback (base 2^32 limbs, little-endian, sign separate).
struct BigNum {
    bool negative{false};
    stdx::small_vector<std::uint32_t, 4> limbs{};   // little-endian, no leading zeros

    [[nodiscard]] bool is_zero() const noexcept { return limbs.empty(); }
    [[nodiscard]] static BigNum from_i64(std::int64_t v) noexcept;
    [[nodiscard]] std::int64_t try_i64(bool& fits) const noexcept;
    [[nodiscard]] int cmp(const BigNum& o) const noexcept;
    [[nodiscard]] static BigNum add(const BigNum& a, const BigNum& b) noexcept;
    [[nodiscard]] static BigNum sub(const BigNum& a, const BigNum& b) noexcept;
    [[nodiscard]] static BigNum mul(const BigNum& a, const BigNum& b) noexcept;
    [[nodiscard]] static BigNum divmod(const BigNum& a, const BigNum& b, BigNum& rem,
                                       bool& div_by_zero) noexcept;
    [[nodiscard]] static BigNum pow_small(const BigNum& base, std::uint64_t exp) noexcept;
    [[nodiscard]] std::size_t hash() const noexcept;
    void to_string(stdx::small_vector<char, 128>& out) const noexcept;
};

struct PyLongObj : PyObj {
    std::int64_t value{0};
    BigNum big{};     // valid only when flags & kBigFlag
    static constexpr std::uint8_t kBigFlag = 1;
};

struct PyFloatObj : PyObj {
    double value{0};
};

struct PyBoolObj : PyObj {
    bool value{false};
};

struct PyNoneObj : PyObj {};

// --- Strings --------------------------------------------------------------------
struct PyStrObj : PyObj {
    std::uint32_t length{0};
    std::uint32_t hash_cache{0};
    bool hashed{false};
    // Flexible array member: char bytes[length+1].
    [[nodiscard]] char* data() noexcept {
        return reinterpret_cast<char*>(this + 1);
    }
    [[nodiscard]] const char* data() const noexcept {
        return reinterpret_cast<const char*>(this + 1);
    }
    [[nodiscard]] std::string_view view() const noexcept {
        return std::string_view(data(), length);
    }
};

// --- Sequences -------------------------------------------------------------------
struct PyListObj : PyObj {
    std::uint32_t length{0};
    std::uint32_t capacity{0};
    Value* items{nullptr};   // owned refs
};

struct PyTupleObj : PyObj {
    std::uint32_t length{0};
    Value* items{nullptr};   // owned refs
};

// --- Dict (open addressing, Rule 17 cache discipline) ------------------------------
struct DictEntry {
    Value key{};      // owned; tag None = empty
    Value value{};    // owned
    std::uint32_t hash{0};
    std::uint32_t seq{0};   // insertion sequence number (iteration order)
    bool used{false}; // tombstone marker
};

struct PyDictObj : PyObj {
    std::uint32_t count{0};
    std::uint32_t capacity{8};   // power of two
    DictEntry* entries{nullptr};

    // Iteration order: insertion order (CPython 3.7+ guarantee) maintained by
    // scanning slots in insertion sequence — entries store seq numbers.
    std::uint32_t insert_seq{0};
};

// --- Shapes & instances ---------------------------------------------------------------
struct ShapeNode {
    ShapeNode* parent{nullptr};
    std::uint32_t symbol{0xFFFF'FFFF};   // attribute added at this depth
    std::uint32_t slot{0};               // absolute slot offset
    std::uint32_t id{0};                 // global ShapeId
    std::uint32_t depth{0};
    // cache: symbol -> slot for THIS shape (flat, built lazily on first miss)
    stdx::small_vector<std::pair<std::uint32_t, std::uint32_t>, 8> cache{};
};

struct PyTypeObj : PyObj {
    std::uint32_t name_symbol{0xFFFF'FFFF};
    PyTypeObj* base{nullptr};          // single inheritance (borrowed)
    PyDictObj* dict{nullptr};          // class namespace (methods; owned)
    ShapeNode* root_shape{nullptr};    // shape tree root for instances
    std::uint32_t type_id{0};
};

struct PyInstanceObj : PyObj {
    PyTypeObj* type{nullptr};          // borrowed
    ShapeNode* shape{nullptr};         // borrowed (global shape tree)
    std::uint32_t slot_capacity{0};
    Value* slots{nullptr};             // owned refs
};

// --- Functions & closures ------------------------------------------------------------
struct CodeUnit;   // bytecode container (vortex/rt/code.hpp)

struct PyFuncObj : PyObj {
    std::uint32_t code_unit_id{0};
    std::uint32_t name_symbol{0xFFFF'FFFF};
    PyTupleObj* defaults{nullptr};     // owned (may be null)
    PyTupleObj* cells{nullptr};        // owned closure tuple (may be null)
    std::uint32_t n_positional{0};
    bool has_varargs{false};
    bool has_kwargs{false};
};

using NativeFnPtr = Value (*)(void* user, Value* args, std::uint32_t argc);

struct PyNativeFnObj : PyObj {
    NativeFnPtr fn{nullptr};
    void* user{nullptr};
    std::uint32_t name_symbol{0xFFFF'FFFF};
};

struct PyCellObj : PyObj {
    Value value{};   // owned; tag None + flags&kUnbound = unbound
    static constexpr std::uint8_t kUnbound = 1;
};

struct PyModuleObj : PyObj {
    std::uint32_t name_symbol{0xFFFF'FFFF};
    PyDictObj* ns{nullptr};   // owned
};

// --- Iterators & generators ------------------------------------------------------------
struct PyRangeIterObj : PyObj {
    std::int64_t current{0};
    std::int64_t stop{0};
    std::int64_t step{1};
};

struct PyListIterObj : PyObj {
    PyListObj* list{nullptr};   // borrowed
    std::uint32_t index{0};
};

struct PyStrIterObj : PyObj {
    PyStrObj* str{nullptr};     // borrowed
    std::uint32_t index{0};
};

struct PyDictIterObj : PyObj {
    PyDictObj* dict{nullptr};   // borrowed
    std::uint32_t slot{0};
    std::uint32_t remaining{0};
};

struct Frame;

struct PyGeneratorObj : PyObj {
    Frame* frame{nullptr};      // suspended frame (owned)
    std::uint32_t code_unit_id{0};
    bool exhausted{false};
};

// =============================================================================
// Global heap & object services
// =============================================================================

/// Global runtime context: singletons, shape table, interned strings, types.
class Runtime {
public:
    static Runtime& instance() noexcept;

    // --- singletons (never freed) ------------------------------------------------
    PyNoneObj* none{};
    PyBoolObj* true_obj{};
    PyBoolObj* false_obj{};
    PyTypeObj* type_int{};       // builtin exception & base types
    PyTypeObj* type_exc_base{};
    PyTypeObj* type_value_error{};
    PyTypeObj* type_type_error{};
    PyTypeObj* type_zero_div{};
    PyTypeObj* type_index_error{};
    PyTypeObj* type_key_error{};
    PyTypeObj* type_stop_iter{};
    PyTypeObj* type_runtime_error{};
    PyTypeObj* type_assertion_error{};
    PyTypeObj* type_attribute_error{};
    PyTypeObj* type_memory_error{};
    PyTypeObj* type_not_implemented_error{};
    PyTypeObj* type_name_error{};

    // --- allocation -----------------------------------------------------------------
    [[nodiscard]] PyStrObj* new_str(std::string_view text) noexcept;
    [[nodiscard]] PyLongObj* new_long_i64(std::int64_t v) noexcept;
    [[nodiscard]] PyLongObj* new_long_big(BigNum b) noexcept;
    [[nodiscard]] PyFloatObj* new_float(double v) noexcept;
    [[nodiscard]] PyListObj* new_list(std::uint32_t cap = 4) noexcept;
    [[nodiscard]] PyTupleObj* new_tuple(std::uint32_t n) noexcept;
    [[nodiscard]] PyDictObj* new_dict() noexcept;
    [[nodiscard]] PyTypeObj* new_type(std::uint32_t name_symbol, PyTypeObj* base,
                                      PyDictObj* dict) noexcept;
    [[nodiscard]] PyInstanceObj* new_instance(PyTypeObj* type) noexcept;
    [[nodiscard]] PyFuncObj* new_func(std::uint32_t code_unit_id, std::uint32_t name_symbol,
                                      PyTupleObj* defaults, PyTupleObj* cells,
                                      std::uint32_t n_positional, bool varargs,
                                      bool kwargs) noexcept;
    [[nodiscard]] PyNativeFnObj* new_native(std::uint32_t name_symbol, NativeFnPtr fn,
                                            void* user) noexcept;
    [[nodiscard]] PyCellObj* new_cell(Value v) noexcept;
    [[nodiscard]] PyModuleObj* new_module(std::uint32_t name_symbol) noexcept;
    [[nodiscard]] PyRangeIterObj* new_range_iter(std::int64_t start, std::int64_t stop,
                                                 std::int64_t step) noexcept;

    /// Exception instance: Exception(args...) — message joined like Python.
    [[nodiscard]] PyInstanceObj* new_exception(PyTypeObj* type, Value* args,
                                               std::uint32_t argc) noexcept;

    // --- shapes ------------------------------------------------------------------------
    [[nodiscard]] ShapeNode* shape_add_attr(ShapeNode* base, std::uint32_t symbol) noexcept;
    [[nodiscard]] bool shape_find(ShapeNode* shape, std::uint32_t symbol,
                                  std::uint32_t& slot_out) noexcept;
    /// Watchdog registry (Rule 42): shape tree version — bumps whenever any
    /// shape transitions; JIT code guards revalidate against it.
    std::uint32_t shape_epoch{0};

    // --- refcounting --------------------------------------------------------------------
    void incref(PyObj* o) noexcept {
        if (o) ++o->refcount;
    }
    void decref(PyObj* o) noexcept;

    // --- type metadata -------------------------------------------------------------------
    [[nodiscard]] PyTypeObj* type_of(const Value& v) noexcept;
    [[nodiscard]] std::uint32_t type_id_of(const Value& v) noexcept;

    /// Exceptions are instances of user classes OR builtin exception types —
    /// both are PyInstanceObj whose type chain is checked for a match.
    [[nodiscard]] bool exception_matches(PyInstanceObj* exc, std::uint32_t type_name_symbol) noexcept;

    // --- universal operations ------------------------------------------------------------
    [[nodiscard]] bool truthy(const Value& v) noexcept;
    [[nodiscard]] bool eq(const Value& a, const Value& b) noexcept;
    [[nodiscard]] std::uint32_t hash(const Value& v) noexcept;
    void repr_into(const Value& v, stdx::small_vector<char, 128>& out) noexcept;
    void str_into(const Value& v, stdx::small_vector<char, 128>& out) noexcept;

    std::uint64_t allocations{0};
    std::uint64_t bytes_allocated{0};

private:
    Runtime() noexcept;
    ~Runtime() = default;
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    stdx::small_vector<ShapeNode*, 64> shape_nodes_{};
    std::uint32_t next_shape_id_{1};
    std::uint32_t next_type_id_{1};
};

// --- free helpers (hot path, inline) ---------------------------------------------
[[nodiscard]] inline PyObj* as_obj(const Value& v) noexcept { return v.tag() == Tag::Obj ? v.as_obj() : nullptr; }

template <typename T>
[[nodiscard]] inline T* obj_cast(PyObj* o) noexcept {
    return o ? static_cast<T*>(o) : nullptr;
}

[[nodiscard]] inline Value obj_value(PyObj* o) noexcept { return Value::object(o); }

/// Box an unboxed Value into a heap object (identity for Tag::Obj).
[[nodiscard]] Value box(const Value& v) noexcept;

/// Box into a NEW owned reference (caller owns).
[[nodiscard]] Value box_owned(const Value& v) noexcept;

// --- container ops (owned-ref discipline documented per signature) -----------------
[[nodiscard]] bool list_push(PyListObj* l, Value v) noexcept;           // takes ownership
[[nodiscard]] bool list_set(PyListObj* l, std::uint32_t i, Value v) noexcept;  // takes ownership
[[nodiscard]] Value list_get(PyListObj* l, std::uint32_t i) noexcept;   // borrowed
[[nodiscard]] bool dict_set(PyDictObj* d, Value key, Value value) noexcept;    // takes ownership of both
[[nodiscard]] bool dict_get(PyDictObj* d, const Value& key, Value& out) noexcept;  // borrowed out
[[nodiscard]] bool dict_del(PyDictObj* d, const Value& key) noexcept;   // frees key/value refs

// --- bignum numeric core (used by both interpreter and JIT slow paths) -----------
[[nodiscard]] bool values_add(const Value& a, const Value& b, Value& out) noexcept;
[[nodiscard]] bool values_sub(const Value& a, const Value& b, Value& out) noexcept;
[[nodiscard]] bool values_mul(const Value& a, const Value& b, Value& out) noexcept;
[[nodiscard]] bool values_truediv(const Value& a, const Value& b, Value& out) noexcept;
[[nodiscard]] bool values_floordiv(const Value& a, const Value& b, Value& out) noexcept;
[[nodiscard]] bool values_mod(const Value& a, const Value& b, Value& out) noexcept;
[[nodiscard]] bool values_pow(const Value& a, const Value& b, Value& out) noexcept;
[[nodiscard]] bool values_bitop(const Value& a, const Value& b, std::uint16_t op, Value& out) noexcept;
[[nodiscard]] bool values_shift(const Value& a, const Value& b, bool left, Value& out) noexcept;
[[nodiscard]] bool values_neg(const Value& a, Value& out) noexcept;
[[nodiscard]] bool values_compare(const Value& a, const Value& b, std::uint16_t op, bool& out) noexcept;

// Numeric tower helpers: promote to double when mixed (Python semantics).
[[nodiscard]] bool as_f64(const Value& v, double& out) noexcept;
[[nodiscard]] bool as_i64(const Value& v, std::int64_t& out) noexcept;

}  // namespace abi_v1
}  // namespace vortex::rt
