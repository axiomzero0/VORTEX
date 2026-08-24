// =============================================================================
// vortex/frontend/lowering.hpp — Pass 1: AST -> VORTEX IR (Rule 33)
//
// Purpose:
//   Lowers one function body (or the module toplevel) to a Sea-of-Nodes
//   graph. The initial graph is in the "Python value domain": data edges
//   carry Python semantic values (fast-path unboxed int/float or boxed
//   PyObject). NO implicit conversions exist anywhere in the IR — every
//   domain crossing inserted later is an explicit Unbox/Box/SExt/... node
//   (Rule 33). Literals lower to ConstInt/ConstFloat/ConstPy.
//
// Control model:
//   Start -> Region/If/Loop control projections; a lowering cursor holds the
//   current control + effect-chain tail. Phis are inserted at merges for
//   every variable whose SSA value differs between predecessors. Loops
//   pre-create LoopPhis (incl. EffectPhi) patched with backedges after the
//   body lowers — classic structured SSA construction.
//
// Closures (Pass 23 devirtualizes these later):
//   Child functions capture enclosing locals through cell objects. The child
//   receives a hidden trailing parameter `__cells__` (tuple of PyCell).
//   Free-variable reads/writes lower to CallNative(cell_get/cell_set).
//
// Generators:
//   Any `yield` marks the unit a generator; Yield nodes suspend at Tier 0.
//   Pass 24 deforests fully-consumed generator loops at higher tiers.
// =============================================================================

#pragma once

#include "vortex/frontend/ast.hpp"
#include "vortex/ir/graph.hpp"

namespace vortex::fe {

inline namespace abi_v1 {

using vortex::ir::Graph;

/// A function awaiting compilation (children discovered while lowering).
struct PendingFunction {
    SymbolId name{0xFFFF'FFFF};
    Stmt* def{nullptr};
    std::uint32_t code_unit_hint{0};
};

/// Native runtime helper ids used by CallNative nodes (resolved by the
/// runtime linker; kept in one enum so the IR stays symbol-free — Rule 16).
enum class NativeHelper : std::uint16_t {
    MakeFunction = 1,   // (code_unit_id, defaults_tuple, cells_tuple) -> PyFunc
    MakeCell,           // (initial_value) -> PyCell
    CellGet,            // (PyCell) -> value (raises UnboundLocalError if empty)
    CellSet,            // (PyCell, value) -> None
    ImportModule,       // (module_name_str) -> module object
    MakeClass,          // (name_str, dict, base_or_none) -> PyType
    UnpackSequence,     // (value, n) -> PyTuple of exactly n
    PrintValue,         // runtime print(...): variadic via args tuple
    ListToTuple,        // internal conversions
    UnboundCheck,       // (value) -> raises if unbound sentinel
    DelSubscript,       // (obj, key) -> None
    DelAttr,            // (obj, name_sym) -> None
    TypeOf,             // (value) -> type object (used by isinstance paths)
    GetCurrentException,// () -> active exception value (handler entry)
    IsInstance,         // (value, type_name_str) -> bool (class-name match)
    FormatValue,        // (value) -> str(value) for print
    NextIterator,       // (iterator) -> value or raises StopIteration
    RangeNew,           // fast range object constructor (used by for-range opt)
    HelperCount
};

/// Result of lowering one unit.
struct LoweredUnit {
    Graph graph{};
    SymbolId name{0xFFFF'FFFF};
    stdx::small_vector<SymbolId, 8> param_names{};
    stdx::small_vector<Expr*, 4> default_exprs{};   // lowered to a defaults tuple at def-site
    bool has_varargs{false};
    bool has_kwargs{false};
    bool is_generator{false};
    stdx::small_vector<PendingFunction, 8> children{};
    /// Free variables captured from the enclosing scope (child side view).
    stdx::small_vector<SymbolId, 8> captures{};
    std::uint32_t code_unit_id{0};
};

/// Compilation-wide context shared across units (id assignment).
struct LowerContext {
    std::uint32_t next_code_unit_id{1};   // 0 reserved for the module toplevel
    std::uint32_t units_lowered{0};
};

/// Lower a function definition (or module toplevel when `def` is nullptr).
/// `module` provides the AST + string pool; `captured_names` are the free
/// variables resolved through cells (empty for non-closures).
/// `class_body` marks namespace-dict units (class bodies return their dict).
[[nodiscard]] Result<LoweredUnit> lower_unit(Module& module, LowerContext& ctx, Stmt* def,
                                             SymbolId name,
                                             stdx::small_vector<SymbolId, 8>& captured_names,
                                             bool class_body = false) noexcept;

/// Compute the set of names assigned or bound in a function body (used by
/// capture analysis: a child name neither assigned locally nor declared
/// global/nonlocal is a capture candidate).
[[nodiscard]] stdx::small_vector<SymbolId, 16> bound_names(const StmtList& body) noexcept;

/// Compute free variables referenced but not bound in a body (transitive:
/// nested defs propagate their free names upward through this function).
[[nodiscard]] stdx::small_vector<SymbolId, 16> free_names(const StmtList& body) noexcept;

}  // namespace abi_v1
}  // namespace vortex::fe
