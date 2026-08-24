// =============================================================================
// vortex/ir/node_kind.hpp — NodeKind taxonomy (Rules 8, 15, 58)
//
// Purpose:
//   Closed enum of all Sea-of-Nodes node kinds. Kind ranges are grouped so
//   "class of node" queries are single comparisons (cache-friendly, and the
//   switch dispatch used by passes is dense).
//
// Invariants:
//   - Values are dense and stable: serialized IR (golden tests, AOT caches)
//     depends on them; adding kinds is append-only (Rule 31 versioning).
//   - Every switch on NodeKind is exhaustive or ends in VORTEX_UNREACHABLE.
// =============================================================================

#pragma once

#include <cstdint>

namespace vortex::ir {

inline namespace abi_v1 {

enum class NodeKind : std::uint16_t {
    // ---- Control (0..31) ----------------------------------------------------
    Start = 0,        // function entry: produces initial control + memory + params
    Region,           // merge of N control predecessors
    If,               // branch on predicate
    IfTrue,           // true projection of If
    IfFalse,          // false projection of If
    Jump,             // unconditional control edge
    Loop,             // loop header (Region with cyclic backedge)
    LoopExit,         // leaving a loop
    Return,           // function return (control + data)
    Throw,            // raise exception to caller (control + exception value)
    Unreachable,      // statically dead control
    Catch,            // exception handler merge (try body raise edges land here)

    // ---- Data basics (32..63) ------------------------------------------------
    Parameter,        // function argument
    ConstInt,         // unboxed int64 constant
    ConstFloat,       // unboxed double constant
    ConstPy,          // boxed constant (string literal, code, etc.)
    Phi,              // SSA merge
    EffectPhi,        // memory/effect merge

    // ---- Native arithmetic (64..127) — operate on unboxed scalars -----------
    Add, Sub, Mul, Div, Mod, Pow, Neg,
    BitAnd, BitOr, BitXor, Shl, Shr,
    CmpLT, CmpLE, CmpGT, CmpGE, CmpEQ, CmpNE,
    Not,              // logical not on i1

    // ---- Explicit conversions (128..143) — Rule 33: no implicit coercions ----
    SExt, ZExt, Trunc, BitCast,
    I2F, F2I,
    Unbox,            // PyObjToNative: value + expected type -> native scalar
    Box,              // NativeToPyObj: native scalar -> PyObject

    // ---- Python-level operations (144..255) ----------------------------------
    PyBinary,         // + - * / // % ** << >> & | ^ @ with full dyn semantics
    PyUnary,          // - ~ not
    PyCompare,        // < <= > >= == != is is-not in not-in (chained lowered)
    CallPy,           // generic dynamic call
    CallDirect,       // statically resolved call (post devirtualization)
    GuardedDirectCall,// speculative direct call + type/shape guard
    CallNative,       // runtime helper call (never throws, returns Result)
    DispatchCache,    // inline-cache node (first-class IC, Part III of spec)
    LoadAttr,         // obj.name (dict/shape lookup)
    StoreAttr,        // obj.name = v
    LoadIndex,        // obj[i]
    StoreIndex,       // obj[i] = v
    LoadGlobal,       // module global (versioned IC)
    StoreGlobal,      // module global write (invalidates version)
    Len,              // len(x)
    Iter,             // iter(x)
    IterNext,         // next iterator (raises StopIteration at end)
    NewList, NewDict, NewTuple, NewObject,
    ListAppend,       // list.append in comprehension hot loop
    Yield,            // generator suspension point
    GetIterCheck,     // for-loop protocol fast path check

    // ---- Memory (256..287) ----------------------------------------------------
    Load,             // raw native load (base, index) — after specialization
    Store,            // raw native store
    LoadField,        // shape-guarded fixed-offset field read
    StoreField,       // shape-guarded fixed-offset field write
    Allocated,        // PEA materialization point (heap alloc moved here)
    RegionFree,       // region-based bulk free (Pass 42)
    Altered,          // speculatively reordered memory write (Rule 4 rollback unit)

    // ---- Speculation & guards (288..319) ---------------------------------------
    Guard,            // hardware guard; failure -> deopt via FrameState
    DeoptBarrier,     // safepoint where deopt may occur
    FrameStateNode,   // snapshot descriptor attachment point

    // ---- Vector / SLP (320..351) ------------------------------------------------
    VecPack,          // pack N scalars into a vector
    VecExtract,       // extract lane i
    VecLoad, VecStore,// contiguous vector memory access
    VecOp,            // elementwise vector arithmetic
    Gather, Scatter,  // gather/scatter for pointer arrays (Pass 31d)
    VecShuffle,       // lane permutation
    VecReduce,        // horizontal reduction

    // ---- Misc (352..) -----------------------------------------------------------
    TupleProj,        // extract element k of a tuple-producing node
    CallResult,       // multi-value projection of calls
    DebugBreak,       // debugger hook (cold)

    KindCount
};

// --- Class predicates ----------------------------------------------------------
// Switch-based (not numeric ranges): the enum is append-only, and a numeric
// range breaks silently when kinds are inserted (this exact bug was caught by
// tests/unit/ir_test.cpp — the verifier misclassified ConstInt as control).

enum class NodeClass : std::uint8_t {
    Control,     // Start Region If IfTrue IfFalse Jump Loop LoopExit Return Throw Unreachable Catch
    Data,        // Parameter ConstInt ConstFloat ConstPy Phi EffectPhi
    Arith,       // native scalar arithmetic
    Conversion,  // SExt ZExt Trunc BitCast I2F F2I Unbox Box (Rule 33)
    PythonOp,    // dynamic-semantics operations
    Memory,      // raw / field memory effects
    Speculation, // guards, deopt barriers, frame states
    Vector,      // SLP packet machinery
    Misc,
};

[[nodiscard]] constexpr NodeClass node_class(NodeKind k) noexcept {
    switch (k) {
        case NodeKind::Start: case NodeKind::Region: case NodeKind::If:
        case NodeKind::IfTrue: case NodeKind::IfFalse: case NodeKind::Jump:
        case NodeKind::Loop: case NodeKind::LoopExit: case NodeKind::Return:
        case NodeKind::Throw: case NodeKind::Unreachable: case NodeKind::Catch:
            return NodeClass::Control;
        case NodeKind::Parameter: case NodeKind::ConstInt: case NodeKind::ConstFloat:
        case NodeKind::ConstPy: case NodeKind::Phi: case NodeKind::EffectPhi:
            return NodeClass::Data;
        case NodeKind::Add: case NodeKind::Sub: case NodeKind::Mul: case NodeKind::Div:
        case NodeKind::Mod: case NodeKind::Pow: case NodeKind::Neg:
        case NodeKind::BitAnd: case NodeKind::BitOr: case NodeKind::BitXor:
        case NodeKind::Shl: case NodeKind::Shr:
        case NodeKind::CmpLT: case NodeKind::CmpLE: case NodeKind::CmpGT:
        case NodeKind::CmpGE: case NodeKind::CmpEQ: case NodeKind::CmpNE:
        case NodeKind::Not:
            return NodeClass::Arith;
        case NodeKind::SExt: case NodeKind::ZExt: case NodeKind::Trunc:
        case NodeKind::BitCast: case NodeKind::I2F: case NodeKind::F2I:
        case NodeKind::Unbox: case NodeKind::Box:
            return NodeClass::Conversion;
        case NodeKind::PyBinary: case NodeKind::PyUnary: case NodeKind::PyCompare:
        case NodeKind::CallPy: case NodeKind::CallDirect:
        case NodeKind::GuardedDirectCall: case NodeKind::CallNative:
        case NodeKind::DispatchCache: case NodeKind::LoadAttr: case NodeKind::StoreAttr:
        case NodeKind::LoadIndex: case NodeKind::StoreIndex:
        case NodeKind::LoadGlobal: case NodeKind::StoreGlobal:
        case NodeKind::Len: case NodeKind::Iter: case NodeKind::IterNext:
        case NodeKind::NewList: case NodeKind::NewDict: case NodeKind::NewTuple:
        case NodeKind::NewObject: case NodeKind::ListAppend: case NodeKind::Yield:
        case NodeKind::GetIterCheck:
            return NodeClass::PythonOp;
        case NodeKind::Load: case NodeKind::Store: case NodeKind::LoadField:
        case NodeKind::StoreField: case NodeKind::Allocated:
        case NodeKind::RegionFree: case NodeKind::Altered:
            return NodeClass::Memory;
        case NodeKind::Guard: case NodeKind::DeoptBarrier: case NodeKind::FrameStateNode:
            return NodeClass::Speculation;
        case NodeKind::VecPack: case NodeKind::VecExtract: case NodeKind::VecLoad:
        case NodeKind::VecStore: case NodeKind::VecOp: case NodeKind::Gather:
        case NodeKind::Scatter: case NodeKind::VecShuffle: case NodeKind::VecReduce:
            return NodeClass::Vector;
        case NodeKind::TupleProj: case NodeKind::CallResult: case NodeKind::DebugBreak:
            return NodeClass::Misc;
        case NodeKind::KindCount:
            break;
    }
    return NodeClass::Misc;
}

[[nodiscard]] constexpr bool is_control(NodeKind k) noexcept {
    return node_class(k) == NodeClass::Control;
}
[[nodiscard]] constexpr bool is_arith(NodeKind k) noexcept {
    return node_class(k) == NodeClass::Arith;
}
[[nodiscard]] constexpr bool is_pyop(NodeKind k) noexcept {
    return node_class(k) == NodeClass::PythonOp;
}
[[nodiscard]] constexpr bool is_memory(NodeKind k) noexcept {
    return node_class(k) == NodeClass::Memory;
}
[[nodiscard]] constexpr bool is_vector(NodeKind k) noexcept {
    return node_class(k) == NodeClass::Vector;
}
[[nodiscard]] constexpr bool is_value_node(NodeKind k) noexcept {
    return !is_control(k);
}

/// Canonical short names — used by the IR printer/parser (golden format).
[[nodiscard]] const char* node_kind_name(NodeKind k) noexcept;
/// Inverse of node_kind_name; returns false on unknown token.
[[nodiscard]] bool node_kind_from_name(const char* name, std::size_t len, NodeKind& out) noexcept;

}  // namespace abi_v1
}  // namespace vortex::ir
