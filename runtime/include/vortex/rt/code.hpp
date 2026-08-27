// =============================================================================
// vortex/rt/code.hpp — Tier-0 register bytecode & frames
//
// Purpose:
//   The scheduled form of a VORTEX IR unit: a linear register-based
//   instruction stream executed by the Tier-0 direct-threaded interpreter.
//   Register indices are SSA NodeIds (dense per unit) — the same mapping the
//   FrameState uses, so deoptimization from JIT code rebuilds Tier-0 frames
//   by writing FrameState values straight into register slots (Rule 4).
//
// Instruction encoding: fixed 16 bytes (op, dst, a, b, c, imm) — uniform
// decode keeps the computed-goto dispatch loop tight. TTC-10: the previous
// header said "12 bytes" but the static_assert (line 76) enforces 16 — five
// 16-bit fields plus one 32-bit immediate. The header now matches reality.
// =============================================================================

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "vortex/common/value.hpp"
#include "vortex/rt/object.hpp"
#include "vortex/stdx/small_vector.hpp"
#include "vortex/support/symbol_table.hpp"

// Task 24: extern "C" shim for munmap. Defined in runtime/src/jit.cpp
// (which already includes <sys/mman.h>). The CodeUnit destructor calls
// this to free the RWX JIT buffer without forcing every TU that
// includes code.hpp to pull <sys/mman.h>. Declared BEFORE the
// vortex::rt namespace opens so the CodeUnit destructor's name lookup
// finds it (the destructor body is parsed at the class definition
// point, where only prior declarations are visible).
extern "C" void vortex_rt_munmap_jit_buffer(void* buf, std::size_t cap) noexcept;

namespace vortex::rt {

inline namespace abi_v1 {

enum class Op : std::uint16_t {
    // data movement
    LOAD_CONST = 0,   // dst, imm=const pool index
    MOVE,             // dst, a=src
    // python-level ops (full dynamic semantics via runtime)
    PY_BINOP,         // dst, a, b, imm=BinOpKind
    PY_UNOP,          // dst, a, imm=unary kind
    PY_CMP,           // dst, a, b, imm=CmpOpKind
    LOAD_GLOBAL,      // dst, imm=symbol
    STORE_GLOBAL,     // a, imm=symbol
    LOAD_ATTR,        // dst, a=obj, imm=symbol
    STORE_ATTR,       // a=obj, b=value, imm=symbol
    LOAD_INDEX,       // dst, a=obj, b=index
    STORE_INDEX,      // a=obj, b=index, c=value
    NEW_LIST,         // dst, a=first elem reg, b=count
    NEW_TUPLE,        // dst, a=first elem reg, b=count
    NEW_DICT,         // dst
    LIST_APPEND,      // a=list, b=value
    CALL,             // dst, a=func, b=first arg reg, c=argc, imm=flags
    CALL_KW,          // dst, a=func, b=first arg reg, c=argc, imm=kw names const idx
    NATIVE,           // dst, a=first arg reg, b=argc, imm=NativeHelper
    ITER,             // dst, a=obj
    ITER_CHECK,       // dst=bool, a=iterator (may advance generators)
    ITER_NEXT,        // dst, a=iterator (returns cached value)
    YIELD,            // dst=result(sent), a=value
    // control
    JUMP,             // imm=target pc
    JUMP_IF_FALSE,    // a=cond, imm=target
    JUMP_IF_TRUE,     // a=cond, imm=target
    RETURN,           // a=value
    RAISE,            // a=exception value
    // exception machinery
    TRY_BEGIN,        // imm=handler pc
    TRY_END,          // closes innermost TRY_BEGIN
    GET_EXC,          // dst=current exception (owned ref)
    LOAD_FIELD,       // dst, a=base, imm=slot        (pass 46 fast path)
    STORE_FIELD,      // a=base, b=value, imm=slot    (pass 46 fast path)
};

struct Instr {
    std::uint16_t op{0};
    std::uint16_t dst{0};
    std::uint16_t a{0};
    std::uint16_t b{0};
    std::uint16_t c{0};
    std::uint32_t imm{0};
};
static_assert(sizeof(Instr) == 16, "Instr stays 16 bytes (uniform decode, cache-aligned pairs)");

struct TryRange {
    std::uint32_t start_pc{0};
    std::uint32_t end_pc{0};
    std::uint32_t handler_pc{0};
};

struct CodeUnit {
    std::uint32_t id{0};
    SymbolId name{0xFFFF'FFFF};
    stdx::small_vector<Instr, 64> code{};
    stdx::small_vector<Value, 16> constants{};      // owned refs (Tag::Obj)
    stdx::small_vector<std::uint32_t, 8> param_regs{};
    stdx::small_vector<SymbolId, 8> param_names{};
    stdx::small_vector<TryRange, 4> try_ranges{};
    std::uint32_t n_registers{0};
    bool is_generator{false};
    bool has_varargs{false};
    bool has_kwargs{false};

    // --- tiering state (Rule 11: mutator never blocks on JIT) ------------------
    mutable std::atomic<std::uint64_t> call_count{0};
    mutable std::atomic<std::uint64_t> backedge_count{0};
    mutable std::atomic<std::int32_t> current_tier{0};
    std::atomic<void*> jit_entry{nullptr};   // safepoint-swapped machine code
    void* jit_metadata{nullptr};             // deopt tables, owned by backend
    std::uint32_t deopt_count{0};            // telemetry (Rule 26)
    /// Task 24: RWX mmap'd buffer holding the JIT-compiled machine code.
    /// Owned by the runtime; freed in the destructor. Allocated by the
    /// driver when `backend::compile_unit` succeeds; null when the unit
    /// runs Tier-0 only.
    std::byte* jit_code_buffer{nullptr};
    std::size_t jit_code_capacity{0};
    /// Task 24: true if the JIT'd code for this unit contains any CALLri
    /// fallback (PyBinary/PyCompare/CallPy/etc. without a fast path).
    /// Copied from `backend::CompiledCode::has_dynamic_ops` by the
    /// driver. The CALL handler consults this to decide whether to
    /// invoke jit_entry: when true, calling jit_entry would hit the
    /// bridge path which needs safepoint_pcs populated (not yet
    /// implemented); fall back to exec_frame instead.
    bool has_dynamic_ops{false};
    /// Task 24: initial values for Phi home slots at function entry.
    /// Populated by the driver after compile_unit, by walking the IR
    /// graph and recording each Phi whose entry input (ins[0]) is a
    /// ConstInt or ConstFloat. The CALL handler writes these to
    /// regs[phi_node_id] before calling jit_entry — without this, the
    /// JIT's prologue (which assumes the entry block's home slots are
    /// populated by the runtime, the way Tier-0's LOAD_CONST+MOVE
    /// prologue does) would read Value::none() and every GUARD_INT /
    /// GUARD_FLOAT would fail on the first read.
    ///
    /// Three parallel arrays:
    ///   phi_init_node_ids[i]  = the Phi's IR NodeId (= home slot index)
    ///   phi_init_values[i]    = the entry value (int64 — for floats,
    ///                           the bit-reinterpreted double bits)
    ///   phi_init_is_float[i]  = 1 if Value::real should be used,
    ///                           0 if Value::integer
    stdx::small_vector<std::uint32_t, 8> phi_init_node_ids{};
    stdx::small_vector<std::int64_t, 8>  phi_init_values{};
    stdx::small_vector<std::uint8_t, 8>  phi_init_is_float{};

    // --- JIT safepoint → Tier-0 resume-point map (Rule 4) ---------------------
    // Indexed by the safepoint_index that the JIT's deopt stub passes to
    // vortex_deopt_entry / vortex_jit_bridge. Each entry is the Tier-0
    // bytecode offset at which to resume interpretation after a guard
    // failure or a dynamic-op fallback. Populated by the runtime when
    // JIT code is installed; zero entries means "no JIT installed".
    stdx::small_vector<std::uint32_t, 16> safepoint_pcs{};

    ~CodeUnit() {
        Runtime& rt = Runtime::instance();
        for (Value& v : constants) {
            if (v.tag == Tag::Obj && v.as.obj) rt.decref(v.as.obj);
        }
        // Task 24: free the RWX mmap'd JIT code buffer (if any).
        // We don't pull <sys/mman.h> into every TU that includes this
        // header; instead the runtime exposes a small C shim
        // `vortex_rt_munmap_jit_buffer(buf, cap)` defined in
        // runtime/src/jit.cpp, which calls munmap.
        if (jit_code_buffer) {
            vortex_rt_munmap_jit_buffer(jit_code_buffer, jit_code_capacity);
            jit_code_buffer = nullptr;
        }
    }
};

// --- Frame --------------------------------------------------------------------
struct Frame {
    CodeUnit* unit{nullptr};
    std::uint32_t pc{0};
    Value* regs{nullptr};             // owned refs, n_registers slots
    std::uint32_t n_regs{0};

    // exception handler stack (sp + ranges)
    struct Handler {
        std::uint32_t try_pc_start{0};
        std::uint32_t try_pc_end{0};
        std::uint32_t handler_pc{0};
    };
    stdx::small_vector<Handler, 4> handlers{};
    Value current_exception{};        // owned ref

    // generator suspension
    bool suspended{false};

    explicit Frame(CodeUnit* u) noexcept : unit(u) {
        n_regs = u ? u->n_registers : 0;
        // TTC-17 fix: malloc can return nullptr under memory pressure.
        // We used to write `for (i=0; i<n_regs; ++i) regs[i] = ...` unconditionally —
        // a null regs dereferences, segfaults the whole VM, and the failure
        // mode is opaque. Mark the frame as "no regs" so the destructor skips
        // the decref loop and future dispatch can either raise an OOM-style
        // Python exception or, lacking one, raise a clean fatal diagnostic.
        const std::size_t bytes = sizeof(Value) * (n_regs ? n_regs : 1);
        regs = static_cast<Value*>(std::malloc(bytes));
        if (!regs) [[unlikely]] {
            // OOM. The destructor checks regs before iterating; the
            // interpreter checks `unit != nullptr && regs != nullptr`
            // before running. Print a hard diagnostic (Rule 47) — we
            // cannot raise a Python-level MemoryError without a frame.
            std::fprintf(stderr,
                         "VORTEX FATAL: Frame allocation of %zu bytes failed "
                         "(resource exhaustion — see docs/adr/0007-resource-policy.md)\n",
                         bytes);
            n_regs = 0;
            return;
        }
        for (std::uint32_t i = 0; i < n_regs; ++i) regs[i] = Value::none();
    }
    ~Frame() {
        Runtime& rt = Runtime::instance();
        for (std::uint32_t i = 0; i < n_regs; ++i) {
            if (regs && regs[i].tag == Tag::Obj && regs[i].as.obj) rt.decref(regs[i].as.obj);
        }
        if (current_exception.tag == Tag::Obj && current_exception.as.obj) {
            rt.decref(current_exception.as.obj);
        }
        std::free(regs);
    }
    Frame(const Frame&) = delete;
    Frame& operator=(const Frame&) = delete;
};

}  // namespace abi_v1
}  // namespace vortex::rt
