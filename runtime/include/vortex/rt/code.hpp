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
// Instruction encoding: fixed 12 bytes (op, dst, a, b, c, imm) — uniform
// decode keeps the computed-goto dispatch loop tight.
// =============================================================================

#pragma once

#include <atomic>
#include <cstdint>

#include "vortex/common/value.hpp"
#include "vortex/rt/object.hpp"
#include "vortex/stdx/small_vector.hpp"
#include "vortex/support/symbol_table.hpp"

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
        regs = static_cast<Value*>(std::malloc(sizeof(Value) * (n_regs ? n_regs : 1)));
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
