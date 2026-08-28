// =============================================================================
// vortex/rt/interp.hpp — Tier-0 direct-threaded register interpreter
//
// Purpose:
//   Executes CodeUnit bytecode with computed-goto dispatch (GCC labels as
//   values). Python-level exceptions are explicit control flow (Rule 6: no
//   C++ exceptions): ops either succeed or set the frame's pending exception
//   which unwinds through try ranges or to the caller.
//
// PGO hooks (Rule 2/11): call sites bump CodeUnit::call_count; backward
//   jumps bump backedge_count — the tiering daemon reads these counters to
//   trigger Tier 1/2 compilation without ever blocking the mutator.
//
// Deopt contract (Rule 4): enter_at() resumes execution at an arbitrary
//   bytecode offset with a reconstructed register file — exactly what the
//   Deoptimizer uses to land from JIT guards.
// =============================================================================

#pragma once

#include "vortex/stdx/flat_map.hpp"
#include "vortex/rt/code.hpp"
#include "vortex/rt/object.hpp"
#include "vortex/support/result.hpp"
#include "vortex/stdx/small_vector.hpp"
#include "vortex/support/telemetry.hpp"

namespace vortex::rt {

inline namespace abi_v1 {

/// Bound method: func(recv, args...).
struct PyBoundMethodObj : PyObj {
    Value func{};   // owned
    Value recv{};   // owned
};

struct Program {
    stdx::small_vector<CodeUnit*, 64> units{};   // index == code unit id
    PyDictObj* globals{nullptr};                 // module globals (owned)
    stdx::small_vector<PyModuleObj*, 4> native_modules{};
    stdx::flat_map<std::uint32_t, PyStrObj*, 64> sym_key_cache{};   // owned lookup keys

    Program() { globals = Runtime::instance().new_dict(); }
    ~Program();
};

enum class ExecStatus : std::uint8_t { Returned, Raised, Suspended };

class Vm {
public:
    Vm() noexcept = default;

    Program program{};

    /// Run the module toplevel unit; returns the module's last value or a
    /// Diagnostic describing the uncaught exception (with traceback text).
    [[nodiscard]] Result<Value> run_module(CodeUnit* unit) noexcept;

    /// The dispatch loop. Returns Returned (reg per frame convention: result
    /// in `frame_return_`), Raised (pending_exception_), or Suspended
    /// (frame parked; yielded value in frame_return_).
    ExecStatus exec_frame(Frame& f) noexcept;

    /// Call protocol: functions, natives, types (instantiation), bound
    /// methods, generators (creation). `out` receives an owned ref.
    [[nodiscard]] bool call_value(const Value& callee, Value* args, std::uint32_t argc,
                                  Value& out) noexcept;

    /// Keyword-aware call: kw_names is a tuple of symbol-int consts.
    [[nodiscard]] bool call_value_kw(const Value& callee, Value* args, std::uint32_t argc,
                                     PyTupleObj* kw_names, std::uint32_t nkw, Value& out) noexcept;

    /// Deopt entry (Rule 4): materialize `regs` (owned refs transferred in)
    /// and resume at `pc`.
    [[nodiscard]] bool enter_at(CodeUnit* unit, Value* regs, std::uint32_t n_regs,
                                std::uint32_t pc, Value& out) noexcept;

    /// Execute ONE Tier-0 instruction at `pc`, reading/writing `regs`
    /// directly. Used by the JIT bridge — no Frame allocation overhead.
    /// Returns true on success (result in `out`), false on exception.
    /// On exception, the caller should fall back to enter_at.
    [[nodiscard]] bool step_one(CodeUnit* unit, Value* regs, std::uint32_t n_regs,
                                std::uint32_t pc, Value& out) noexcept;

    /// Native helper dispatch for CallNative IR ops.
    [[nodiscard]] bool native_helper(std::uint16_t helper, Value* args, std::uint32_t argc,
                                     Value& out) noexcept;

    [[nodiscard]] Value pending_exception() const noexcept { return pending_exception_; }
    [[nodiscard]] bool has_pending() const noexcept {
        return pending_exception_.tag == Tag::Obj && pending_exception_.as.obj != nullptr;
    }

    /// Raise a builtin exception by type (sets pending).
    void raise_builtin(PyTypeObj* type, const char* message) noexcept;

    /// Generator stepping (ITER_CHECK on generators advances + caches).
    [[nodiscard]] bool generator_step(PyGeneratorObj* gen, bool& has_value,
                                      Value& out) noexcept;

    Telemetry telemetry{};
    std::uint32_t call_depth{0};
    Value frame_return_{};

    /// When true, the CALL handler skips jit_entry and runs Tier-0 directly.
    /// Set by the bridge before calling step_one, so ops that need full
    /// Frame machinery (call_value → exec_frame) don't re-enter the JIT
    /// and cause infinite recursion.
    bool jit_disabled_in_bridge{false};

private:
    Value pending_exception_{};   // owned ref
    stdx::small_vector<char, 256> traceback_{};

    [[nodiscard]] bool unwind_to_handler(Frame& f) noexcept;
    void set_pending(Value exc_owned) noexcept;
    [[nodiscard]] Value take_pending() noexcept;
    [[nodiscard]] bool bind_parameters(Frame& f, PyFuncObj* fn, Value* args,
                                       std::uint32_t argc, PyTupleObj* kw_names,
                                       std::uint32_t nkw) noexcept;
    [[nodiscard]] bool get_attr(const Value& obj, std::uint32_t symbol, Value& out) noexcept;
    [[nodiscard]] bool set_attr(const Value& obj, std::uint32_t symbol, Value value) noexcept;
public:
    [[nodiscard]] bool get_iter(const Value& obj, Value& out) noexcept;
    [[nodiscard]] bool iter_check(Value& it, bool& more) noexcept;
    [[nodiscard]] bool iter_next(const Value& it, Value& out) noexcept;
    [[nodiscard]] bool call_function(const Value& callee, Value* args, std::uint32_t argc,
                                     Value& out) noexcept {
        return call_value(callee, args, argc, out);
    }

private:
    [[nodiscard]] bool get_global(std::uint32_t symbol, Value& out) noexcept;
    [[nodiscard]] bool builtin_call(PyNativeFnObj* fn, Value* args, std::uint32_t argc,
                                    Value& out) noexcept;
    [[nodiscard]] bool builtin_bound_method(std::uint64_t kind, const Value& recv,
                                            Value* args, std::uint32_t argc,
                                            Value& out) noexcept;
};

/// Install builtin functions & exception types into program globals.
void install_builtins(Program& program) noexcept;

/// Bind the active VM for builtins needing call-backs (map/filter/next).
void set_vm_for_builtins(Vm* vm) noexcept;

/// Access the active VM (set by set_vm_for_builtins). Used by the JIT
/// deopt/bridge entry points (deopt.cpp, jit.cpp) to find the CodeUnit
/// and call enter_at on guard failure / dynamic-op fallback.
[[nodiscard]] Vm* active_vm() noexcept;

/// Load a native module by name (math / time / random); null if unknown.
[[nodiscard]] PyModuleObj* load_native_module(std::uint32_t name_symbol) noexcept;

}  // namespace abi_v1
}  // namespace vortex::rt
