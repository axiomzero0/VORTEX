// =============================================================================
// vortex/rt/jit.cpp — Tier 1/2 JIT runtime integration.
//
// vortex_jit_bridge(regs, unit_id, node_id):
//   Called when the JIT hits a dynamic op (CALLri). The bridge:
//   1. Looks up the Tier-0 PC for the IR NodeId via node_id_to_pc
//   2. Creates a Frame using the JIT's register file
//   3. Executes ONE Tier-0 instruction (step_one)
//   4. Returns the result Value — the JIT CONTINUES after the CALL
//
// This is the key change: the bridge RETURNS to the JIT. Previously it
// used JMP (one-way tail call) and ran the entire function in Tier-0.
// Now the JIT runs the whole function, only bridging for individual
// dynamic ops — the all-or-nothing has_dynamic_ops gate is removed.
// =============================================================================

#include "vortex/rt/interp.hpp"

#include <cstdio>
#include <cstdlib>
#include <sys/mman.h>
#include <unistd.h>

namespace vortex::rt {
inline namespace abi_v1 {

[[nodiscard]] static CodeUnit* find_unit(std::uint32_t unit_id) noexcept {
    Vm* vm = active_vm();
    if (!vm) return nullptr;
    if (unit_id >= vm->program.units.size()) return nullptr;
    return vm->program.units[unit_id];
}

}  // namespace abi_v1
}  // namespace vortex::rt;

extern "C" vortex::Value vortex_jit_bridge(void* regs_raw, std::uint32_t unit_id,
                                           std::uint64_t op_hint) noexcept {
    using namespace vortex::rt;
    // Rule 120: All error paths return none() for graceful Tier-0 fallback.
    // Rule 26: All error paths record CompileFailed in telemetry.
    CodeUnit* unit = find_unit(unit_id);
    if (!unit) {
        if (Vm* vm = active_vm()) {
            vm->telemetry.record(vortex::TelemetryEventKind::CompileFailed, unit_id, 0, 0);
        }
        return vortex::Value::none();
    }
    Vm* vm = active_vm();
    if (!vm) {
        return vortex::Value::none();
    }
    if (!regs_raw) {
        vm->telemetry.record(vortex::TelemetryEventKind::CompileFailed, unit_id, 1, 0);
        return vortex::Value::none();
    }

    Value* regs = static_cast<Value*>(regs_raw);
    const std::uint32_t n_regs = unit->n_registers;

    // op_hint is the IR NodeId (the CALLri's home slot). Look up the
    // corresponding Tier-0 PC via the node_id_to_pc map.
    const std::uint32_t node_id = static_cast<std::uint32_t>(op_hint);
    std::uint32_t resume_pc = 0xFFFF'FFFFu;
    if (node_id < unit->node_id_to_pc.size()) {
        resume_pc = unit->node_id_to_pc[node_id];
    }
    // Fallback: linear scan of the code array for an instruction with
    // dst == node_id. O(N) but only for the rare case where the map
    // wasn't populated (e.g., the scheduler emitted the instruction in
    // a different block than expected, or the node was rematerialized).
    if (resume_pc == 0xFFFF'FFFFu || resume_pc >= unit->code.size()) {
        for (std::uint32_t i = 0; i < unit->code.size(); ++i) {
            if (unit->code[i].dst == node_id) {
                resume_pc = i;
                break;
            }
        }
    }
    if (resume_pc == 0xFFFF'FFFFu || resume_pc >= unit->code.size()) {
        // Truly unmapped — this is a bug in the node_id_to_pc map.
        // Return none() and set a runtime error so the caller can handle it.
        // Do NOT call enter_at here — it would shallow-copy regs and
        // double-decref them in the Frame destructor.
        vm->raise_builtin(Runtime::instance().type_runtime_error,
                          "jit bridge: unmapped node");
        return vortex::Value::none();
    }

    Value out;
    bool ok = vm->step_one(unit, regs, n_regs, resume_pc, out);
    if (!ok) {
        // step_one failed (exception or unhandled op).
        // If there's a pending exception, propagate it.
        if (vm->has_pending()) {
            return vortex::Value::none();
        }
        // Unhandled op: run the rest of the function in Tier-0.
        // We can't return to the JIT with a wrong result (the JIT would
        // silently produce incorrect values). Instead, create a Frame,
        // copy regs (with incref), run exec_frame with jit_disabled_in_bridge
        // = true to prevent recursion, copy results back, and return
        // the function's return value.
        Frame f(unit);
        Runtime& rt = Runtime::instance();
        for (std::uint32_t i = 0; i < n_regs && i < f.n_regs; ++i) {
            f.regs[i] = regs[i];
            if (regs[i].tag == Tag::Obj && regs[i].as.obj) rt.incref(regs[i].as.obj);
        }
        f.pc = resume_pc;
        bool prev = vm->jit_disabled_in_bridge;
        vm->jit_disabled_in_bridge = true;
        ExecStatus st = vm->exec_frame(f);
        vm->jit_disabled_in_bridge = prev;
        // Copy results back to the JIT's register file.
        for (std::uint32_t i = 0; i < n_regs && i < f.n_regs; ++i) {
            regs[i] = f.regs[i];
            // The Frame's destructor will decref its copy; the JIT's copy
            // needs its own incref.
            if (f.regs[i].tag == Tag::Obj && f.regs[i].as.obj) rt.incref(f.regs[i].as.obj);
        }
        if (st == ExecStatus::Returned) {
            out = vm->frame_return_;
            vm->frame_return_ = vortex::Value::none();
            return out;
        }
        return vortex::Value::none();
    }
    return out;
}

extern "C" void vortex_rt_munmap_jit_buffer(void* buf, std::size_t cap) noexcept {
    if (!buf) return;
    long pagesz = sysconf(_SC_PAGESIZE);
    if (pagesz <= 0) pagesz = 4096;
    std::size_t mapped = ((cap + pagesz - 1) / pagesz) * pagesz;
    munmap(buf, mapped);
}

// =============================================================================
// Trace compiler C shim: executes ONE Tier-0 instruction via step_one.
//
// The trace compiler emits a CALL to this function for any op it can't
// compile inline (CALL, LOAD_GLOBAL, LOAD_ATTR, etc.). The shim
// executes the instruction via step_one and returns. The trace
// continues after the shim returns.
//
// Parameters:
//   regs    — the register file (same as trace_fn's argument)
//   unit_id — the code unit ID (for looking up the CodeUnit)
//   pc      — the Tier-0 PC of the instruction to execute
//
// Returns: the result value of the instruction (for most ops, this is
// written to regs[dst] by step_one; for CALL, the result is returned).
// =============================================================================

extern "C" vortex::Value vortex_trace_step_one(void* regs_raw, std::uint32_t unit_id,
                                                std::uint32_t pc) noexcept {
    using namespace vortex::rt;
    Vm* vm = active_vm();
    if (!vm) return vortex::Value::none();
    CodeUnit* unit = nullptr;
    if (unit_id < vm->program.units.size()) {
        unit = vm->program.units[unit_id];
    }
    if (!unit) return vortex::Value::none();
    Value* regs = static_cast<Value*>(regs_raw);
    Value out;
    bool ok = vm->step_one(unit, regs, unit->n_registers, pc, out);
    if (!ok) {
        return vortex::Value::none();  // deopt signal
    }
    return out;
}

// =============================================================================
// Trace call: fast function call from compiled traces.
//
// When a compiled trace encounters a CALL instruction, it calls this
// function instead of step_one. This function:
//   1. Reads the CALL instruction to get callee + args.
//   2. If the callee has a compiled function trace, invokes it DIRECTLY
//      — no Frame allocation, no interpreter dispatch. Just copy args
//      to a stack-allocated register array and call the trace function.
//   3. If no trace exists, falls back to call_value (creates Frame,
//      calls exec_frame).
//
// This is the key optimization for recursive functions: each recursive
// call invokes the callee's trace directly, skipping Frame allocation
// and interpreter dispatch. For fib(20), this eliminates 21891 Frame
// constructions and 21891 exec_frame dispatch loops.
// =============================================================================

extern "C" vortex::Value vortex_trace_call(void* regs_raw, std::uint32_t unit_id,
                                             std::uint32_t pc) noexcept {
    using namespace vortex::rt;
    Vm* vm = active_vm();
    if (!vm) return vortex::Value::none();
    if (unit_id >= vm->program.units.size()) return vortex::Value::none();
    CodeUnit* unit = vm->program.units[unit_id];
    if (!unit || pc >= unit->code.size()) return vortex::Value::none();

    Value* caller_regs = static_cast<Value*>(regs_raw);
    const Instr& instr = unit->code[pc];
    Value callee = caller_regs[instr.a];
    Value* args = instr.c > 0 ? &caller_regs[instr.b] : nullptr;
    std::uint32_t argc = instr.c;

    // Fast path: if the callee is a Function with a compiled trace,
    // invoke the trace directly with a stack-allocated register file.
    if (callee.tag == Tag::Obj && callee.as.obj &&
        callee.as.obj->tag == ObjTag::Function) {
        auto* fn = static_cast<PyFuncObj*>(callee.as.obj);
        if (fn->code_unit_id < vm->program.units.size()) {
            CodeUnit* callee_unit = vm->program.units[fn->code_unit_id];
            if (callee_unit && !callee_unit->is_generator) {
                // Check for a compiled function trace.
                std::uint64_t fkey = (static_cast<std::uint64_t>(callee_unit->id) << 16);
                Trace** pp = vm->tracer.traces.get(fkey);
                if (pp && *pp && (*pp)->native_code &&
                    // Only invoke if profitable (inline >= shim*3)
                    (*pp)->inline_op_count >= (*pp)->shim_op_count * 3) {
                    // Stack-allocate the callee's register file.
                    // max_registers_per_frame = 256, sizeof(Value) = 16
                    // = 4KB on the stack — fine for recursion depth < ~500.
                    constexpr std::uint32_t kMaxStackRegs = 256;
                    Value callee_regs[kMaxStackRegs];
                    std::uint32_t n_regs = callee_unit->n_registers;
                    if (n_regs > kMaxStackRegs) n_regs = kMaxStackRegs;
                    // Initialize to None.
                    for (std::uint32_t i = 0; i < n_regs; ++i) {
                        callee_regs[i] = Value::none();
                    }
                    // Copy arguments to parameter slots.
                    // callee_unit->param_regs[i] is the register index for param i.
                    for (std::uint32_t i = 0; i < argc && i < callee_unit->param_regs.size(); ++i) {
                        callee_regs[callee_unit->param_regs[i]] = args[i];
                    }
                    // Invoke the callee's trace function.
                    auto trace_fn = reinterpret_cast<Value(*)(Value*)>((*pp)->native_code);
                    Value rv = trace_fn(callee_regs);
                    if (rv.tag != Tag::None) {
                        // Success! Record the guard outcome.
                        vm->profiler.record_guard((*pp)->header_pc, true);
                        (*pp)->consecutive_deopts = 0;
                        return rv;
                    }
                    // Deopt — record and fall through to interpreter.
                    vm->profiler.record_guard((*pp)->header_pc, false);
                }
            }
        }
    }

    // Fallback: call_value (creates Frame, calls exec_frame).
    Value out;
    if (!vm->call_value(callee, args, argc, out)) {
        return vortex::Value::none();  // deopt signal
    }
    return out;
}
