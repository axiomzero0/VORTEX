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
    // Rule 120: Compiler bugs must not crash user programs. All error paths
    // here return none() (triggering Tier-0 fallback) instead of aborting.
    CodeUnit* unit = find_unit(unit_id);
    if (!unit) {
        std::fprintf(stderr, "VORTEX jit bridge: unknown unit id %u — falling back to Tier-0\n",
                     unit_id);
        return vortex::Value::none();
    }
    Vm* vm = active_vm();
    if (!vm) {
        std::fputs("VORTEX jit bridge: no active VM — falling back\n", stderr);
        return vortex::Value::none();
    }
    if (!regs_raw) {
        std::fputs("VORTEX jit bridge: null regs — falling back\n", stderr);
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
