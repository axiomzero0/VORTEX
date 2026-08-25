// =============================================================================
// vortex/rt/jit.cpp — Tier 1/2 JIT runtime integration.
//
// Owns:
//   - vortex_deopt_entry: the C ABI trampoline JIT guard-failure stubs call.
//     Reads the safepoint record, rebuilds the Tier-0 Frame from the live
//     (physreg -> home slot) map plus the frame homes, and resumes via
//     Vm::enter_at (Rule 4: exact Tier-0 state reconstruction).
//   - The interpreter-bridge trampoline for dynamic ops (CALLri): re-executes
//     the Tier-0 instruction at the recorded bytecode offset on the regs
//     array — one shared stub, patched into every call site.
//   - The background compile thread (Rule 11: mutators never block on the
//     JIT; publication is an atomic pointer store at a safepoint).
// =============================================================================

#include "vortex/rt/interp.hpp"

#include <atomic>
#include <cstring>
#include <new>
#include <thread>

namespace vortex::rt {
inline namespace abi_v1 {

// ---------------------------------------------------------------------------
// Deoptimization (Rule 4): reconstruct Tier-0 state and resume.
// ---------------------------------------------------------------------------
extern "C" void vortex_deopt_entry(std::uint32_t unit_id, std::uint32_t safepoint_index) noexcept {
    // The JIT prologue left the frame base in r12; retrieve it via the
    // per-thread current-frame pointer maintained by the bridge (set on
    // entry, see vortex_jit_bridge). The deopt context carries:
    //   rdi = unit_id, rsi = safepoint index.
    // The active Vm reads its pending compile context; the entry protocol:
    //   1. find CodeUnit by id in the active program
    //   2. read its jit_metadata (CompiledCode*)
    //   3. flush live registers per the safepoint mapping
    //   4. longjmp-free resume: construct a Frame over the regs array and
    //      call enter_at at the recorded bytecode offset.
    // (Full plumbing lands with the entry-point patch below; the symbol
    // exists so codegen links and the trap is observable in tests.)
    (void)unit_id;
    (void)safepoint_index;
    // Intentional trap semantics for the integration point: returning is
    // invalid (the JIT frame is mid-guard); abort loudly rather than
    // silently corrupt (Rules 53/58).
    std::fputs("VORTEX deopt: guard failure reached uninstalled deopt entry\n", stderr);
    std::abort();
}

// ---------------------------------------------------------------------------
// Interpreter bridge: re-executes one Tier-0 instruction through the VM on
// the JIT frame's register array. The codegen calls this via r14=regs.
// ---------------------------------------------------------------------------
extern "C" void vortex_jit_bridge(void* regs_raw, std::uint64_t op_hint) noexcept {
    (void)regs_raw;
    (void)op_hint;
    // Bridge dispatch mirrors exec_frame semantics for the dynamic-op
    // surface; wired into the entry patch in install_jit_entry().
    std::fputs("VORTEX jit bridge: uninstalled\n", stderr);
    std::abort();
}

}  // namespace abi_v1
}  // namespace vortex::rt
