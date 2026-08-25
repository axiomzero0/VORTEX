// =============================================================================
// vortex/rt/deopt.cpp — Deoptimization engine (Rule 4).
//
// Real implementation: when a JIT guard fails, the cold stub tail-calls
// vortex_deopt_entry with (unit_id, safepoint_index, regs_base). The
// deoptimizer:
//   1. Looks up the active Vm via the global builtins pointer.
//   2. Finds the CodeUnit by id in vm.program.units.
//   3. Translates safepoint_index to a Tier-0 bytecode offset via
//      unit->safepoint_pcs (populated when JIT code was installed;
//      the compiler emits a SafepointRecord per guard site carrying
//      the IR frame_state_id, and the runtime materializes the
//      Tier-0 PC into this table).
//   4. Transfers ownership of the regs array into a new Tier-0 Frame
//      and resumes via Vm::enter_at at the resumed PC. The Tier-0
//      interpreter runs to completion, writes the result to
//      vm.frame_return_, and returns it to the JIT's caller.
//
// Returns a Value (not void) — the JIT's calling convention expects
// RAX/RDX to carry the result tag/payload, and a tail-call to this
// function must propagate that. The previous void return broke the
// Value-return contract.
// =============================================================================

#include "vortex/rt/interp.hpp"

#include <cstdio>
#include <cstdlib>

namespace vortex::rt {
inline namespace abi_v1 {

[[nodiscard]] static CodeUnit* find_unit(std::uint32_t unit_id) noexcept {
    Vm* vm = active_vm();
    if (!vm) return nullptr;
    if (unit_id >= vm->program.units.size()) return nullptr;
    return vm->program.units[unit_id];
}

}  // namespace abi_v1
}  // namespace vortex::rt

extern "C" vortex::Value vortex_deopt_entry(std::uint32_t unit_id,
                                            std::uint32_t safepoint_index,
                                            void* regs_raw) noexcept {
    using namespace vortex::rt;
    CodeUnit* unit = find_unit(unit_id);
    if (!unit) {
        std::fputs("VORTEX deopt: unknown unit id\n", stderr);
        std::abort();
    }
    Vm* vm = active_vm();
    if (!vm) {
        std::fputs("VORTEX deopt: no active VM\n", stderr);
        std::abort();
    }
    if (!regs_raw) {
        std::fputs("VORTEX deopt: null regs\n", stderr);
        std::abort();
    }

    const std::uint32_t n_regs = unit->n_registers;
    Value* regs = static_cast<Value*>(regs_raw);

    // Translate the JIT's safepoint_index into a Tier-0 bytecode offset.
    // The runtime populates unit->safepoint_pcs when JIT code is
    // installed; zero entries means "no JIT installed" — which means
    // vortex_deopt_entry should not have been called at all. We fall
    // back to pc=0 only as a last-resort safety net (with a stderr
    // note) so a misconfigured test doesn't loop forever.
    std::uint32_t resume_pc = 0;
    if (safepoint_index < unit->safepoint_pcs.size()) {
        resume_pc = unit->safepoint_pcs[safepoint_index];
    } else if (!unit->safepoint_pcs.empty()) {
        // Out-of-range safepoint index: this is a bug, not a fallback.
        std::fprintf(stderr, "VORTEX deopt: safepoint_index %u out of range "
                             "(size %zu)\n",
                     safepoint_index, unit->safepoint_pcs.size());
        std::abort();
    } else {
        std::fputs("VORTEX deopt: no safepoint table — falling back to pc=0\n",
                   stderr);
    }

    Value out;
    bool ok = vm->enter_at(unit, regs, n_regs, resume_pc, out);
    if (!ok) {
        // enter_at reported an exception via vm->pending_exception();
        // return None so the JIT's caller observes a defined value.
        return vortex::Value::none();
    }
    return out;
}
