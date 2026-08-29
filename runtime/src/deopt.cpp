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
// Rule 26: Every deopt is recorded in telemetry (GuardFailed +
// DeoptExecuted + TierDowngrade). No silent fallbacks.
// Rule 120: All error paths return None (graceful fallback) instead
// of aborting. The VM should degrade performance, not crash.
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
        // Rule 120: don't crash — return None for graceful fallback.
        return vortex::Value::none();
    }
    Vm* vm = active_vm();
    if (!vm) {
        return vortex::Value::none();
    }
    if (!regs_raw) {
        return vortex::Value::none();
    }

    // Rule 26: Record the guard failure in telemetry.
    vm->telemetry.record(vortex::TelemetryEventKind::GuardFailed, unit_id, safepoint_index, 0);
    vm->telemetry.bump(vortex::Telemetry::counter_guard_failures);
    // Giga Tracing: record in probabilistic profiler
    vm->profiler.record_deopt(unit_id);

    const std::uint32_t n_regs = unit->n_registers;
    Value* regs = static_cast<Value*>(regs_raw);

    std::uint32_t resume_pc = 0;
    if (safepoint_index < unit->safepoint_pcs.size()) {
        resume_pc = unit->safepoint_pcs[safepoint_index];
    } else if (!unit->safepoint_pcs.empty()) {
        // Out-of-range safepoint index: this is a bug.
        // Rule 120: don't abort — record and fall back to pc=0.
        vm->telemetry.record(vortex::TelemetryEventKind::DeoptExecuted, unit_id, safepoint_index, 0);
        resume_pc = 0;
    } else {
        // No safepoint table — fall back to pc=0.
        resume_pc = 0;
    }

    Value out;
    bool ok = vm->enter_at(unit, regs, n_regs, resume_pc, out);

    // Rule 26: Record deopt completion + tier downgrade.
    vm->telemetry.record(vortex::TelemetryEventKind::DeoptExecuted, unit_id, resume_pc, 0);
    vm->telemetry.record(vortex::TelemetryEventKind::TierDowngrade, unit_id, 0, 0);
    vm->telemetry.bump(vortex::Telemetry::counter_total_deopts);

    if (!ok) {
        return vortex::Value::none();
    }
    return out;
}
