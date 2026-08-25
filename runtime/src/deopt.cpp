// =============================================================================
// vortex/rt/deopt.cpp — Deoptimization engine (Rule 4).
//
// Real implementation: when a JIT guard fails, the cold stub tail-calls
// vortex_deopt_entry with (unit_id, safepoint_index, regs_base). The
// deoptimizer:
//   1. Looks up the active Vm via the global builtins pointer.
//   2. Finds the CodeUnit by id in vm.program.units.
//   3. Reads the safepoint record (mostly for diagnostics — the codegen's
//      home-slot write-back discipline makes the home slots authoritative
//      at every safepoint, so the live physreg->slot map is informational).
//   4. Transfers ownership of the regs array into a new Tier-0 Frame and
//      resumes via Vm::enter_at at PC 0. The Tier-0 interpreter runs to
//      completion, writes the result to vm.frame_return_, and the JIT's
//      caller observes the same Value it would have observed if the unit
//      had run entirely in Tier-0.
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

extern "C" void vortex_deopt_entry(std::uint32_t unit_id, std::uint32_t safepoint_index,
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
    (void)safepoint_index;

    Value* regs = static_cast<Value*>(regs_raw);
    Value out;
    bool ok = vm->enter_at(unit, regs, n_regs, /*pc=*/0, out);
    if (!ok) {
        return;
    }
}
