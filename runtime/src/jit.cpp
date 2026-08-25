// =============================================================================
// vortex/rt/jit.cpp — Tier 1/2 JIT runtime integration.
//
// vortex_jit_bridge(regs, unit_id, op_hint):
//   Called when the JIT hits a dynamic op it couldn't lower (CALLri).
//   Transitions execution to Tier-0 at PC 0 via Vm::enter_at.
//
// vortex_deopt_entry is in deopt.cpp — same transition, but invoked on
// guard failure rather than dynamic-op fallback.
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

extern "C" void vortex_jit_bridge(void* regs_raw, std::uint32_t unit_id,
                                  std::uint64_t op_hint) noexcept {
    using namespace vortex::rt;
    (void)op_hint;
    CodeUnit* unit = find_unit(unit_id);
    if (!unit) {
        std::fputs("VORTEX jit bridge: unknown unit id\n", stderr);
        std::abort();
    }
    Vm* vm = active_vm();
    if (!vm) {
        std::fputs("VORTEX jit bridge: no active VM\n", stderr);
        std::abort();
    }
    if (!regs_raw) {
        std::fputs("VORTEX jit bridge: null regs\n", stderr);
        std::abort();
    }

    Value* regs = static_cast<Value*>(regs_raw);
    const std::uint32_t n_regs = unit->n_registers;
    Value out;
    bool ok = vm->enter_at(unit, regs, n_regs, /*pc=*/0, out);
    if (!ok) {
        return;
    }
}
