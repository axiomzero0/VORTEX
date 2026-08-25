// =============================================================================
// vortex/rt/jit.cpp — Tier 1/2 JIT runtime integration.
//
// vortex_jit_bridge(regs, unit_id, op_hint):
//   Called when the JIT hits a dynamic op it couldn't lower (CALLri).
//   `op_hint` is the safepoint_index of the dynamic-op site — the
//   runtime translates it to a Tier-0 PC via unit->safepoint_pcs and
//   resumes interpretation at that point. The Tier-0 interpreter runs
//   to completion and returns the same Value the JIT would have
//   produced.
//
// Returns a Value (not void) — the JIT's calling convention expects
// RAX/RDX to carry the result tag/payload. The previous void return
// broke the Value-return contract.
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
}  // namespace vortex::rt;

extern "C" vortex::Value vortex_jit_bridge(void* regs_raw, std::uint32_t unit_id,
                                           std::uint64_t op_hint) noexcept {
    using namespace vortex::rt;
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

    // op_hint is the safepoint_index of the dynamic-op site. Translate
    // it to the Tier-0 PC where the op should resume. Without this,
    // pc=0 restart double-executes effects for non-idempotent ops
    // (the bug the audit flagged).
    std::uint32_t resume_pc = 0;
    const std::uint32_t safepoint_index =
        static_cast<std::uint32_t>(op_hint);
    if (safepoint_index < unit->safepoint_pcs.size()) {
        resume_pc = unit->safepoint_pcs[safepoint_index];
    } else if (!unit->safepoint_pcs.empty()) {
        std::fprintf(stderr, "VORTEX jit bridge: op_hint %llu out of range "
                             "(size %zu)\n",
                     static_cast<unsigned long long>(op_hint),
                     unit->safepoint_pcs.size());
        std::abort();
    } else {
        std::fputs("VORTEX jit bridge: no safepoint table — falling back to "
                   "pc=0\n", stderr);
    }

    Value out;
    bool ok = vm->enter_at(unit, regs, n_regs, resume_pc, out);
    if (!ok) {
        return vortex::Value::none();
    }
    return out;
}
