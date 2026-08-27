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
//
// Task 24: vortex_rt_munmap_jit_buffer — extern "C" shim for munmap,
// used by the CodeUnit destructor to free the RWX JIT buffer without
// pulling <sys/mman.h> into every TU that includes code.hpp.
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
        // Task 24: no safepoint_pcs populated. This indicates the
        // driver installed jit_entry for a unit with has_dynamic_ops ==
        // false (no CALLri fallback), but the JIT'd code STILL emitted
        // a CALLri somewhere. That's a backend bug — the has_dynamic_ops
        // flag should have been set. Abort to surface the bug clearly
        // rather than silently returning wrong values.
        std::fputs("VORTEX jit bridge: no safepoint table — backend bug? "
                   "(has_dynamic_ops should have gated jit_entry off)\n",
                   stderr);
        std::abort();
    }

    Value out;
    bool ok = vm->enter_at(unit, regs, n_regs, resume_pc, out);
    if (!ok) {
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
