// =============================================================================
// vortex/backend/regalloc.hpp — Pass 53: Linear Scan + Interval Splitting
//
// Priority-based linear scan over the MIR (graph coloring is far too slow
// for sub-ms Tier 2 compiles). Upgrades over textbook LSRA:
//   - Live intervals split at loop boundaries: a value crossing a loop
//     header gets a fresh interval per iteration region, so the inner
//     region's allocation isn't poisoned by a long outer lifetime.
//   - Priority = loop depth weighting: hot-path values claim physical
//     registers first; cold tails spill without evicting loop-carried
//     values.
//   - PyObject-class spills insert INCREF at the spill point and DECREF at
//     reload (the reference is owned by the stack slot while cached in a
//     register it is the register's transient copy).
//
// The physical register file is TargetDescriptor data (allocatable set +
// frame-protocol roles): 10 allocatable GPRs on x86-64 (r12-r15 reserved by
// the frame protocol), 24 on AArch64 (x24-x27, x18, FP/LR excluded). Every
// vreg's spill home is its Tier-0 frame slot, so spill code is a plain MOV
// off the frame-base register — and the deoptimizer can reconstruct state
// from safepoint (physreg -> slot) maps alone.
// =============================================================================

#pragma once

#include "vortex/backend/mir.hpp"

namespace vortex::backend {

inline namespace abi_v1 {

struct RegAllocResult {
    /// vreg -> physical GPR encoding byte, or -1 when spilled (or when the
    /// vreg is FPR-class and was assigned an XMM via `assignment_fpr`).
    stdx::small_vector<std::int32_t, 128> assignment{};
    /// vreg -> physical FPR encoding index into target.allocatable_fpr[],
    /// or -1 when spilled (or when the vreg is GPR-class and was assigned
    /// a GPR via `assignment`). Symmetric to `assignment` — a vreg has
    /// EITHER a GPR (GPR-class vregs) OR an FPR (FPR-class vregs) OR
    /// neither (spilled). The two arrays never disagree on the same vreg:
    /// if assignment[v] >= 0 then assignment_fpr[v] == -1, and vice versa.
    /// A zero-length `assignment_fpr` (the legacy contract before the
    /// LSRA->XMM extension) means "FPR allocation is disabled" — FPR-class
    /// vregs read/write through home (the ALWAYS-SPILL discipline for FPRs).
    stdx::small_vector<std::int32_t, 128> assignment_fpr{};
    /// Spill/reload code is encoded as MIR nodes appended after allocation:
    /// the codegen consumes `spill_ops` in position order.
    std::uint32_t spills{0};
    std::uint32_t splits{0};
    std::uint32_t max_pressure{0};
};

/// Pass 53 entry point. Mutates nothing in the MIR; returns the assignment
/// plus an interval list for the safepoint mapper.
[[nodiscard]] RegAllocResult linear_scan(MachineGraph& mir,
                                         const stdx::small_vector<std::uint32_t, 16>& order,
                                         const TargetDescriptor& target,
                                         stdx::small_vector<LiveInterval, 64>& intervals_out) noexcept;

}  // namespace abi_v1
}  // namespace vortex::backend
