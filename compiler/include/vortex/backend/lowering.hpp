// =============================================================================
// vortex/backend/lowering.hpp — Pass 52: Machine Instruction Selection
//
// Table-driven lowering of the optimized Sea of Nodes into Machine IR.
// (NodeKind x TypeClass) -> MOp mapping built once at compile time; the
// walk is O(N) with zero allocations beyond the MIR arena.
//
// ABI contract with the Tier-0 runtime (SysV x86-64):
//   r12 = Value* regs   (the Tier-0 register file — home slots)
//   r13 = const pool base
//   r14 = runtime context
//   r15 = reserved (deopt metadata)
//   Entry: (rdi = regs). Return: Value in RAX (tag word) + RDX (payload).
//
// Unboxing rule: PyBinary/PyCompare on operands whose IR nodes carry the
// Unboxed marker (pass 47) or are int constants lower to native GPR ops
// with an entry guard per operand (tag == Int); the guard's failure path
// is a deopt trap carrying the FrameState — Rules 3/4/5.
// =============================================================================

#pragma once

#include "vortex/backend/mir.hpp"
#include "vortex/ir/graph.hpp"

namespace vortex::backend {

inline namespace abi_v1 {

struct LoweringResult {
    MachineGraph mir{};
    /// Blocks in emission order (RPO, cold blocks last).
    stdx::small_vector<std::uint32_t, 16> block_order{};
    /// Frame slots the function touches (max Tier-0 reg index + 1).
    std::uint32_t frame_slots{0};
    /// FrameState ids referenced by guards (for the unwind table).
    stdx::small_vector<std::uint32_t, 8> referenced_frame_states{};
    /// Task 24: true if any CALLri fallback (PyBinary/PyCompare/CallPy/
    /// LoadGlobal/StoreGlobal/LoadAttr/StoreAttr/LoadIndex/StoreIndex/
    /// ListAppend/Iter/GetIterCheck/IterNext/Yield/NewList/NewTuple/
    /// NewDict/Guard-with-no-frame-state) was emitted. Copied into
    /// CompiledCode by the codegen; propagated to CodeUnit by the
    /// runtime driver. The runtime uses this to skip jit_entry for
    /// functions that would require the bridge path (which needs a
    /// populated safepoint_pcs table — not yet implemented).
    bool has_dynamic_ops{false};
};

/// Pass 52 entry point. `target` supplies the cost/SIMD capabilities.
[[nodiscard]] LoweringResult lower_to_mir(const ir::Graph& g,
                                          const TargetDescriptor& target) noexcept;

}  // namespace abi_v1
}  // namespace vortex::backend
