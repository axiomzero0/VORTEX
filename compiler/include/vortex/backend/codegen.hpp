// =============================================================================
// vortex/backend/codegen.hpp — Pass 54/55: emission, layout, deopt metadata
//
// Translates allocated MIR into executable machine code:
//   - Prologue: push callee-saved, materialize r12 = regs base.
//   - Peephole (Pass 54): MOVri + ALUrr fuse to ALU with imm32 where the
//     constant fits; MOVrr pairs collapse; redundant stores to the same
//     home slot within a block drop.
//   - Hot/cold partitioning (Pass 55): cold blocks (deopt traps, handler
//     bodies) emit into a secondary region appended after the hot body;
//     intra-unit jumps link them by rel32 patch.
//   - Safepoints: every call, backedge, and guard records a
//     SafepointRecord (pc offset, frame state, live physreg->slot map) into
//     the unit's unwind table — the .vortex_unwind section.
//   - Deopt traps: GUARD failures jump to a cold stub that calls
//     vortex_deopt_entry(unit_id, safepoint_index); the runtime rebuilds
//     the Tier-0 frame and resumes via Vm::enter_at.
// =============================================================================

#pragma once

#include <cstdint>

#include "vortex/backend/assembler.hpp"
#include "vortex/backend/lowering.hpp"
#include "vortex/backend/mir.hpp"
#include "vortex/backend/regalloc.hpp"
#include "vortex/common/value.hpp"

namespace vortex::backend {

inline namespace abi_v1 {

/// Runtime helper functions the codegen calls (resolved by jit.cpp).
struct HelperTable {
    void (*binop)(void* ctx, std::uint64_t op, std::uint64_t a, std::uint64_t b) = nullptr;
    void* ctx{nullptr};
};

/// Full backend output for one unit.
struct CompiledCode {
    std::byte* code{nullptr};
    std::size_t code_size{0};
    std::size_t cold_offset{0};   // hot region = [0, cold_offset)
    stdx::small_vector<SafepointRecord, 16> safepoints{};
    stdx::small_vector<SafepointMapping, 64> mappings{};   // concat per record
    std::uint32_t frame_slots{0};
    std::uint32_t unit_id{0};
    /// Pass 54: count of peephole fusions applied during emission. Telemetry
    /// hook (Rule 26): zero in a freshly-lowered unit with no ALU+const
    /// patterns; positive when MOVri+ALUrr fused to ALU r64, imm32.
    std::uint32_t peephole_fusions{0};
    /// Pass 54: count of operand reads served from the GPR cache instead
    /// of from the home slot. Zero under the legacy ALWAYS-SPILL codegen
    /// path; positive when the regalloc-aware resolve finds the operand's
    /// assigned GPR still holding its value. Each hit saves a memory load
    /// (4-7 bytes, 3-5 cycles) and replaces it with a reg-to-reg move
    /// (3 bytes, 1 cycle).
    std::uint32_t gpr_cache_hits{0};
    /// Pass 54 V2: count of stage_rax / stage_rcx calls where the
    /// operand was already in the staging register (cache hit on RAX for
    /// stage_rax, on RCX for stage_rcx). The naive path emits a 3-byte
    /// `mov reg, reg` self-move (which is a no-op at the ISA level but
    /// still occupies the decode stream and 3 bytes of i-cache). The
    /// cache-aware path skips the mov entirely AND avoids the cache
    /// clobber, so subsequent resolves in the same op can still hit
    /// the cache for OTHER vregs that happen to live in the staging
    /// register's slot. Each elimination saves 3 bytes and one decode
    /// slot.
    std::uint32_t self_mov_eliminations{0};
    /// Pass 54 V3 (LSRA->XMM): count of operand reads served from the
    /// XMM cache instead of from the home slot. Mirrors `gpr_cache_hits`
    /// but for FPR-class vregs. Zero under the legacy write-through-home
    /// scheme (no FPR allocation, every FPR op reloads from home);
    /// positive when the regalloc-aware resolve finds the operand's
    /// assigned XMM still holding its value. Each hit replaces a
    /// `movsd xmm, [r12 + slot*16 + 8]` (5-8 bytes, 4-6 cycles L1
    /// dependent) with a `movsd xmm_dst, xmm_src` (4 bytes, 1 cycle).
    /// The hot path this fires on is the float-loop pattern: any tight
    /// loop with FADD/FSUB/FMUL/FDIV chain (e.g., `s += x[i] * y[i]`
    /// over an array) benefits the most — the loop-carried `s` lives
    /// in an XMM for the whole loop body, reloaded from home only at
    /// the loop boundary.
    std::uint32_t xmm_cache_hits{0};
    bool valid{false};
};

/// Entry-point signature the runtime calls. SysV return convention for a
/// 16-byte POD: tag word in RAX, payload in RDX — exactly what the JIT's
/// RET path loads from home slot 0. Calling convention:
///   RDI = Value* regs   (the Tier-0 register file — home slots)
///   RAX (return) = result tag word
///   RDX (return) = result payload
/// The runtime transfers ownership of the regs array to the JIT; the JIT
/// transfers it back via the deopt/bridge path if it can't complete.
using JitEntryFn = vortex::Value (*)(void* regs);

/// Pass 54/55 driver: lower -> allocate -> emit.
[[nodiscard]] CompiledCode compile_unit(const ir::Graph& g, std::uint32_t unit_id,
                                        std::byte* buffer, std::size_t capacity,
                                        const TargetDescriptor& target) noexcept;

}  // namespace abi_v1
}  // namespace vortex::backend
