// =============================================================================
// vortex/backend/target.hpp — Target Abstraction Layer (Rules 24, 27, 45)
//
// Purpose:
//   All hardware specifics live behind TargetDescriptor — never #ifdef in
//   pass or backend logic (Rule 24). The descriptor carries register
//   counts, SIMD width, cache line size, and a per-opcode latency table
//   feeding every cost decision (Rule 45).
//
//   - Tier 3 (AOT): the descriptor is consteval-baked at build time.
//   - Tier 2 (JIT): the descriptor is selected at process start via CPUID
//     feature detection — a single atomic-capability load, no per-query
//     cost.
//
// Invariants:
//   - No virtual functions: the descriptor is a value type; cost queries
//     are constexpr-friendly and inlinable.
//   - Latency table indexed by MachineOpcode byte — bounds elided via
//     [[assume]] at call sites (Rule 21).
//   - SIMD width is queried, never assumed (Rule 27): SLP packetization
//     (pass 31) reads simd_width_bytes instead of hardcoding 4/8 lanes.
// =============================================================================

#pragma once

#include <array>
#include <cstdint>

namespace vortex::backend {

inline namespace abi_v1 {

enum class Arch : std::uint8_t {
    Unknown = 0,
    X86_64,
    AArch64,
};

/// Register classes — every MIR def belongs to exactly one.
enum class MachineRegClass : std::uint8_t {
    GPR = 0,   // general purpose (int64 payloads, pointers)
    FPR,       // scalar float (xmm low 64 bits)
    VEC,       // SIMD vector (full xmm width)
};

/// Callee-saved register roles the codegen reserves for its own frame
/// protocol (SysV x86-64). The allocator never assigns these.
enum class ReservedGPR : std::uint8_t {
    FrameBase = 0,    // r12: Tier-0 register file base (Value* regs)
    ConstPool = 1,    // r13: constant pool base
    VMContext = 2,    // r14: runtime context pointer
    DeoptCtx = 3,     // r15: active deopt metadata pointer
    Count = 4,
};

/// x86-64 physical GPR encoding (SysV order). The allocatable set excludes
/// RSP/RBP and the four reserved roles above.
struct PhysGPR {
    static constexpr std::uint8_t RAX = 0, RCX = 1, RDX = 2, RBX = 3;
    static constexpr std::uint8_t RSP = 4, RBP = 5, RSI = 6, RDI = 7;
    static constexpr std::uint8_t R8 = 8, R9 = 9, R10 = 10, R11 = 11;
    static constexpr std::uint8_t R12 = 12, R13 = 13, R14 = 14, R15 = 15;
};

/// Number of allocatable x86-64 GPRs under the VORTEX frame protocol:
/// 16 total - RSP - RBP - 4 reserved = 10.
inline constexpr std::uint32_t kX86AllocatableGPRs = 10;

/// The allocatable GPR list (encoding order); index == allocation slot.
inline constexpr std::uint8_t kAllocatableGPR[10] = {
    PhysGPR::RAX, PhysGPR::RCX, PhysGPR::RDX, PhysGPR::RSI, PhysGPR::RDI,
    PhysGPR::R8,  PhysGPR::R9,  PhysGPR::R10, PhysGPR::R11, PhysGPR::RBX,
};

/// Target capability + cost descriptor (Rules 27, 45).
struct TargetDescriptor {
    Arch architecture{Arch::Unknown};
    std::uint32_t gpr_count{16};
    std::uint32_t allocatable_gprs{kX86AllocatableGPRs};
    std::uint32_t simd_width_bytes{16};   // 16 SSE, 32 AVX2, 64 AVX-512
    std::uint32_t cache_line_bytes{64};
    bool has_avx2{false};
    bool has_avx512{false};
    /// Latencies indexed by MachineOpcode & 0xFF. Conservative SnB-family
    /// values; consumers branch on these for cost decisions, never on
    /// hardcoded numbers (Rule 23/61).
    std::array<std::uint8_t, 256> latencies{};

    [[nodiscard]] constexpr std::uint32_t latency(std::uint8_t machine_op) const noexcept {
        return latencies[machine_op];
    }

    /// Rule 45 cost hook: does moving `lanes` scalars into the vector file
    /// pay? Cross-file moves cost ~1 cycle per lane on both x86 and ARM.
    [[nodiscard]] constexpr bool vector_pays(std::uint32_t lanes,
                                             std::uint32_t scalar_ops) const noexcept {
        if (simd_width_bytes < 16) return false;
        const std::uint32_t vec_lanes = simd_width_bytes / 8;
        if (lanes > vec_lanes) return false;
        const std::uint32_t move_cost = lanes;              // insert/extract
        const std::uint32_t scalar_cost = scalar_ops * lanes;
        return scalar_cost > move_cost + lanes;             // vector wins
    }
};

/// Runtime CPUID-based descriptor selection (Tier 2). Executed once at
/// process start; the result is published through a relaxed-atomic global
/// so the JIT thread and mutators read identical capabilities.
[[nodiscard]] TargetDescriptor detect_host_target() noexcept;

/// Access the process-wide host descriptor (set by detect_host_target()).
const TargetDescriptor& host_target() noexcept;

/// Tier 3 AOT descriptor: fixed at compile time via TARGET_ARCH.
consteval TargetDescriptor aot_target() noexcept {
    TargetDescriptor t{};
    t.architecture = Arch::X86_64;
    t.gpr_count = 16;
    t.allocatable_gprs = kX86AllocatableGPRs;
    t.simd_width_bytes = 32;   // AVX2 baseline for AOT builds
    t.cache_line_bytes = 64;
    t.has_avx2 = true;
    t.has_avx512 = false;
    // Dominant latencies (reciprocal-throughput model, SnB family):
    t.latencies.fill(1);
    t.latencies[0x28] = 6;   // DIV
    t.latencies[0x2F] = 12;  // IDIV-class
    t.latencies[0x50] = 3;   // IMUL r64
    t.latencies[0x8B] = 5;   // MOV load (L1)
    t.latencies[0x89] = 1;   // MOV store
    t.latencies[0xE8] = 4;   // CALL direct
    t.latencies[0xFF] = 4;   // CALL indirect
    t.latencies[0xC5] = 1;   // VEX ADD
    return t;
}

}  // namespace abi_v1
}  // namespace vortex::backend
