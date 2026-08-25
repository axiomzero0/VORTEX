// =============================================================================
// vortex/backend/target.hpp — Target Abstraction Layer (Rules 24, 27, 45)
//
// Purpose:
//   All hardware specifics live behind TargetDescriptor — never #ifdef in
//   pass or backend logic (Rule 24). The descriptor carries the register
//   partition (allocatable set + frame-protocol roles), SIMD width, cache
//   line size, and a per-cost-class latency table feeding every cost
//   decision (Rule 45).
//
//   - Tier 3 (AOT): aot_target() is consteval-baked from compiled_arch() —
//     the architecture the compiler itself was built for. There is no
//     "default x86": a port to a new machine that forgets to describe its
//     registers fails the static_assert below at build time.
//   - Tier 2 (JIT): detect_host_target() selects at process start via real
//     hardware probes (CPUID on x86-64, getauxval/CTR_EL0 on AArch64) — a
//     single cached read, no per-query cost.
//
// Where machine facts are allowed to live (and ONLY there):
//   1. compiled_arch()      — the ONE preprocessor dispatch site.
//   2. namespace x86 / aarch64 — ISA encoding + ABI facts per architecture.
//   3. x86_64_baseline()/aarch64_baseline() — the shared descriptor builders
//      used by BOTH the consteval AOT descriptor and the runtime detector,
//      so the two can never drift apart.
//
// Invariants:
//   - No virtual functions: the descriptor is a value type; cost queries
//     are constexpr-friendly and inlinable.
//   - Latencies are indexed by CostClass (arch-neutral), never by machine
//     opcode bytes — the x86 encoding belongs to the x86 emitter alone.
//   - SIMD width is queried, never assumed (Rule 27): SLP packetization
//     (pass 31) reads target->simd_width_bytes; nothing in cfg hardcodes
//     lane counts.
// =============================================================================

#pragma once

#include <array>
#include <cstdint>

#include "vortex/support/config.hpp"
#include "vortex/support/flags.hpp"

namespace vortex::backend {

inline namespace abi_v1 {

enum class Arch : std::uint8_t {
    Unknown = 0,
    X86_64,
    AArch64,
};

/// The architecture this compiler was BUILT for (Tier 3 AOT target). This is
/// the single preprocessor dispatch point in the entire codebase (Rule 24);
/// every other component asks the descriptor.
consteval Arch compiled_arch() noexcept {
#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64__)
    return Arch::X86_64;
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm64__)
    return Arch::AArch64;
#else
    return Arch::Unknown;
#endif
}

/// Portability contract: building on an undescribed machine is a BUILD
/// failure with instructions — never a silent x86-64-shaped guess.
static_assert(compiled_arch() != Arch::Unknown,
              "VORTEX: no TargetDescriptor for this architecture. "
              "Add register facts + a baseline builder in "
              "vortex/backend/target.hpp and a probe in target.cpp.");

/// Register classes — every MIR def belongs to exactly one.
enum class MachineRegClass : std::uint8_t {
    GPR = 0,   // general purpose (int64 payloads, pointers)
    FPR,       // scalar float (xmm low 64 bits / d-register)
    VEC,       // SIMD vector (full vector width)
};

/// Callee-saved register roles of the VORTEX frame protocol. Arch-neutral:
/// the ROLE is fixed by the protocol, the ENCODING is per-arch data in the
/// descriptor (x86 reserves r12-r15, aarch64 reserves x24-x27).
enum class ReservedGPR : std::uint8_t {
    FrameBase = 0,    // Tier-0 register file base (Value* regs)
    ConstPool = 1,    // constant pool base
    VMContext = 2,    // runtime context pointer
    DeoptCtx = 3,     // active deopt metadata pointer
    Count = 4,
};

/// Arch-neutral operation cost classes (Rule 45). The latency model is
/// per-architecture data keyed by this enum — never by machine opcodes.
enum class CostClass : std::uint8_t {
    Move = 0,   // register move
    Alu,        // add/sub/logic
    Mul,        // integer multiply
    Div,        // integer divide
    Load,       // L1-dependent load
    Store,      // store
    Branch,     // predicted taken/not-taken branch
    Call,       // call+ret pair
    VecAlu,     // SIMD arithmetic lane count 1
    Count,
};

/// Hardware feature bits — probed at runtime (Tier 2) or derived from the
/// ISA baseline (Tier 3). No pass reads raw booleans.
enum class TargetFeature : std::uint16_t {
    AVX2 = 1 << 0,      // x86-64: 256-bit integer SIMD
    AVX512F = 1 << 1,   // x86-64: 512-bit foundation
    ASIMD = 1 << 2,     // aarch64: NEON 128-bit (baseline on armv8-a)
    SVE = 1 << 3,       // aarch64: scalable vector extension
    LSE = 1 << 4,       // aarch64: atomics (LSE) — matters for refcount ops
};

// =============================================================================
// Per-architecture ISA + ABI facts. These namespaces are the ONLY place raw
// register encodings and opcode-shaped constants may appear outside the
// emitters.
// =============================================================================

namespace x86 {

// GPR encodings (Intel order). SysV x86-64 facts.
inline constexpr std::uint8_t RAX = 0, RCX = 1, RDX = 2, RBX = 3;
inline constexpr std::uint8_t RSP = 4, RBP = 5, RSI = 6, RDI = 7;
inline constexpr std::uint8_t R8 = 8, R9 = 9, R10 = 10, R11 = 11;
inline constexpr std::uint8_t R12 = 12, R13 = 13, R14 = 14, R15 = 15;

inline constexpr std::uint32_t kGPRCount = 16;

// Frame-protocol role assignment (all callee-saved under SysV).
inline constexpr std::uint8_t kRoleReg[enum_size(ReservedGPR::Count)] = {
    R12,   // FrameBase
    R13,   // ConstPool
    R14,   // VMContext
    R15,   // DeoptCtx
};

// Allocatable set: 16 - RSP (stack) - RBP (frame) - 4 roles = 10.
// Order IS allocation priority: argument/temp registers first, RBX (the
// only additional callee-saved) last.
inline constexpr std::uint32_t kAllocatableCount = 10;
inline constexpr std::uint8_t kAllocatable[kAllocatableCount] = {
    RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11, RBX,
};

// ISA floor of the x86-64 baseline: SSE2 128-bit SIMD is architectural.
inline constexpr std::uint32_t kBaselineSimdBytes = 16;

}  // namespace x86

namespace aarch64 {

// GPR encodings: x0..x30 (x31 doubles as SP/ZR).
inline constexpr std::uint32_t kGPRCount = 31;   // x0..x30, SP excluded

inline constexpr std::uint8_t kPlatform = 18;    // x18: platform register (PAC/SCS)
inline constexpr std::uint8_t kFramePointer = 29; // x29 (FP)
inline constexpr std::uint8_t kLinkRegister = 30; // x30 (LR)

// Frame-protocol role assignment (callee-saved pool x19-x28).
inline constexpr std::uint8_t kRoleReg[enum_size(ReservedGPR::Count)] = {
    27,   // FrameBase  (x27)
    26,   // ConstPool  (x26)
    25,   // VMContext  (x25)
    24,   // DeoptCtx   (x24)
};

// Allocatable set: x0-x15 (argument/temp), x19-x23 + x28 (callee-saved
// beyond the roles), then x16/x17 (IP0/IP1 — veneer registers, usable
// inside JIT code but lowest priority so they stay free for trampolines).
// Excluded: SP, x18 (platform), x29/x30, the four roles above.
inline constexpr std::uint32_t kAllocatableCount = 24;
inline constexpr std::uint8_t kAllocatable[kAllocatableCount] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    19, 20, 21, 22, 23, 28,   // callee-saved surplus
    16, 17,                   // IP0/IP1 last
};

// ISA floor of armv8-a: ASIMD 128-bit is mandatory.
inline constexpr std::uint32_t kBaselineSimdBytes = 16;

}  // namespace aarch64

/// Upper bound on the allocatable set across all supported architectures —
/// sizes the descriptor's inline register table.
inline constexpr std::uint32_t kMaxAllocatable = 32;

// =============================================================================
// TargetDescriptor
// =============================================================================

/// Target capability + cost descriptor (Rules 27, 45). A default-constructed
/// descriptor is deliberately useless (Unknown arch, zero registers): the
/// only valid producers are the baseline builders below and
/// detect_host_target(). Consumers must treat Unknown as "do not compile".
struct TargetDescriptor {
    Arch architecture{Arch::Unknown};
    std::uint32_t gpr_count{0};

    /// Allocatable GPR encodings; valid indices are [0, allocatable_gprs).
    /// Index order IS allocation priority. The allocator and both emitters
    /// read THIS table — never a per-arch global.
    std::uint32_t allocatable_gprs{0};
    std::uint8_t allocatable[kMaxAllocatable]{};

    /// Frame-protocol role encodings, indexed by ReservedGPR.
    std::uint8_t reserved[enum_size(ReservedGPR::Count)]{};

    std::uint32_t simd_width_bytes{0};   // 16 SSE2/ASIMD, 32 AVX2/SVE, 64 AVX-512
    std::uint32_t cache_line_bytes{0};

    Flags<TargetFeature> features{};

    /// Latencies (reciprocal-throughput model, conservative) indexed by
    /// CostClass. Consumers branch on these for cost decisions, never on
    /// hardcoded numbers (Rule 23/61).
    std::array<std::uint8_t, enum_size(CostClass::Count)> latencies{};

    [[nodiscard]] constexpr std::uint32_t latency(CostClass c) const noexcept {
        [[assume(static_cast<std::size_t>(c) < latencies.size())]];
        return latencies[static_cast<std::size_t>(c)];
    }

    [[nodiscard]] constexpr bool has(TargetFeature f) const noexcept {
        return features.has(f);
    }

    /// Rule 45 cost hook: does moving `lanes` 64-bit scalars into the
    /// vector file pay? Cross-file moves cost ~1 cycle per lane on both
    /// x86 and ARM.
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

// =============================================================================
// Baseline builders — the single source of truth shared by the consteval
// AOT descriptor and the runtime host probe, so the two can never drift.
// =============================================================================

namespace detail {

/// Cache line size without runtime probes, resolved in priority order:
///   1. VORTEX_CACHE_LINE_BYTES  — CMake configure-time override
///   2. the compiler's destructive-interference constant (query, not guess)
///   3. cfg::cache_line_fallback_bytes — documented last resort
consteval std::uint32_t unprobed_cache_line() noexcept {
#if defined(VORTEX_CACHE_LINE_BYTES)
    return VORTEX_CACHE_LINE_BYTES;
#elif defined(__GCC_DESTRUCTIVE_SIZE)
    return __GCC_DESTRUCTIVE_SIZE;
#elif defined(__APPLE__) && (defined(__aarch64__) || defined(_M_ARM64))
    // Apple Silicon packs pairs of 64-byte lines; the ABI-level destructive
    // interference span is 128.
    return 128;
#else
    return cfg::cache_line_fallback_bytes;
#endif
}

}  // namespace detail

/// x86-64 ISA baseline (SSE2 floor). Runtime detection overlays CPUID caps.
constexpr TargetDescriptor x86_64_baseline() noexcept {
    TargetDescriptor t{};
    t.architecture = Arch::X86_64;
    t.gpr_count = x86::kGPRCount;
    t.allocatable_gprs = x86::kAllocatableCount;
    for (std::uint32_t i = 0; i < x86::kAllocatableCount; ++i) {
        t.allocatable[i] = x86::kAllocatable[i];
    }
    for (std::uint32_t r = 0; r < enum_size(ReservedGPR::Count); ++r) {
        t.reserved[r] = x86::kRoleReg[r];
    }
    t.simd_width_bytes = x86::kBaselineSimdBytes;
    t.cache_line_bytes = detail::unprobed_cache_line();

    // Reciprocal throughput, conservative SnB-family model.
    // Order: Move, Alu, Mul, Div, Load, Store, Branch, Call, VecAlu.
    t.latencies = {{1, 1, 3, 6, 5, 1, 1, 4, 1}};
    return t;
}

/// AArch64 ISA baseline (armv8-a: ASIMD floor). Runtime detection overlays
/// HWCAP/CTR_EL0 caps.
constexpr TargetDescriptor aarch64_baseline() noexcept {
    TargetDescriptor t{};
    t.architecture = Arch::AArch64;
    t.gpr_count = aarch64::kGPRCount;
    t.allocatable_gprs = aarch64::kAllocatableCount;
    for (std::uint32_t i = 0; i < aarch64::kAllocatableCount; ++i) {
        t.allocatable[i] = aarch64::kAllocatable[i];
    }
    for (std::uint32_t r = 0; r < enum_size(ReservedGPR::Count); ++r) {
        t.reserved[r] = aarch64::kRoleReg[r];
    }
    t.simd_width_bytes = aarch64::kBaselineSimdBytes;
    t.cache_line_bytes = detail::unprobed_cache_line();
    t.features.set(TargetFeature::ASIMD);

    // Reciprocal throughput, conservative Cortex-A model.
    // Order: Move, Alu, Mul, Div, Load, Store, Branch, Call, VecAlu.
    t.latencies = {{1, 1, 3, 6, 4, 1, 1, 4, 2}};
    return t;
}

/// Tier 3 AOT descriptor: baked at compile time from the architecture the
/// compiler itself targets. Baseline features only — AOT binaries must run
/// on every machine of the family.
consteval TargetDescriptor aot_target() noexcept {
    if constexpr (compiled_arch() == Arch::X86_64) {
        return x86_64_baseline();
    } else {
        return aarch64_baseline();
    }
}

/// Runtime hardware-probe descriptor (Tier 2): baseline overlaid with real
/// CPUID / getauxval / CTR_EL0 queries for SIMD width and cache line size.
/// Executed once; published through host_target().
[[nodiscard]] TargetDescriptor detect_host_target() noexcept;

/// Access the process-wide host descriptor (detect_host_target, cached).
const TargetDescriptor& host_target() noexcept;

}  // namespace abi_v1
}  // namespace vortex::backend
