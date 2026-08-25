// =============================================================================
// vortex/backend/target.cpp — host capability detection (Tier 2).
//
// CPUID-based x86-64 feature probe executed once; the descriptor is
// published through a function-local static with relaxed publication
// (single-writer at process init; every later read is a plain load of a
// fully-constructed object — safe under the C++11 static-init guarantee).
// =============================================================================

#include "vortex/backend/target.hpp"

#if defined(__x86_64__) || defined(_M_X64)
    #define VORTEX_X86_64 1
    #include <cpuid.h>
#endif

namespace vortex::backend {
inline namespace abi_v1 {

TargetDescriptor detect_host_target() noexcept {
    TargetDescriptor t{};
    t.architecture = Arch::X86_64;
    t.gpr_count = 16;
    t.allocatable_gprs = kX86AllocatableGPRs;
    t.cache_line_bytes = 64;

    // Baseline latencies (reciprocal throughput, conservative).
    t.latencies.fill(1);
    t.latencies[0x28] = 6;    // DIV
    t.latencies[0x2F] = 12;   // IDIV
    t.latencies[0x50] = 3;    // IMUL
    t.latencies[0x8B] = 5;    // MOV load
    t.latencies[0x89] = 1;    // MOV store
    t.latencies[0xE8] = 4;    // CALL
    t.latencies[0xFF] = 4;    // CALL indirect

#if VORTEX_X86_64
    // Leaf 1: ECX bit 28 = AVX, bit 27 = OSXSAVE; leaf 7 subleaf 0:
    // EBX bit 5 = AVX2, EBX bit 16 = AVX-512F.
    std::uint32_t eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        const bool has_avx = (ecx & (1u << 28)) != 0;
        const bool osxsave = (ecx & (1u << 27)) != 0;
        if (has_avx && osxsave) {
            t.simd_width_bytes = 16;   // AVX baseline (128-bit ops)
            std::uint32_t e7a = 0, e7b = 0, e7c = 0, e7d = 0;
            if (__get_cpuid_count(7, 0, &e7a, &e7b, &e7c, &e7d)) {
                if (e7b & (1u << 5)) {
                    t.has_avx2 = true;
                    t.simd_width_bytes = 32;
                }
                if (e7b & (1u << 16)) {
                    t.has_avx512 = true;
                    t.simd_width_bytes = 64;
                }
            }
        }
    }
#else
    t.simd_width_bytes = 16;
#endif
    return t;
}

const TargetDescriptor& host_target() noexcept {
    static TargetDescriptor cached = detect_host_target();
    return cached;
}

}  // namespace abi_v1
}  // namespace vortex::backend
