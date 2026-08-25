// =============================================================================
// vortex/backend/target.cpp — host capability detection (Tier 2).
//
// detect_host_target() starts from the arch's constexpr baseline (the same
// builder the AOT descriptor uses — one source of truth) and overlays REAL
// hardware probes:
//   x86-64:  CPUID leaf 1 (AVX/OSXSAVE), leaf 7 subleaf 0 (AVX2, AVX-512F),
//            leaf 1 EBX[15:8] CLFLUSH line size (x8 = bytes)
//   aarch64: getauxval(AT_HWCAP) on Linux (ASIMD/SVE), CTR_EL0 system
//            register for the D-cache line size (4 << CTR.DminLine)
//
// The result is published through a function-local static (single writer at
// first use; C++11 static-init guarantees every later read observes a fully
// constructed object).
//
// Portability contract: the architecture is selected by compiled_arch() at
// COMPILE time. This file never assigns an architecture by default — an
// undescribed machine cannot reach a working descriptor at all (the header
// static_asserts it out of the build).
// =============================================================================

#include "vortex/backend/target.hpp"

#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64__)
    #define VORTEX_HOST_X86_64 1
    #include <cpuid.h>
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm64__)
    #define VORTEX_HOST_AARCH64 1
    #if defined(__linux__)
        #include <sys/auxv.h>
    #endif
#endif

namespace vortex::backend {
inline namespace abi_v1 {

TargetDescriptor detect_host_target() noexcept {
    // The header's static_assert guarantees exactly these two cases exist;
    // there is no "default to x86" branch anywhere in this file.
    if constexpr (compiled_arch() == Arch::X86_64) {
        TargetDescriptor t = x86_64_baseline();

#if VORTEX_HOST_X86_64
        // Leaf 1: ECX bit 28 = AVX, bit 27 = OSXSAVE (OS saves YMM state).
        // Leaf 7 subleaf 0: EBX bit 5 = AVX2, EBX bit 16 = AVX-512F.
        std::uint32_t eax = 0, ebx = 0, ecx = 0, edx = 0;
        if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
            // CLFLUSH line size: EBX[15:8] * 8 bytes. Zero on some
            // virtualized/emulated CPUs — only trust sane values.
            const std::uint32_t flush_lines = (ebx >> 8) & 0xFFu;
            if (flush_lines != 0) {
                const std::uint32_t line = flush_lines * 8u;
                // accept powers of two in [16, 256]
                if (line >= 16 && line <= 256 && (line & (line - 1)) == 0) {
                    t.cache_line_bytes = line;
                }
            }

            const bool has_avx = (ecx & (1u << 28)) != 0;
            const bool osxsave = (ecx & (1u << 27)) != 0;
            if (has_avx && osxsave) {
                std::uint32_t e7a = 0, e7b = 0, e7c = 0, e7d = 0;
                if (__get_cpuid_count(7, 0, &e7a, &e7b, &e7c, &e7d)) {
                    if (e7b & (1u << 16)) {
                        t.features.set(TargetFeature::AVX512F);
                        t.simd_width_bytes = 64;
                    } else if (e7b & (1u << 5)) {
                        t.features.set(TargetFeature::AVX2);
                        t.simd_width_bytes = 32;
                    }
                    // neither: keep the 128-bit SSE2 baseline
                }
            }
        }
#endif
        return t;
    } else {
        // AArch64 host (compiled_arch() == AArch64 by the header contract).
        TargetDescriptor t = aarch64_baseline();

#if VORTEX_HOST_AARCH64
        // CTR_EL0: bits [19:16] DminLine — D-cache line = 4 << DminLine.
        // EL0-readable on armv8-a Linux and darwin; the probe is a single
        // register read, no syscall.
        std::uint64_t ctr = 0;
        asm volatile("mrs %0, ctr_el0" : "=r"(ctr));
        const std::uint32_t dmin_line = static_cast<std::uint32_t>((ctr >> 16) & 0xFu);
        const std::uint32_t line = 4u << dmin_line;
        if (line >= 16 && line <= 256 && (line & (line - 1)) == 0) {
            t.cache_line_bytes = line;
        }

#if defined(__linux__) && defined(AT_HWCAP)
        // Kernel HWCAP bits (asm/hwcap.h values, inlined to avoid the
        // arch-specific header): ASIMD = 1<<1, SVE = 1<<22.
        const unsigned long hwcap = getauxval(AT_HWCAP);
        if (hwcap & (1UL << 1)) t.features.set(TargetFeature::ASIMD);
        if (hwcap & (1UL << 22)) {
            t.features.set(TargetFeature::SVE);
            t.simd_width_bytes = 32;   // conservative fixed 256-bit SVE claim
        }
#endif
#endif
        return t;
    }
}

const TargetDescriptor& host_target() noexcept {
    static TargetDescriptor cached = detect_host_target();
    return cached;
}

}  // namespace abi_v1
}  // namespace vortex::backend
