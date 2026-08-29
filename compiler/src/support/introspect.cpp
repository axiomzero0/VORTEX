// =============================================================================
// vortex/support/introspect.cpp — Introspector implementation
//
// Global state for the phase hook. Deliberately simple: one function pointer,
// set once at startup. No mutex (Rule 118: no locks on hot paths). The
// pointer is read by PhaseScope's constructor — a single branch on armed_.
// =============================================================================

#include "vortex/support/introspect.hpp"

namespace vortex::support {

inline namespace abi_v1 {

namespace {
PhaseHook g_phase_hook{nullptr};
}

void set_phase_hook(PhaseHook hook) noexcept {
    g_phase_hook = hook;
}

PhaseHook get_phase_hook() noexcept {
    return g_phase_hook;
}

}  // namespace abi_v1
}  // namespace vortex::support
