// =============================================================================
// vortex/support/assume.hpp — Compiler invariants & unreachable (Rules 21, 58)
//
// Purpose:
//   - VORTEX_ASSUME(c): expands to C++26 [[assume(c)]] — documents invariants
//     so the compiler elides bounds checks with zero runtime cost.
//   - VORTEX_UNREACHABLE(): for switches on closed enums that were made
//     exhaustive; emits __builtin_unreachable in release, aborts in debug
//     (Rule 58 forbids silent fallthrough; we fail loudly when violated).
// =============================================================================

#pragma once

#include <cstdio>
#include <cstdlib>

#if VORTEX_DEBUG
    #define VORTEX_ASSUME(cond)                                                \
        do {                                                                   \
            if (!(cond)) [[unlikely]] {                                        \
                std::fprintf(stderr,                                           \
                             "VORTEX FATAL: invariant violated at %s:%d: %s\n",\
                             __FILE__, __LINE__, #cond);                       \
                std::abort();                                                  \
            }                                                                  \
        } while (0)
#else
    #define VORTEX_ASSUME(cond) [[assume(cond)]]
#endif

// Rule 58: non-exhaustive switches require explicit [[assume(false)]] +
// VORTEX_UNREACHABLE(). Raw integers are never silently coerced to a default.
#if VORTEX_DEBUG
    #define VORTEX_UNREACHABLE()                                             \
        do {                                                                 \
            std::fprintf(stderr,                                             \
                         "VORTEX FATAL: unreachable executed at %s:%d\n",    \
                         __FILE__, __LINE__);                                \
            std::abort();                                                    \
        } while (0)
#else
    #define VORTEX_UNREACHABLE() __builtin_unreachable()
#endif
