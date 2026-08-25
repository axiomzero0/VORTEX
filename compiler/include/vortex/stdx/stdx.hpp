// =============================================================================
// vortex/stdx/stdx.hpp — C++26 standard-facility aliases with real fallbacks
//
// Purpose:
//   The VORTEX laws mandate C++26 data structures (std::inplace_vector,
//   std::flat_map, std::mdspan). Vendor support is uneven; this header selects
//   the standard facility when its feature-test macro / header is present and
//   otherwise binds to a real, API-compatible vortex::stdx implementation.
//   No call site ever changes (Rule 56: one abstraction, two concrete uses).
//
// Selection matrix (audited in tests/unit/stdx_alias_test.cpp):
//   facility          std:: when                       else
//   inplace_vector    __cpp_lib_inplace_vector         stdx::small_vector
//   flat_map          __cpp_lib_flat_map               stdx::flat_map
// =============================================================================

#pragma once

#include <version>

#if defined(__cpp_lib_inplace_vector)
    #include <inplace_vector>
    #define VORTEX_HAS_STD_INPLACE_VECTOR 1
#else
    #define VORTEX_HAS_STD_INPLACE_VECTOR 0
#endif

#if defined(__cpp_lib_flat_map)
    #include <flat_map>
    #define VORTEX_HAS_STD_FLAT_MAP 1
#else
    #define VORTEX_HAS_STD_FLAT_MAP 0
#endif

#include "vortex/stdx/flat_map.hpp"
#include "vortex/stdx/small_vector.hpp"

namespace vortex::stdx {
inline namespace abi_v1 {

#if VORTEX_HAS_STD_INPLACE_VECTOR
template <typename T, std::size_t N>
using inplace_vector = std::inplace_vector<T, N>;
#else
/// Fixed-ish inline vector: identical usage profile to std::inplace_vector
/// for the operations VORTEX performs; additionally spill-safe for data
/// dependent sizes (see docs/adr/0002-stdx-fallbacks.md for the deviation
/// audit — capacity() reports inline_capacity, never exceeded silently).
template <typename T, std::size_t N>
using inplace_vector = small_vector<T, N>;
#endif

#if VORTEX_HAS_STD_FLAT_MAP
template <typename Key, typename Value, typename Compare = std::less<Key>>
using flat_map_std = std::flat_map<Key, Value, Compare>;
#endif

// VORTEX passes always use the stdx::flat_map spelling because pass state
// needs reserve()/insert(key, value) ergonomics; when std::flat_map is
// available it is exercised by the alias conformance test to guarantee a
// future switch is a one-line change (Rule 51: automated refactoring).

}  // namespace abi_v1
}  // namespace vortex::stdx
