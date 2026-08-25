// =============================================================================
// vortex/frontend/string_pool.hpp — shared string-pool alias.
//
// The Lexer writes cooked string bytes into the Module's pool; both types
// must be the exact same instantiation. One alias, one truth (Rule 57).
// =============================================================================
#pragma once

#include "vortex/stdx/small_vector.hpp"

namespace vortex::fe {
inline namespace abi_v1 {
using StringPool = stdx::small_vector<char, 4096>;
}  // namespace abi_v1
}  // namespace vortex::fe
