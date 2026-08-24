// =============================================================================
// vortex/ir/printer.hpp / .cpp — canonical text form (golden tests, Rule 35)
//
// Format (deterministic; ids are assigned by creation order):
//   fun <symbol-text> params=<n>
//     n<id> = <kind> [payload] ins: <id id ...>      # live nodes only
//     ...
//   frame_states:
//     fs<idx> bcoff=<u32> unit=<u32> vals: n<id>... kinds: <b>...
//
// Invariants:
//   - Output is a pure function of graph content (no addresses, no PGO
//     counters) so golden files are reproducible (Rule 52).
//   - The parser (ir_parser.hpp) accepts exactly this grammar; round-trip
//     equality is asserted by tests/unit/ir_roundtrip_test.cpp.
// =============================================================================

#pragma once

#include <cstdio>

#include "vortex/ir/graph.hpp"

namespace vortex::ir {

inline namespace abi_v1 {

/// Print graph to `out` in canonical .vortex form.
void print_graph(const Graph& g, std::FILE* out) noexcept;

/// Render to an in-memory buffer (tests). Returns false on allocation failure.
[[nodiscard]] bool graph_to_string(const Graph& g, stdx::small_vector<char, 4096>& out) noexcept;

}  // namespace abi_v1
}  // namespace vortex::ir
