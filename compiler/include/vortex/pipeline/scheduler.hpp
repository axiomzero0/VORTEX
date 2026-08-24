// =============================================================================
// vortex/pipeline/scheduler.hpp — Sea of Nodes -> Tier-0 register bytecode
//
// Purpose:
//   Linearizes one IR unit into CodeUnit bytecode:
//     1. Blocks are control projections (Start, IfTrue, IfFalse, Region,
//        Loop, Catch) ordered by creation order (structured lowering makes
//        creation order topological for forward edges).
//     2. Effect ops are pre-assigned to their control block and emitted in
//        node-id order (creation order == program order).
//     3. Pure values are scheduled lazily at first use inside the emitting
//        block (recursively), giving deterministic, correct ordering.
//     4. Phi moves are emitted on incoming edges; parallel-move hazards
//        (source is also a phi written on the same edge) go through scratch
//        registers — two-phase copy.
//     5. try bodies emit TRY_BEGIN/TRY_END from Catch merge ranges.
//
// Register indices are NodeIds — the FrameState contract (Rule 4) holds by
// construction: deopt writes FrameState values into reg slots directly.
// =============================================================================

#pragma once

#include "vortex/ir/graph.hpp"
#include "vortex/rt/code.hpp"
#include "vortex/rt/object.hpp"

namespace vortex::pipeline {

inline namespace abi_v1 {

/// Schedule `graph` into `unit`. `const_strings` supplies module string pool
/// bytes for ConstPy strings. Returns diagnostics on malformed graphs.
[[nodiscard]] Result<void> schedule_unit(const ir::Graph& graph, rt::CodeUnit& unit,
                                         const stdx::small_vector<char, 4096>& string_pool,
                                         const stdx::small_vector<std::uint32_t, 8>& param_regs,
                                         const stdx::small_vector<SymbolId, 8>& param_names,
                                         bool is_generator) noexcept;

}  // namespace abi_v1
}  // namespace vortex::pipeline
