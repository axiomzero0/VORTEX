// =============================================================================
// vortex/passes/analyses/loops.hpp — natural loop detection.
//
// Back edge a->h (a's DFS-discovered ancestor): the natural loop of h is
// {h} plus every node that can reach a without passing h (union-find on
// the dom tree). Consumed by LICM (27/28), unrolling (30), IV analysis
// (29), vectorization (31/32), fusion (35), and the polyhedral pass (33).
// =============================================================================

#pragma once

#include "vortex/passes/analyses/dominators.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

struct LoopInfo {
    struct Loop {
        ir::NodeId header{ir::invalid_node};        // the Loop node
        stdx::small_vector<ir::NodeId, 8> blocks{}; // all leaders in the loop
        stdx::small_vector<ir::NodeId, 4> backedges{}; // sources of back edges
        std::uint32_t depth{0};                     // nesting depth
    };
    stdx::small_vector<Loop, 8> loops{};

    /// Innermost loop containing block `b` (invalid_node if none).
    [[nodiscard]] const Loop* innermost_of(ir::NodeId b) const noexcept {
        const Loop* best = nullptr;
        for (const Loop& l : loops) {
            for (ir::NodeId m : l.blocks) {
                if (m == b) {
                    if (!best || l.depth > best->depth) best = &l;
                    break;
                }
            }
        }
        return best;
    }

    /// Loop nesting depth of block b (0 = not in a loop).
    [[nodiscard]] std::uint32_t depth_of(ir::NodeId b) const noexcept {
        const Loop* l = innermost_of(b);
        return l ? l->depth : 0;
    }
};

[[nodiscard]] LoopInfo compute_loops(const ir::Graph& g, const DomTree& dom) noexcept;

}  // namespace abi_v1
}  // namespace vortex::passes
