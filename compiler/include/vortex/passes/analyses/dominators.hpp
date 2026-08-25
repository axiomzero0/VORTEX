// =============================================================================
// vortex/passes/analyses/dominators.hpp — Lengauer-Tarjan immediate
// dominators (spec: Pass 2 computes them; every loop/region pass consumes).
//
// Semilrang-Tarjan "A Fast Algorithm for Finding Dominators in a Flowgraph"
// (1979): DFS numbering, semi-dominator via splay forest, idom derivation.
// O((n+m) α(n)) — the mandated algorithm, not the naive iterative one.
//
// Graph model: the Sea of Nodes is scheduled to blocks by the scheduler; for
// dominator purposes we operate on the BLOCK graph: block leaders (control
// projections) with edges derived from Region/Loop/Catch/If wiring.
// =============================================================================

#pragma once

#include <cstdint>

#include "vortex/ir/graph.hpp"
#include "vortex/stdx/small_vector.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

/// Dominator tree over the block graph of one unit.
struct DomTree {
    /// Block leaders in reverse-postorder.
    stdx::small_vector<ir::NodeId, 64> rpo{};
    /// idom[leader] == leader's immediate dominator (invalid for entry).
    stdx::flat_map<ir::NodeId, ir::NodeId, 32> idom{};
    /// Depth in the dominator tree (entry = 0) — used by LICM legality.
    stdx::flat_map<ir::NodeId, std::uint32_t, 32> depth{};
    /// RPO index per leader (fast "comes before" queries).
    stdx::flat_map<ir::NodeId, std::uint32_t, 32> rpo_index{};

    [[nodiscard]] bool dominates(ir::NodeId a, ir::NodeId b) const noexcept {
        // a dominates b iff a is an ancestor of b in the dom tree. Walk
        // depths: standard ancestor test using depth only when the tree is
        // a forest with unique parents (it is: idom is unique).
        if (a == b) return true;
        const ir::NodeId* da = depth.get(a);
        const ir::NodeId* db = depth.get(b);
        if (!da || !db) return false;
        ir::NodeId cur = b;
        std::uint32_t db_depth = *db;
        const std::uint32_t da_depth = *da;
        while (db_depth > da_depth) {
            const ir::NodeId* up = idom.get(cur);
            if (!up || *up == ir::invalid_node) return false;
            cur = *up;
            const std::uint32_t* d2 = depth.get(cur);
            if (!d2) return false;
            db_depth = *d2;
        }
        return cur == a;
    }

    [[nodiscard]] bool valid() const noexcept { return !rpo.empty(); }
};

/// Compute the dominator tree for `g`'s block graph (entry = Start).
[[nodiscard]] DomTree compute_dominators(const ir::Graph& g) noexcept;

}  // namespace abi_v1
}  // namespace vortex::passes
