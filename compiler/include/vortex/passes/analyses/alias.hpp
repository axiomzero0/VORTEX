// =============================================================================
// vortex/passes/analyses/alias.hpp — alias-analysis result interface.
//
// Shared query surface consumed by passes 12 (CFL), 13 (flow-sensitive),
// 14 (demand-driven), 31/32 (vectorization legality), 49 (reordering).
// Points-to sets are produced by pass 11 (Andersen).
// =============================================================================

#pragma once

#include "vortex/ir/graph.hpp"
#include "vortex/stdx/small_vector.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

enum class AliasKind : std::uint8_t {
    NoAlias,      // provably different objects
    MayAlias,     // conservative: may or may not
    MustAlias,    // provably the same object
};

struct AliasResult {
    // Sorted-by-NodeId point-to sets for memory-relevant nodes (lists,
    // dicts, instances, parameters). Empty set = no info (MayAlias all).
    stdx::flat_map<ir::NodeId, stdx::small_vector<ir::NodeId, 4>, 32> points_to{};

    [[nodiscard]] std::size_t tracked_count() const noexcept { return points_to.size(); }

    /// Two nodes alias iff their point-to sets intersect (Andersen's
    /// conservative intersection rule).
    [[nodiscard]] AliasKind alias(ir::NodeId a, ir::NodeId b) const noexcept {
        const auto* sa = points_to.get(a);
        const auto* sb = points_to.get(b);
        if (!sa || !sb) return AliasKind::MayAlias;
        bool intersect = false;
        for (ir::NodeId x : *sa) {
            for (ir::NodeId y : *sb) {
                if (x == y) { intersect = true; break; }
            }
            if (intersect) break;
        }
        return intersect ? AliasKind::MayAlias : AliasKind::NoAlias;
    }

    [[nodiscard]] bool is_tracked(ir::NodeId n) const noexcept {
        return points_to.contains(n);
    }
};

}  // namespace abi_v1
}  // namespace vortex::passes
