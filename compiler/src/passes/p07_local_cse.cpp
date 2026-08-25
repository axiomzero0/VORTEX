// =============================================================================
// Pass 07 — Common Subexpression Elimination (local, per-block).
//
// Hash-conses pure, side-effect-free nodes (constants, conversions,
// native arithmetic) keyed by structural hash. Python-level ops are
// excluded: they may dispatch user code (MayCall) and therefore are not
// context-insensitive-value-equal even with identical inputs.
// =============================================================================

#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

namespace {

bool local_cse(Graph& g) noexcept {
    bool changed = false;
    stdx::flat_map<std::uint64_t, NodeId, 64> table;
    g.for_each_live([&](NodeId id) {
        Node& n = g.node(id);
        if (is_control(n.kind)) return;
        if (!n.has(NodeFlag::Pure)) return;
        if (n.has(NodeFlag::MayCall) || n.has(NodeFlag::MayThrow)) return;
        std::uint64_t h = g.node_hash(id);
        for (std::uint32_t attempt = 0; attempt < 2; ++attempt) {
            std::uint64_t key = h + attempt;
            if (NodeId* existing = table.get(key)) {
                if (g.node(*existing).structurally_equal(n)) {
                    g.replace_all_uses(id, *existing);
                    g.kill(id);
                    changed = true;
                    return;
                }
            } else {
                table.insert(key, id);
                return;
            }
        }
        table.insert(h ^ 0x9e3779b97f4a7c15ull, id);
    });
    return changed;
}

}  // namespace

Result<PassResult> P07_LocalCSE::run(Graph& g, const PassContext& c) noexcept {
    (void)c;
    std::uint32_t before = g.live_node_count();
    local_cse(g);
    return result_of(g, before);
}

}  // namespace abi_v1
}  // namespace vortex::passes
