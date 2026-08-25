// =============================================================================
// Pass 09 — Redundant Store Elimination.
//
// A global store immediately overwritten by a later store to the SAME
// symbol/base with no intervening load or effect is dead. The pass walks
// the effect chain in program order (node-id order within a block equals
// creation order from the lowering), tracking the last store per (kind,
// symbol) key; any other effect invalidates the tracked set.
// =============================================================================

#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

Result<PassResult> P09_RedundantStoreElimination::run(Graph& g, const PassContext& c) noexcept {
    (void)c;
    std::uint32_t before = g.live_node_count();
    bool changed = false;

    stdx::flat_map<std::uint64_t, NodeId, 32> last_store;
    g.for_each_live([&](NodeId id) {
        Node& n = g.node(id);
        if (!n.has(NodeFlag::OnEffectChain)) return;
        if (n.kind == NodeKind::StoreGlobal) {
            std::uint64_t key = 0x100000000ull + n.symbol;
            if (NodeId* prev = last_store.get(key)) {
                // PASS-3 fix: g.kill(*prev) drops the previous store but
                // leaves its effect-chain consumers (the new store's
                // ins[1], plus any intervening effect op's ins[1]) pointing
                // at a dead node — broken effect chain. Rewire first:
                // replace_all_uses(*prev, prev's effect input) makes every
                // consumer of the killed store (data and effect) point at
                // the killed store's own effect input, bypassing the dead
                // node. StoreGlobal has no data consumers, so this only
                // rewires effect-chain links.
                Node& killed = g.node(*prev);
                if (killed.ins.size() >= 2 && killed.ins[1] != vortex::ir::invalid_node) {
                    g.replace_all_uses(*prev, killed.ins[1]);
                }
                g.kill(*prev);
                changed = true;
            }
            last_store.insert_or_assign(key, id);
        } else if (n.kind == NodeKind::LoadGlobal) {
            last_store.erase(0x100000000ull + n.symbol);
        } else {
            // Any other effect (call, attribute store, index store) may read
            // globals: drop everything tracked.
            last_store.clear();
        }
    });
    return result_of(g, before);
}

}  // namespace abi_v1
}  // namespace vortex::passes
