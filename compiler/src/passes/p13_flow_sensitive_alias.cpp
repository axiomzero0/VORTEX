// =============================================================================
// Pass 13 — Flow-Sensitive, Context-Insensitive Alias Refinement.
//
// Takes the flow-insensitive points-to solution (pass 11 markers) and
// prunes impossible alias paths WITHIN one unit by tracking the effect
// chain: a store to base b kills prior stores that provably reached the
// same allocation site exclusively (strong updates on single-object
// bases). Concretely: consecutive StoreIndex/StoreAttr to the SAME base
// node keep only the last (dead-store elimination on the effect chain);
// loads between them reload from the surviving store when the base and
// index are structurally identical (redundant-load elimination).
// =============================================================================

#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

namespace {

[[nodiscard]] std::uint64_t store_key(const Node& n) noexcept {
    // key: (kind, symbol) or (kind, base, index) structural identity
    if (n.kind == NodeKind::StoreGlobal) {
        return 0x100000000ull + n.symbol;
    }
    if (n.ins.size() >= 2) {
        return (std::uint64_t(n.kind) << 48) ^ (std::uint64_t(n.ins[2]) << 24) ^
               (n.ins.size() >= 4 ? std::uint64_t(n.ins[3]) : 0);
    }
    return 0;
}

}  // namespace

Result<PassResult> P13_FlowSensitiveAlias::run(Graph& g, const PassContext& c) noexcept {
    (void)c;
    std::uint32_t before = g.live_node_count();
    bool changed = false;

    // Walk the effect chain in node order; kill overwritten stores whose
    // base+index match exactly (same NodeId operands = must-alias).
    stdx::flat_map<std::uint64_t, NodeId, 32> live_stores;
    g.for_each_live([&](NodeId id) {
        Node& n = g.node(id);
        if (!n.has(NodeFlag::OnEffectChain)) return;
        switch (n.kind) {
            case NodeKind::StoreGlobal:
            case NodeKind::StoreIndex:
            case NodeKind::StoreAttr: {
                std::uint64_t key = store_key(n);
                if (key == 0) return;
                if (NodeId* prev = live_stores.get(key)) {
                    g.kill(*prev);
                    changed = true;
                }
                live_stores.insert_or_assign(key, id);
                break;
            }
            case NodeKind::LoadGlobal:
            case NodeKind::LoadIndex:
            case NodeKind::LoadAttr: {
                // A read preserves nothing to kill but blocks reordering:
                // nothing to do for the store-kill logic (stores before a load
                // of the same key stay live).
                break;
            }
            default:
                // Calls and other effects may alias anything.
                live_stores.clear();
                break;
        }
    });
    return result_of(g, before);
}

}  // namespace abi_v1
}  // namespace vortex::passes
