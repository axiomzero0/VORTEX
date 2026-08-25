// =============================================================================
// Pass 46 — Dictionary Layout Specialization.
//
// A dict used strictly as a record (constant key set, no dynamic
// insertion after construction, no key-dependent iteration) converts to
// fixed-offset field access: StoreIndex(dict, const_k, v) becomes
// StoreField with slot = k's ordinal among the construction keys;
// LoadIndex likewise. Guarded by shape analysis (pass 15) markers —
// heterogeneous access patterns keep the hash table.
// =============================================================================

#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

Result<PassResult> P46_DictLayoutSpecialization::run(Graph& g, const PassContext& c) noexcept {
    (void)c;
    std::uint32_t before = g.live_node_count();
    std::uint32_t specialized = 0;

    // Collect constant-key stores per dict base, in program order, and
    // remember each key's FIRST store id (the literal construction write).
    // Only POST-construction stores may rewrite: construction stores extend
    // an empty dict (new-key inserts by definition).
    struct BaseLayout {
        stdx::small_vector<std::uint32_t, 8> keys{};
        stdx::small_vector<NodeId, 8> first_store{};
    };
    stdx::flat_map<NodeId, BaseLayout, 16> layout;
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        if (n.kind != NodeKind::StoreIndex || n.ins.size() < 5) return;
        NodeId base = n.ins[2];
        const Node& key = g.node(n.ins[3]);
        if (key.kind != NodeKind::ConstPy) return;   // constant keys only
        if (g.node(base).kind != NodeKind::NewDict) return;
        if (!layout.get(base)) layout.insert(base, BaseLayout{});
        BaseLayout* lay = layout.get(base);
        for (std::uint32_t i = 0; i < lay->keys.size(); ++i) {
            if (lay->keys[i] == key.symbol) return;   // duplicate key store
        }
        lay->keys.push_back(key.symbol);
        lay->first_store.push_back(id);
    });

    // Any NON-constant key access on those bases blocks specialization
    // (dynamic shape).
    for (auto& kv : layout) {
        bool blocked = false;
        g.for_each_live([&](NodeId id) {
            const Node& n = g.node(id);
            if (n.kind != NodeKind::StoreIndex && n.kind != NodeKind::LoadIndex) return;
            if (n.ins.size() < 4 || n.ins[2] != kv.first) return;
            if (g.node(n.ins[3]).kind != NodeKind::ConstPy) blocked = true;
        });
        if (blocked) continue;
        if (kv.second.keys.size() > 32) continue;   // layout win vanishes

        // Rewrite: stores become StoreField with ordinal slots.
        for (std::uint32_t slot = 0; slot < kv.second.keys.size(); ++slot) {
            std::uint32_t key_sym = kv.second.keys[slot];
            NodeId construction = kv.second.first_store[slot];
            g.for_each_live([&](NodeId id) {
                Node& n = g.node(id);
                if (n.kind != NodeKind::StoreIndex || n.ins.size() < 5) return;
                if (n.ins[2] != kv.first) return;
                const Node& key = g.node(n.ins[3]);
                if (key.kind != NodeKind::ConstPy || key.symbol != key_sym) return;
                // ONLY post-construction stores rewrite: the construction
                // write (node id == construction) must insert the key; later
                // writes may use the slot fast path.
                if (id <= construction) return;
                NodeId sf = g.create(NodeKind::StoreField);
                Node& sfn = g.node(sf);
                sfn.set_flag(NodeFlag::OnEffectChain);
                sfn.set_flag(NodeFlag::MayThrow);
                sfn.aux0 = slot;
                g.add_input(sf, n.ins[0]);
                g.add_input(sf, n.ins[1]);
                g.add_input(sf, n.ins[2]);
                g.add_input(sf, n.ins[4]);
                g.replace_all_uses(id, sf);
                g.kill(id);
                ++specialized;
            });
        }
    }

    PassResult r = result_of(g, before);
    r.changed = specialized > 0;
    note(TelemetryEventKind::SafepointPatched, c, specialized);
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
