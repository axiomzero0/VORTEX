// =============================================================================
// Pass 46 — Dictionary-to-Struct (D2S) Layout Specialization.
//
// A dict used strictly as a record (constant key set, no dynamic insertion
// after construction, no key-dependent iteration) converts to fixed-offset
// field access:
//   StoreIndex(dict, const_k, v) -> StoreField(dict, slot=k_ordinal, v)
//   LoadIndex(dict, const_k)     -> LoadField(dict, slot=k_ordinal)
//
// The rewrite is GUARDED by a shape check (Rule 3): a Guard(ShapeIs) node
// is emitted at the dict's allocation site, carrying a FrameState (Rule 5)
// so the deoptimizer can reconstruct the hash-table form if a runtime
// mutation changes the dict's shape (Rule 4). The shape_id is derived
// from the dict's construction key set — a stable hash of the key symbols
// in construction order.
//
// Idempotency (Rule 10): bases that already have a Guard(ShapeIs) attached
// are skipped on the second run.
//
// Cost model (Rule 45): the layout win pays only when the dict has between
// cfg::dict_layout_min_fields and cfg::dict_layout_max_fields keys. Below
// the minimum, the shape guard overhead exceeds the single-probe hash
// cost it eliminates; above the maximum, the field-offset table grows
// larger than the hash table's compact representation.
// =============================================================================

#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"
#include "vortex/support/config.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

namespace {

[[nodiscard]] bool is_alloc(NodeKind k) noexcept {
    return k == NodeKind::NewDict;
}

/// Compute a stable shape_id from the construction key set. The shape_id
/// is a hash of the key symbols in construction order — two dicts with
/// the same keys in the same order get the same shape_id, so the Guard
/// can verify the layout is still valid at runtime.
[[nodiscard]] std::uint32_t compute_shape_id(
    const stdx::small_vector<std::uint32_t, 8>& keys) noexcept {
    // FNV-1a — deterministic, cheap, good distribution for small key sets.
    std::uint32_t h = 2166136261u;
    for (std::uint32_t k : keys) {
        h ^= k;
        h *= 16777619u;
    }
    return h == 0 ? 1 : h;   // 0 is reserved for "no shape"
}

/// Check whether a Guard(ShapeIs) already exists for `base` — idempotency.
/// The guard's ins[0] is the base it protects.
[[nodiscard]] bool has_shape_guard(const Graph& g, NodeId base) noexcept {
    bool found = false;
    g.for_each_live([&](NodeId id) {
        if (found) return;
        const Node& n = g.node(id);
        if (n.kind != NodeKind::Guard) return;
        if (n.subop != static_cast<std::uint16_t>(GuardKind::ShapeIs)) return;
        if (!n.ins.empty() && n.ins[0] == base) found = true;
    });
    return found;
}

/// Emit a Guard(ShapeIs) at the dict's allocation site. The guard carries
/// a FrameState (Rule 5) so the deoptimizer can reconstruct the hash-table
/// form on shape mismatch (Rule 4). The base's ShapeGuarded flag is set
/// so downstream LoadField/StoreField reads know the layout is proven.
void emit_shape_guard(Graph& g, const PassContext& c, NodeId base,
                       std::uint32_t shape_id) noexcept {
    if (has_shape_guard(g, base)) return;   // idempotent

    FrameState fs;
    fs.code_unit_id = c.code_unit_id;
    fs.bytecode_offset = 0;   // safepoint_pcs table not yet wired (driver.cpp)
    fs.values.push_back(base);
    fs.kinds.push_back(0);   // tagged
    std::uint32_t fs_idx = g.add_frame_state(fs);

    // Guard node: ins[0]=base, ins[1]=FrameStateNode ref.
    // The shape_id is stored in the guard's shape_id field.
    NodeId guard = g.create(NodeKind::Guard, {base});
    Node& gn = g.node(guard);
    gn.subop = static_cast<std::uint16_t>(GuardKind::ShapeIs);
    gn.shape_id = shape_id;
    gn.aux1 = fs_idx;
    gn.set_flag(NodeFlag::Speculative);   // Rule 5: FrameState attached
    gn.set_flag(NodeFlag::Pure);          // a guard predicate has no side effect
    gn.set_flag(NodeFlag::ShapeGuarded);

    g.node(base).set_flag(NodeFlag::ShapeGuarded);
}

}  // namespace

Result<PassResult> P46_DictLayoutSpecialization::run(Graph& g, const PassContext& c) noexcept {
    if (c.tier == TierMode::Tier1) return PassResult{};   // budget: Tier 2/3 only
    std::uint32_t before = g.live_node_count();
    std::uint32_t specialized = 0;

    // Collect constant-key stores per dict base, in program order, and
    // remember each key's FIRST store id (the literal construction write).
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
        if (!is_alloc(g.node(base).kind)) return;
        if (!layout.get(base)) layout.insert(base, BaseLayout{});
        BaseLayout* lay = layout.get(base);
        for (std::uint32_t i = 0; i < lay->keys.size(); ++i) {
            if (lay->keys[i] == key.symbol) return;   // duplicate key store
        }
        lay->keys.push_back(key.symbol);
        lay->first_store.push_back(id);
    });

    for (auto& kv : layout) {
        // Cost model (Rule 45): skip if the key set is too small (guard
        // overhead exceeds single-probe hash cost) or too large (field
        // table exceeds hash table's compactness).
        if (kv.second.keys.size() < cfg::dict_layout_min_fields) continue;
        if (kv.second.keys.size() > cfg::dict_layout_max_fields) continue;

        // Any operation on this base that requires the hash-table form
        // blocks specialization. This includes:
        //   - Non-constant-key StoreIndex/LoadIndex (dynamic shape)
        //   - Iter (for k in d) — needs the dict's iteration protocol
        //   - Len — needs the dict's internal size counter
        //   - LoadAttr/StoreAttr — attribute access on the dict object
        //   - CallPy/CallNative with the dict as an argument — the callee
        //     may iterate, len-check, or mutate the dict in hash-table form
        //   - Return — the dict escapes to the caller, who may iterate it
        // The spec is explicit: "no key-dependent iteration" blocks D2S.
        bool blocked = false;
        g.for_each_live([&](NodeId id) {
            if (blocked) return;
            const Node& n = g.node(id);
            // Check if this node uses the dict base as a data input.
            bool uses_base = false;
            for (NodeId in : n.ins) {
                if (in == kv.first) { uses_base = true; break; }
            }
            if (!uses_base) return;

            // StoreIndex/LoadIndex with constant keys are the ONLY allowed
            // operations. Everything else blocks.
            if (n.kind == NodeKind::StoreIndex || n.kind == NodeKind::LoadIndex) {
                if (n.ins.size() >= 4 && g.node(n.ins[3]).kind == NodeKind::ConstPy) {
                    return;   // constant-key access: allowed
                }
                blocked = true;
                return;
            }
            // Any other node kind that touches the dict blocks specialization.
            // This covers Iter, Len, CallPy, CallNative, Return, LoadAttr,
            // StoreAttr, and any future dict-consuming operation.
            blocked = true;
        });
        if (blocked) continue;

        // Emit the shape guard once for this base (Rule 3/4/5). The
        // shape_id is derived from the construction key set — if the
        // dict's shape changes at runtime (new key added), the guard
        // fails and the deoptimizer reconstructs the hash-table form.
        const std::uint32_t shape_id = compute_shape_id(kv.second.keys);
        emit_shape_guard(g, c, kv.first, shape_id);

        // Collect ALL stores and loads to rewrite for this base, keyed
        // by their key symbol. We gather first, then rewrite — mutating
        // the graph (g.create, g.kill) while iterating with for_each_live
        // is unsafe because g.create() can grow the nodes_ small_vector
        // and invalidate Node& references held in the lambda body.
        stdx::flat_map<std::uint32_t, stdx::small_vector<NodeId, 8>, 32> stores_by_key;
        stdx::flat_map<std::uint32_t, stdx::small_vector<NodeId, 8>, 32> loads_by_key;
        g.for_each_live([&](NodeId id) {
            const Node& n = g.node(id);
            if (n.kind == NodeKind::StoreIndex && n.ins.size() >= 5 && n.ins[2] == kv.first) {
                const Node& key = g.node(n.ins[3]);
                if (key.kind == NodeKind::ConstPy) {
                    if (!stores_by_key.get(key.symbol)) {
                        stores_by_key.insert(key.symbol, stdx::small_vector<NodeId, 8>{});
                    }
                    stores_by_key.get(key.symbol)->push_back(id);
                }
            } else if (n.kind == NodeKind::LoadIndex && n.ins.size() >= 4 &&
                       n.ins[2] == kv.first) {
                const Node& key = g.node(n.ins[3]);
                if (key.kind == NodeKind::ConstPy) {
                    if (!loads_by_key.get(key.symbol)) {
                        loads_by_key.insert(key.symbol, stdx::small_vector<NodeId, 8>{});
                    }
                    loads_by_key.get(key.symbol)->push_back(id);
                }
            }
        });

        // Rewrite stores: StoreIndex -> StoreField with ordinal slot.
        // Snapshot the original inputs before creating the replacement
        // (g.create may invalidate references into nodes_).
        for (std::uint32_t slot = 0; slot < kv.second.keys.size(); ++slot) {
            std::uint32_t key_sym = kv.second.keys[slot];
            const auto* store_list = stores_by_key.get(key_sym);
            if (!store_list) continue;
            for (NodeId orig_id : *store_list) {
                if (g.node(orig_id).has(NodeFlag::Dead)) continue;   // already rewritten
                // Snapshot inputs BEFORE creating the replacement node.
                const Node& orig = g.node(orig_id);
                NodeId ctrl = orig.ins[0];
                NodeId eff = orig.ins[1];
                NodeId base = orig.ins[2];
                NodeId val = orig.ins[4];

                NodeId sf = g.create(NodeKind::StoreField);
                Node& sfn = g.node(sf);
                sfn.set_flag(NodeFlag::OnEffectChain);
                sfn.set_flag(NodeFlag::MayThrow);
                sfn.set_flag(NodeFlag::ShapeGuarded);
                sfn.aux0 = slot;
                sfn.shape_id = shape_id;
                g.add_input(sf, ctrl);
                g.add_input(sf, eff);
                g.add_input(sf, base);
                g.add_input(sf, val);
                g.replace_all_uses(orig_id, sf);
                g.kill(orig_id);
                ++specialized;
            }
        }

        // Rewrite loads: LoadIndex -> LoadField with ordinal slot.
        for (std::uint32_t slot = 0; slot < kv.second.keys.size(); ++slot) {
            std::uint32_t key_sym = kv.second.keys[slot];
            const auto* load_list = loads_by_key.get(key_sym);
            if (!load_list) continue;
            for (NodeId orig_id : *load_list) {
                if (g.node(orig_id).has(NodeFlag::Dead)) continue;
                const Node& orig = g.node(orig_id);
                NodeId ctrl = orig.ins[0];
                NodeId eff = orig.ins[1];
                NodeId base = orig.ins[2];

                NodeId lf = g.create(NodeKind::LoadField);
                Node& lfn = g.node(lf);
                lfn.set_flag(NodeFlag::OnEffectChain);
                lfn.set_flag(NodeFlag::MayThrow);
                lfn.set_flag(NodeFlag::ShapeGuarded);
                lfn.aux0 = slot;
                lfn.shape_id = shape_id;
                g.add_input(lf, ctrl);
                g.add_input(lf, eff);
                g.add_input(lf, base);
                g.replace_all_uses(orig_id, lf);
                g.kill(orig_id);
                ++specialized;
            }
        }
    }

    PassResult r = result_of(g, before);
    r.changed = specialized > 0;
    note(TelemetryEventKind::SafepointPatched, c, specialized);
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
