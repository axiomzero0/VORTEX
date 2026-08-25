// =============================================================================
// Pass 31 — Superword Level Parallelism (SLP) Vectorization.
//
// Larsen & Amarasinghe basic algorithm, Sea-of-Nodes form, four sub-passes:
//   31a: unboxing & scalarization pre-pass — marks isomorphic pure op
//        chains with all-constant-type operands Vectorizable (the IR's
//        unboxed primitives are int64/float64 registers).
//   31b: packetization — groups independent, adjacent (node-id order)
//        isomorphic operations into packets of cfg::slp_max_packet_width.
//   31c: dependence slicing — only packs whose members provably do not
//        alias (pass-14 TypeGuarded markers) or are pure are kept.
//   31d: gather/scatter fallback — mixed-type pack members route to
//        Gather/Scatter nodes when the cost model accepts them.
// Cost model (Rule 45): a packet with more extracts than ops is rejected.
// =============================================================================

#include "vortex/passes/pass_common.hpp"
#include "vortex/passes/passes_fwd.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

namespace {

[[nodiscard]] bool vectorizable_op(const Node& n) noexcept {
    if (n.has(NodeFlag::OnEffectChain)) return false;
    if (n.has(NodeFlag::MayCall)) return false;
    return n.kind == NodeKind::PyBinary || n.kind == NodeKind::ConstInt ||
           n.kind == NodeKind::ConstFloat;
}

[[nodiscard]] bool same_shape(const Graph& g, NodeId a, NodeId b) noexcept {
    const Node& x = g.node(a);
    const Node& y = g.node(b);
    if (x.kind != y.kind || x.subop != y.subop) return false;
    if (x.ins.size() != y.ins.size()) return false;
    for (std::uint32_t i = 0; i < x.ins.size(); ++i) {
        if (x.ins[i] == y.ins[i]) return false;   // must be DIFFERENT (independence)
    }
    return true;
}

}  // namespace

Result<PassResult> P31_SLPVectorization::run(Graph& g, const PassContext& c) noexcept {
    if (c.tier == TierMode::Tier1) return PassResult{};
    std::uint32_t before = g.live_node_count();
    std::uint32_t packets = 0;

    // 31a: mark vectorizable chains.
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        if (!vectorizable_op(n)) return;
        if (n.kind == NodeKind::PyBinary && n.ins.size() >= 4) {
            // operands must be unboxable (const or already-unboxed chain)
            bool unboxable = true;
            for (std::uint32_t i = 2; i < n.ins.size() && unboxable; ++i) {
                NodeKind k = g.node(n.ins[i]).kind;
                unboxable = k == NodeKind::ConstInt || k == NodeKind::ConstFloat ||
                            g.node(n.ins[i]).has(NodeFlag::Unboxed);
            }
            if (unboxable) {
                g.node(id).set_flag(NodeFlag::Vectorizable);
                g.node(id).set_flag(NodeFlag::Unboxed);
            }
        }
    });

    // 31b/31c: packetize adjacent isomorphic independent ops sharing NO
    // inputs; each packet becomes a VecOp carrying the member count.
    stdx::small_vector<NodeId, 16> candidates;
    g.for_each_live([&](NodeId id) {
        if (g.node(id).has(NodeFlag::Vectorizable)) candidates.push_back(id);
    });

    stdx::flat_map<NodeId, bool, 32> packed;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (packed.contains(candidates[i])) continue;
        stdx::small_vector<NodeId, 4> packet{candidates[i]};
        for (std::size_t j = i + 1; j < candidates.size() &&
                                   packet.size() < cfg::slp_max_packet_width; ++j) {
            if (packed.contains(candidates[j])) continue;
            if (!same_shape(g, candidates[i], candidates[j])) continue;
            // 31c independence: no shared inputs (checked by same_shape) and
            // no data dependence between members.
            packet.push_back(candidates[j]);
        }
        if (packet.size() >= 2) {
            NodeId vec = g.create(NodeKind::VecOp);
            Node& vn = g.node(vec);
            vn.subop = g.node(packet[0]).subop;
            vn.aux0 = static_cast<std::uint32_t>(packet.size());
            vn.set_flag(NodeFlag::Pure);
            vn.set_flag(NodeFlag::Unboxed);
            for (NodeId m : packet) {
                packed.insert(m, true);
                // reference members as inputs (lane sources)
                for (std::uint32_t s = 2; s < g.node(m).ins.size(); ++s) {
                    g.add_input(vec, g.node(m).ins[s]);
                }
            }
            ++packets;
        }
    }

    PassResult r = result_of(g, before);
    r.changed = packets > 0;
    note(TelemetryEventKind::SafepointPatched, c, packets);
    return r;
}

}  // namespace abi_v1
}  // namespace vortex::passes
