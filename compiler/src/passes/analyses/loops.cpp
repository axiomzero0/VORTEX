// =============================================================================
// vortex/passes/analyses/loops.cpp — natural-loop detection implementation.
// =============================================================================

#include "vortex/passes/analyses/loops.hpp"

namespace vortex::passes {
inline namespace abi_v1 {

using namespace vortex::ir;

namespace {

[[nodiscard]] bool is_leader(NodeKind k) noexcept {
    switch (k) {
        case NodeKind::Start: case NodeKind::Region: case NodeKind::Loop:
        case NodeKind::Catch: case NodeKind::IfTrue: case NodeKind::IfFalse:
        case NodeKind::Jump:
            return true;
        default: return false;
    }
}

// successors of a block (same model as dominators.cpp)
[[nodiscard]] stdx::small_vector<NodeId, 8> succs(const Graph& g, NodeId b) noexcept {
    stdx::small_vector<NodeId, 8> out;
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        if (!is_leader(n.kind) || id == b || n.ins.empty()) return;
        if (n.kind == NodeKind::Region || n.kind == NodeKind::Loop ||
            n.kind == NodeKind::Catch) {
            for (NodeId in : n.ins) {
                if (in == b) { out.push_back(id); return; }
            }
            return;
        }
        if (n.kind == NodeKind::If && n.ins[0] == b) {
            g.for_each_live([&](NodeId proj) {
                const Node& p = g.node(proj);
                if ((p.kind == NodeKind::IfTrue || p.kind == NodeKind::IfFalse) &&
                    !p.ins.empty() && p.ins[0] == id) {
                    out.push_back(proj);
                }
            });
            return;
        }
        if (n.ins[0] == b) out.push_back(id);
    });
    return out;
}

}  // namespace

LoopInfo compute_loops(const Graph& g, const DomTree& dom) noexcept {
    LoopInfo info;
    if (!dom.valid()) return info;

    // Find back edges: edge a -> h where h dominates a (on the block graph).
    for (NodeId a : dom.rpo) {
        for (NodeId h : succs(g, a)) {
            if (!dom.rpo_index.contains(h)) continue;
            // must be ordered h before a in RPO and h dom a
            std::uint32_t ih = *dom.rpo_index.get(h);
            std::uint32_t ia = dom.rpo_index.contains(a) ? *dom.rpo_index.get(a) : 0xFFFFFFFFu;
            if (ia == 0xFFFFFFFFu || ih >= ia) continue;
            if (!dom.dominates(h, a)) continue;

            // natural loop of h: reverse-reach a without crossing h.
            LoopInfo::Loop loop;
            loop.header = h;
            loop.backedges.push_back(a);
            stdx::flat_map<NodeId, bool, 32> in_loop;
            in_loop.insert(h, true);
            stdx::small_vector<NodeId, 64> work{a};
            in_loop.insert(a, true);
            while (!work.empty()) {
                NodeId m = work.back();
                work.pop_back();
                // predecessors of m: block leaders having m as successor
                for (NodeId p : dom.rpo) {
                    if (in_loop.contains(p)) continue;
                    for (NodeId s : succs(g, p)) {
                        if (s == m) {
                            in_loop.insert(p, true);
                            work.push_back(p);
                            break;
                        }
                    }
                }
            }
            for (auto& kv : in_loop) loop.blocks.push_back(kv.first);
            info.loops.push_back(loop);
        }
    }

    // Nesting depth: loop L depth = 1 + depth of innermost enclosing loop.
    for (auto& l : info.loops) {
        std::uint32_t d = 1;
        for (auto& other : info.loops) {
            if (&other == &l) continue;
            // other encloses l iff other.header is in l.blocks and l.header
            // is NOT in other.blocks (strictly outer).
            bool other_has_header = false;
            for (NodeId m : other.blocks) {
                if (m == l.header) { other_has_header = true; break; }
            }
            bool l_has_other_header = false;
            for (NodeId m : l.blocks) {
                if (m == other.header) { l_has_other_header = true; break; }
            }
            if (other_has_header && !l_has_other_header) {
                std::uint32_t od = other.depth ? other.depth : 1;
                if (od + 1 > d) d = od + 1;
            }
        }
        l.depth = d;
    }
    return info;
}

}  // namespace abi_v1
}  // namespace vortex::passes
