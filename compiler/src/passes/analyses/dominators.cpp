// =============================================================================
// vortex/passes/analyses/dominators.cpp — Lengauer-Tarjan implementation.
// =============================================================================

#include "vortex/passes/analyses/dominators.hpp"

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

[[nodiscard]] stdx::small_vector<NodeId, 8> block_succs(const Graph& g, NodeId b) noexcept {
    stdx::small_vector<NodeId, 8> out;
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        if (id == b || n.ins.empty()) return;
        // If nodes are not block leaders, but they carry a block's outgoing
        // conditional edges: an If controlled by b makes its projections
        // successors of b. This case MUST run before the is_leader filter —
        // the old order filtered If out first and silently dropped every
        // conditional edge, truncating the CFG at the first branch inside
        // any loop header (dominators then saw no loops at all).
        if (n.kind == NodeKind::If) {
            if (n.ins[0] != b) return;
            g.for_each_live([&](NodeId proj) {
                const Node& p = g.node(proj);
                if ((p.kind == NodeKind::IfTrue || p.kind == NodeKind::IfFalse) &&
                    !p.ins.empty() && p.ins[0] == id) {
                    out.push_back(proj);
                }
            });
            return;
        }
        if (!is_leader(n.kind)) return;
        if (n.kind == NodeKind::Region || n.kind == NodeKind::Loop ||
            n.kind == NodeKind::Catch) {
            for (NodeId in : n.ins) {
                if (in == b) { out.push_back(id); return; }
            }
            return;
        }
        if (n.ins[0] == b) out.push_back(id);
    });
    return out;
}

}  // namespace

DomTree compute_dominators(const Graph& g) noexcept {
    DomTree dt;
    if (g.start() == invalid_node) return dt;

    stdx::small_vector<NodeId, 64> order;
    stdx::flat_map<NodeId, std::uint32_t, 32> dfn;
    stdx::flat_map<NodeId, NodeId, 32> parent;
    {
        stdx::small_vector<NodeId, 64> stack;
        stack.push_back(g.start());
        stdx::flat_map<NodeId, bool, 32> seen;
        seen.insert(g.start(), true);
        while (!stack.empty()) {
            NodeId b = stack.back();
            stack.pop_back();
            dfn.insert(b, static_cast<std::uint32_t>(order.size()));
            order.push_back(b);
            for (NodeId s : block_succs(g, b)) {
                if (!seen.contains(s)) {
                    seen.insert(s, true);
                    parent.insert(s, b);
                    stack.push_back(s);
                }
            }
        }
    }
    const std::uint32_t n = static_cast<std::uint32_t>(order.size());
    if (n == 0) return dt;

    stdx::small_vector<NodeId, 64> vertex(n, invalid_node);
    stdx::small_vector<std::uint32_t, 64> semi(n, 0);
    stdx::small_vector<std::uint32_t, 64> ancestor(n, 0);
    stdx::small_vector<std::uint32_t, 64> label(n, 0);
    stdx::small_vector<std::uint32_t, 64> idom_num(n, 0);
    stdx::small_vector<stdx::small_vector<std::uint32_t, 4>, 64> bucket(n);

    for (std::uint32_t i = 0; i < n; ++i) {
        vertex[i] = order[i];
        semi[i] = i;
        label[i] = i;
        ancestor[i] = 0xFFFFFFFFu;
        idom_num[i] = 0xFFFFFFFFu;
    }

    stdx::small_vector<stdx::small_vector<std::uint32_t, 4>, 64> preds(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        for (NodeId s : block_succs(g, vertex[i])) {
            if (dfn.contains(s)) {
                std::uint32_t j = *dfn.get(s);
                if (j < n) preds[j].push_back(i);
            }
        }
    }

    auto compress = [&](std::uint32_t v) noexcept {
        stdx::small_vector<std::uint32_t, 32> path;
        std::uint32_t a = ancestor[v];
        while (a != 0xFFFFFFFFu && ancestor[a] != 0xFFFFFFFFu) {
            path.push_back(v);
            v = a;
            a = ancestor[v];
        }
        for (std::size_t k = path.size(); k-- > 0;) {
            std::uint32_t w = path[k];
            if (semi[label[ancestor[w]]] < semi[label[w]]) {
                label[w] = label[ancestor[w]];
            }
            ancestor[w] = ancestor[ancestor[w]];
        }
    };
    auto eval = [&](std::uint32_t v) noexcept -> std::uint32_t {
        if (ancestor[v] == 0xFFFFFFFFu) return v;
        compress(v);
        return label[v];
    };

    for (std::uint32_t i = n; i-- > 1;) {
        std::uint32_t w = i;
        for (std::uint32_t p : preds[w]) {
            std::uint32_t u = eval(p);
            if (semi[u] < semi[w]) semi[w] = semi[u];
        }
        bucket[semi[w]].push_back(w);
        ancestor[w] = parent.contains(vertex[w]) && dfn.contains(*parent.get(vertex[w]))
                          ? *dfn.get(*parent.get(vertex[w]))
                          : 0xFFFFFFFFu;
        std::uint32_t par = ancestor[w];
        if (par != 0xFFFFFFFFu) {
            for (std::uint32_t v : bucket[par]) {
                std::uint32_t u = eval(v);
                idom_num[v] = (semi[u] < semi[v]) ? u : par;
            }
            bucket[par].clear();
        }
    }
    for (std::uint32_t i = 1; i < n; ++i) {
        if (idom_num[i] != 0xFFFFFFFFu && idom_num[i] != semi[i]) {
            idom_num[i] = idom_num[idom_num[i]];
        }
    }

    for (std::uint32_t i = 0; i < n; ++i) {
        NodeId b = vertex[i];
        dt.rpo.push_back(b);
        dt.rpo_index.insert(b, i);
        if (i == 0) {
            dt.idom.insert(b, invalid_node);
            dt.depth.insert(b, 0);
        } else {
            NodeId dom = idom_num[i] == 0xFFFFFFFFu ? invalid_node : vertex[idom_num[i]];
            dt.idom.insert(b, dom);
            std::uint32_t d = 0;
            if (const std::uint32_t* dd = dt.depth.get(dom)) d = *dd + 1;
            dt.depth.insert(b, d);
        }
    }
    return dt;
}

}  // namespace abi_v1
}  // namespace vortex::passes
