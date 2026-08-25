// =============================================================================
// vortex/ir/graph.cpp — Sea-of-Nodes container implementation.
// =============================================================================

#include "vortex/ir/graph.hpp"

#include <cstring>

namespace vortex::ir {
inline namespace abi_v1 {

namespace {
[[nodiscard]] std::uint64_t mix64(std::uint64_t x) noexcept {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ull;
    x ^= x >> 33;
    return x;
}
}  // namespace

NodeId Graph::create(NodeKind kind) noexcept {
    Node n;
    n.kind = kind;
    n.id = static_cast<NodeId>(nodes_.size());
    nodes_.push_back(n);
    return n.id;
}

NodeId Graph::create(NodeKind kind, std::initializer_list<NodeId> ins) noexcept {
    NodeId id = create(kind);
    for (NodeId input : ins) {
        if (input != invalid_node) add_input(id, input);
    }
    return id;
}

NodeId Graph::create_pure_consed(NodeKind kind, std::initializer_list<NodeId> ins,
                                 std::uint16_t subop) noexcept {
    // Hash-cons lookup: linear scan is fine at construction time (nodes are
    // created in the hundreds, and pass-level GVN uses its own table).
    Node probe;
    probe.kind = kind;
    probe.subop = subop;
    for (NodeId input : ins) {
        if (input != invalid_node) probe.ins.push_back(input);
    }
    for (std::uint32_t i = 1; i < nodes_.size(); ++i) {
        const Node& existing = nodes_[i];
        if (existing.has(NodeFlag::Dead) || !existing.has(NodeFlag::Pure)) continue;
        if (existing.kind == kind && existing.subop == subop &&
            existing.ins.size() == probe.ins.size() &&
            std::memcmp(existing.ins.data(), probe.ins.data(),
                        probe.ins.size() * sizeof(NodeId)) == 0) {
            return NodeId(i);
        }
    }
    NodeId id = create(kind);
    Node& n = nodes_[id];
    n.subop = subop;
    n.set_flag(NodeFlag::Pure);
    for (NodeId input : probe.ins) add_input(id, input);
    return id;
}

void Graph::add_input(NodeId n, NodeId input) noexcept {
    VORTEX_ASSUME(n < nodes_.size());
    VORTEX_ASSUME(input < nodes_.size());
    nodes_[n].ins.push_back(input);
    nodes_[input].use_count++;
}

void Graph::set_input(NodeId n, std::uint32_t index, NodeId input) noexcept {
    VORTEX_ASSUME(n < nodes_.size());
    VORTEX_ASSUME(index < nodes_[n].ins.size());
    NodeId old = nodes_[n].ins[index];
    if (old == input) return;
    if (old != invalid_node && old < nodes_.size()) {
        if (nodes_[old].use_count > 0) nodes_[old].use_count--;
    }
    nodes_[n].ins[index] = input;
    if (input != invalid_node && input < nodes_.size()) nodes_[input].use_count++;
}

void Graph::replace_all_uses(NodeId old_node, NodeId new_node) noexcept {
    VORTEX_ASSUME(old_node < nodes_.size());
    VORTEX_ASSUME(new_node < nodes_.size());
    for (std::uint32_t i = 1; i < nodes_.size(); ++i) {
        Node& n = nodes_[i];
        if (n.has(NodeFlag::Dead)) continue;
        for (std::uint32_t s = 0; s < n.ins.size(); ++s) {
            if (n.ins[s] == old_node) {
                n.ins[s] = new_node;
                nodes_[new_node].use_count++;
                if (nodes_[old_node].use_count > 0) nodes_[old_node].use_count--;
            }
        }
    }
}

void Graph::kill(NodeId n) noexcept {
    VORTEX_ASSUME(n < nodes_.size());
    Node& victim = nodes_[n];
    if (victim.has(NodeFlag::Dead)) return;
    for (NodeId input : victim.ins) {
        if (input != invalid_node && input < nodes_.size()) {
            if (nodes_[input].use_count > 0) nodes_[input].use_count--;
        }
    }
    victim.ins.clear();
    victim.set_flag(NodeFlag::Dead);
}

std::uint32_t Graph::live_node_count() const noexcept {
    std::uint32_t count = 0;
    for (std::uint32_t i = 1; i < nodes_.size(); ++i) {
        if (!nodes_[i].has(NodeFlag::Dead)) ++count;
    }
    return count;
}

stdx::small_vector<NodeId, 8> Graph::users_of(NodeId n) const noexcept {
    stdx::small_vector<NodeId, 8> users;
    for (std::uint32_t i = 1; i < nodes_.size(); ++i) {
        const Node& u = nodes_[i];
        if (u.has(NodeFlag::Dead)) continue;
        for (NodeId input : u.ins) {
            if (input == n) {
                users.push_back(NodeId(i));
                break;
            }
        }
    }
    return users;
}

bool Graph::is_used_by(NodeId n, NodeId maybe_user) const noexcept {
    const Node& u = nodes_[maybe_user];
    for (NodeId input : u.ins) {
        if (input == n) return true;
    }
    return false;
}

std::uint64_t Graph::node_hash(NodeId n) const noexcept {
    const Node& x = nodes_[n];
    std::uint64_t h = mix64(static_cast<std::uint64_t>(x.kind) << 48 |
                            static_cast<std::uint64_t>(x.subop) << 32 | x.symbol);
    h = mix64(h ^ x.const_value.as.i);
    h = mix64(h ^ (std::uint64_t{x.shape_id} << 32 | x.aux0));
    for (NodeId input : x.ins) {
        h = mix64(h ^ mix64(input + 0x9e3779b97f4a7c15ull));
    }
    return h;
}

void Graph::gather_kinds(stdx::small_vector<std::uint16_t, 256>& kinds,
                         stdx::small_vector<std::uint32_t, 256>& flags) const {
    kinds.clear();
    flags.clear();
    kinds.reserve(nodes_.size());
    flags.reserve(nodes_.size());
    for (std::uint32_t i = 0; i < nodes_.size(); ++i) {
        kinds.push_back(static_cast<std::uint16_t>(nodes_[i].kind));
        flags.push_back(nodes_[i].flags.raw());
    }
}

}  // namespace abi_v1
}  // namespace vortex::ir
