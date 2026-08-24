// =============================================================================
// vortex/ir/graph.hpp — Sea-of-Nodes graph container (Rules 7, 9, 15)
//
// Purpose:
//   Owns all Nodes of one compilation unit (function). Nodes are addressed
//   by dense NodeId; storage is arena-backed (Rule 7) and stable — node
//   addresses never move (Rule 9: raw pointers inside a pass are legal
//   because the arena does not relocate; the serialized form still uses ids).
//
// Invariants:
//   - ids are append-dense; killed nodes are tombstoned (Dead flag) and
//     compacted only by Graph::compact() at explicit pass boundaries so
//     passes may hold raw Node& across mutations within one pass.
//   - Use counts are maintained eagerly on input-list edits.
//   - Effect-chain continuity is enforced by the Verifier (Rule 40).
//
// Cross-references: frame_state.hpp (Rule 5 attachments), verifier.hpp.
// =============================================================================

#pragma once

#include <cstdint>

#include "vortex/ir/node.hpp"
#include "vortex/ir/ty.hpp"
#include "vortex/stdx/small_vector.hpp"
#include "vortex/support/arena.hpp"
#include "vortex/support/result.hpp"
#include "vortex/support/symbol_table.hpp"

namespace vortex::ir {

inline namespace abi_v1 {

/// FrameState descriptor (Rule 5): everything the Deoptimizer needs to rebuild
/// the Tier-0 world at a guard failure.
struct FrameState {
    std::uint32_t bytecode_offset{0};        // Tier-0 resume point
    std::uint32_t code_unit_id{0};
    stdx::small_vector<NodeId, 8> values{};  // SSA nodes materialized into regs
    stdx::small_vector<std::uint8_t, 8> kinds{};  // 0=tagged 1=int64 2=float64
};

struct Graph {
    // --- construction ----------------------------------------------------------
    Graph() { nodes_.push_back(Node{});   // id 0 reserved: never allocated
               nodes_[0].kind = NodeKind::Unreachable;
               nodes_[0].set_flag(NodeFlag::Dead); }

    [[nodiscard]] NodeId create(NodeKind kind) noexcept;
    [[nodiscard]] NodeId create(NodeKind kind, std::initializer_list<NodeId> ins) noexcept;
    /// Creates or returns the structurally-identical existing node (hash
    /// consing for pure nodes — used by GVN-style construction; Rule 10).
    [[nodiscard]] NodeId create_pure_consed(NodeKind kind, std::initializer_list<NodeId> ins,
                                            std::uint16_t subop = 0) noexcept;

    // --- access ------------------------------------------------------------------
    [[nodiscard]] Node& node(NodeId id) noexcept {
        VORTEX_ASSUME(id < nodes_.size());
        return nodes_[id];
    }
    [[nodiscard]] const Node& node(NodeId id) const noexcept {
        VORTEX_ASSUME(id < nodes_.size());
        return nodes_[id];
    }
    [[nodiscard]] std::uint32_t node_count() const noexcept {
        return static_cast<std::uint32_t>(nodes_.size());
    }
    [[nodiscard]] std::uint32_t live_node_count() const noexcept;

    /// Iterate all live node ids (dense order).
    template <typename F>
    void for_each_live(F&& fn) const {
        for (std::uint32_t i = 1; i < nodes_.size(); ++i) {
            if (!nodes_[i].has(NodeFlag::Dead)) fn(NodeId(i));
        }
    }

    // --- mutation utilities --------------------------------------------------------
    void add_input(NodeId n, NodeId input) noexcept;
    void set_input(NodeId n, std::uint32_t index, NodeId input) noexcept;
    /// Replace every use of `old_node` with `new_node` (classic RAUW).
    void replace_all_uses(NodeId old_node, NodeId new_node) noexcept;
    /// Kill node: mark dead, decrement input use counts.
    void kill(NodeId n) noexcept;

    // --- FrameState management (Rule 5) ---------------------------------------------
    [[nodiscard]] std::uint32_t add_frame_state(FrameState fs) noexcept {
        frame_states_.push_back(fs);
        return static_cast<std::uint32_t>(frame_states_.size() - 1);
    }
    [[nodiscard]] const FrameState& frame_state(std::uint32_t idx) const noexcept {
        VORTEX_ASSUME(idx < frame_states_.size());
        return frame_states_[idx];
    }
    [[nodiscard]] std::uint32_t frame_state_count() const noexcept {
        return static_cast<std::uint32_t>(frame_states_.size());
    }

    // --- helpers -----------------------------------------------------------------------
    [[nodiscard]] NodeId start() const noexcept { return start_; }
    void set_start(NodeId s) noexcept { start_ = s; }
    [[nodiscard]] NodeId end() const noexcept { return end_; }   // unique Return/Throw sink
    void set_end(NodeId e) noexcept { end_ = e; }

    /// Collect users of `n` (scan — passes call this at fixpoints, never in
    /// innermost loops; hot queries maintain their own use maps).
    [[nodiscard]] stdx::small_vector<NodeId, 8> users_of(NodeId n) const noexcept;

    /// True if `maybe_user` consumes `n` in any input slot.
    [[nodiscard]] bool is_used_by(NodeId n, NodeId maybe_user) const noexcept;

    /// Deterministic deep structural hash for one node (GVN key).
    [[nodiscard]] std::uint64_t node_hash(NodeId n) const noexcept;

    /// Bulk SoA extraction (Rule 20): fills contiguous arrays of kinds/flags.
    void gather_kinds(stdx::small_vector<std::uint16_t, 256>& kinds,
                      stdx::small_vector<std::uint32_t, 256>& flags) const;

    /// Type table associated with this graph's inference runs.
    TyTable types;

    /// Source metadata for diagnostics (Rule 47).
    SymbolId function_name{0xFFFF'FFFF};
    std::uint32_t n_parameters{0};

private:
    stdx::small_vector<Node, 512> nodes_;   // index == NodeId; slot 0 reserved
    stdx::small_vector<FrameState, 16> frame_states_;
    NodeId start_{invalid_node};
    NodeId end_{invalid_node};
};

}  // namespace abi_v1
}  // namespace vortex::ir
