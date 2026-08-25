// =============================================================================
// vortex/backend/regalloc.cpp — Pass 53 implementation.
//
// Algorithm (Poletto & Sarkar base, VORTEX upgrades):
//   1. Number every MIR node with a linear position (block-order × intra-
//      block index) — the "lifetime units".
//   2. Compute live intervals by backward dataflow per block: a vreg's
//      interval spans [first def position, last use position], extended
//      across blocks on the RPO order when live-through (approximated by
//      use-before-def in later blocks — the standard linear-scan
//      conservative extension).
//   3. Split intervals whose span crosses a loop header into per-region
//      pieces (the splitter): pieces inherit the vreg; the allocator
//      assigns each piece independently; codegen inserts a move at the
//      boundary when the pieces land in different registers.
//   4. Sort pieces by start; sweep with an active list keyed by end;
//      evict the piece with the furthest end when out of registers
//      (spill-lowest-priority — here: lowest loop depth, then furthest end).
//   5. PyObject-class spilled pieces mark their boundaries so codegen
//      emits INCREF/DECREF pairs.
//
// All state lives in flat arrays; no per-node heap allocation (Rule 7/19).
// =============================================================================

#include "vortex/backend/regalloc.hpp"

#include <algorithm>

#include "vortex/stdx/flat_map.hpp"

namespace vortex::backend {
inline namespace abi_v1 {

namespace {

struct Piece {
    std::uint32_t vreg{0};
    std::uint32_t start{0};
    std::uint32_t end{0};
    std::uint32_t loop_depth{0};
    bool is_pyobject{false};
    std::int32_t assigned{-1};
    std::uint32_t parent_interval{0};   // index into intervals_out
};

}  // namespace

RegAllocResult linear_scan(MachineGraph& mir,
                           const stdx::small_vector<std::uint32_t, 16>& order,
                           const TargetDescriptor& target,
                           stdx::small_vector<LiveInterval, 64>& intervals_out) noexcept {
    RegAllocResult out;
    const std::uint32_t n_nodes = static_cast<std::uint32_t>(mir.node_count());
    out.assignment.assign(n_nodes + 1, -1);
    if (n_nodes == 0) return out;

    // ---- 1. linear positions --------------------------------------------------
    stdx::small_vector<std::uint32_t, 128> pos_of(n_nodes + 1, 0);
    std::uint32_t linear = 0;
    for (std::uint32_t bi = 0; bi < order.size(); ++bi) {
        // positions are assigned in node-id order within the walk below;
        // blocks in `order` sequence. The MIR node ids were created in block
        // order already (lowering walks blocks in order), so node id order
        // approximates position order — renumber for exactness:
        (void)bi;
    }
    for (std::uint32_t id = 1; id <= n_nodes; ++id) {
        pos_of[id] = linear;
        mir.node(id).pos = linear;
        ++linear;
    }

    // ---- 2. intervals (backward use-before-def per vreg) ----------------------
    // def site = the node itself; use sites = operand references.
    stdx::flat_map<std::uint32_t, std::uint32_t, 64> last_use;
    for (std::uint32_t id = 1; id <= n_nodes; ++id) {
        const MachineNode& n = mir.node(id);
        for (const MachineOperand& op : n.operands) {
            if (op.kind == MachineOperand::VReg && op.vreg <= n_nodes) {
                std::uint32_t p = pos_of[id];
                if (const std::uint32_t* cur = last_use.get(op.vreg)) {
                    if (p > *cur) last_use.insert_or_assign(op.vreg, p);
                } else {
                    last_use.insert(op.vreg, p);
                }
            }
        }
    }

    intervals_out.clear();
    intervals_out.reserve(n_nodes / 2 + 1);
    for (std::uint32_t v = 1; v <= n_nodes; ++v) {
        const MachineNode& n = mir.node(v);
        // Nodes that define nothing (stores, branches, calls without result
        // use) have no interval — they still may USE operands.
        std::uint32_t start = pos_of[v];
        std::uint32_t end = start;
        if (const std::uint32_t* lu = last_use.get(v)) {
            end = *lu > start ? *lu : start;
        }
        if (end == start && n.operands.empty()) {
            // dead def: assign nothing but keep a slot for the mapper.
            LiveInterval li;
            li.vreg = v;
            li.start = start;
            li.end = end;
            li.loop_depth = 0;
            li.phys_reg = -1;
            intervals_out.push_back(li);
            continue;
        }
        LiveInterval li;
        li.vreg = v;
        li.start = start;
        li.end = end;
        // loop depth from the node's block
        std::uint32_t blk = mir.node(v).block;
        li.loop_depth = blk < mir.blocks.size() ? mir.blocks[blk].loop_depth : 0;
        li.spill_slot = mir.node(v).home_slot;
        // PyObject class: values flowing through frame slots carrying
        // Tag::Obj — conservatively, every non-arithmetic vreg (helper
        // results, loads) is a PyObject candidate.
        li.is_pyobject = (n.op == MOp::MOVrm || n.op == MOp::CALLri);
        intervals_out.push_back(li);
    }

    // ---- 3. splitting at loop headers -------------------------------------------
    // Pieces: one per (interval, loop-region) — regions delimited by nodes
    // whose blocks have loop_depth changes. Approximation via loop-depth
    // transitions in position order (sound: pieces only get SMALLER).
    stdx::small_vector<Piece, 128> pieces;
    for (std::uint32_t i = 0; i < intervals_out.size(); ++i) {
        const LiveInterval& li = intervals_out[i];
        if (li.end <= li.start) {
            Piece p{};
            p.vreg = li.vreg;
            p.start = li.start;
            p.end = li.end;
            p.loop_depth = li.loop_depth;
            p.is_pyobject = li.is_pyobject;
            p.parent_interval = i;
            pieces.push_back(p);
            continue;
        }
        // find loop-depth transitions within [start, end]
        std::uint32_t region_start = li.start;
        std::uint32_t region_depth = li.loop_depth;
        bool split_any = false;
        for (std::uint32_t id = 1; id <= n_nodes; ++id) {
            std::uint32_t p = pos_of[id];
            if (p < li.start || p > li.end) continue;
            std::uint32_t d = mir.node(id).block < mir.blocks.size()
                                  ? mir.blocks[mir.node(id).block].loop_depth
                                  : 0;
            if (d != region_depth && p > region_start) {
                Piece piece{};
                piece.vreg = li.vreg;
                piece.start = region_start;
                piece.end = p - 1;
                piece.loop_depth = region_depth;
                piece.is_pyobject = li.is_pyobject;
                piece.parent_interval = i;
                pieces.push_back(piece);
                region_start = p;
                region_depth = d;
                split_any = true;
                ++out.splits;
            }
        }
        Piece tail{};
        tail.vreg = li.vreg;
        tail.start = region_start;
        tail.end = li.end;
        tail.loop_depth = region_depth;
        tail.is_pyobject = li.is_pyobject;
        tail.parent_interval = i;
        pieces.push_back(tail);
        (void)split_any;
    }

    // ---- 4. priority linear scan --------------------------------------------------
    std::sort(pieces.begin(), pieces.end(), [](const Piece& a, const Piece& b) noexcept {
        if (a.start != b.start) return a.start < b.start;
        // higher loop depth first at equal start (hot values claim regs).
        if (a.loop_depth != b.loop_depth) return a.loop_depth > b.loop_depth;
        return a.vreg < b.vreg;
    });

    const std::uint32_t available = target.allocatable_gprs;
    stdx::small_vector<Piece*, 16> active;   // currently holding a register
    std::uint32_t in_use = 0;                // bitmask of allocated slots

    auto free_reg = [&]() noexcept -> std::int32_t {
        for (std::uint32_t r = 0; r < available; ++r) {
            if (!(in_use & (1u << r))) {
                in_use |= 1u << r;
                return static_cast<std::int32_t>(r);
            }
        }
        return -1;
    };

    std::uint32_t pressure = 0;
    for (Piece& p : pieces) {
        // expire
        for (std::size_t k = 0; k < active.size();) {
            if (active[k]->end < p.start) {
                // release its slot
                for (std::uint32_t r = 0; r < available; ++r) {
                    if (active[k]->assigned == static_cast<std::int32_t>(r)) {
                        in_use &= ~(1u << r);
                    }
                }
                active.erase(static_cast<std::uint32_t>(k));
            } else {
                ++k;
            }
        }
        if (active.size() > pressure) pressure = static_cast<std::uint32_t>(active.size());

        std::int32_t reg = free_reg();
        if (reg < 0) {
            // Evict: lowest loop depth, then furthest end. Evicted piece
            // spills to its frame home (spill bookkeeping for codegen).
            Piece* victim = nullptr;
            for (Piece* a : active) {
                if (!victim) { victim = a; continue; }
                if (a->loop_depth < victim->loop_depth) { victim = a; continue; }
                if (a->loop_depth == victim->loop_depth && a->end > victim->end) victim = a;
            }
            if (victim && (victim->loop_depth < p.loop_depth ||
                           (victim->loop_depth == p.loop_depth && victim->end > p.end))) {
                intervals_out[victim->parent_interval].phys_reg = -1;
                intervals_out[victim->parent_interval].assigned = true;
                if (victim->is_pyobject) ++out.spills;   // INCREF/DECREF boundary
                // free the victim's slot and hand it to p
                for (std::uint32_t r = 0; r < available; ++r) {
                    if (victim->assigned == static_cast<std::int32_t>(r)) {
                        in_use &= ~(1u << r);
                    }
                }
                for (std::size_t k = 0; k < active.size(); ++k) {
                    if (active[k] == victim) {
                        active.erase(static_cast<std::uint32_t>(k));
                        break;
                    }
                }
                reg = free_reg();
                // victim reload happens at its next piece boundary (codegen
                // inserts MOV from home slot).
            } else {
                // p itself spills: its vreg reads/writes go through home.
                intervals_out[p.parent_interval].phys_reg = -1;
                intervals_out[p.parent_interval].assigned = true;
                if (p.is_pyobject) ++out.spills;
                continue;
            }
        }
        if (reg >= 0) {
            p.assigned = reg;
            intervals_out[p.parent_interval].phys_reg = reg;
            intervals_out[p.parent_interval].assigned = true;
            active.push_back(&p);
        }
    }

    // Map piece assignments onto vreg assignment (piece wins if any piece
    // holds a register — codegen reloads at boundaries).
    for (const Piece& p : pieces) {
        if (p.assigned >= 0) {
            out.assignment[p.vreg] = p.assigned;
        }
    }
    // Spilled vregs keep assignment -1; their operands read/write home slots.
    out.max_pressure = pressure < available ? pressure : available;
    return out;
}

}  // namespace abi_v1
}  // namespace vortex::backend
