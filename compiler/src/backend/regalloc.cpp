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
    bool is_fpr{false};        // reg class: false = GPR pool, true = FPR pool
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
    // LSRA->XMM extension: parallel assignment for FPR-class vregs. The
    // pool is sized to zero when the target descriptor doesn't expose an
    // FPR pool (the legacy contract). When the pool is non-empty, the
    // allocator runs a class-split linear scan: GPR pieces draw from
    // target.allocatable (the GPR pool), FPR pieces draw from
    // target.allocatable_fpr (the FPR pool). The two pools are independent
    // — a GPR-piece cannot evict an FPR-piece, and vice versa — so a
    // float-heavy workload that saturates the FPR pool still leaves the
    // GPR pool untouched for the loop-IV integers.
    const std::uint32_t available_fpr = target.allocatable_fprs;
    if (available_fpr > 0) {
        out.assignment_fpr.assign(n_nodes + 1, -1);
    } else {
        // Make explicit that the FPR pool is disabled — the codegen reads
        // a non-existent entry as "spilled to home" via the same `>= 0`
        // check that the GPR pool uses. An empty vector safely returns
        // false for any vreg-size check; we keep it empty so a future
        // caller iterating assignment_fpr[] doesn't pay an n_nodes-sized
        // zero-initialization cost when no FPR pool exists.
    }
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
        // LSRA->XMM extension: a vreg's reg class is read from its MIR
        // node's `rc` field. GPR-class vregs draw from the GPR pool;
        // FPR-class vregs (FP arithmetic: FADD/FSUB/FMUL/FDIV, FCONSTri,
        // FMOV*, GUARD_FLOAT) draw from the FPR pool. The regalloc's two
        // pools stay independent — a piece's reg class never changes
        // across a loop-header split (it's a property of the vreg, not
        // the position). The same is_pyobject / loop_depth invariants
        // apply in both pools.
        const bool piece_is_fpr =
            (available_fpr > 0) &&
            (mir.node(li.vreg).rc == MachineRegClass::FPR);
        if (li.end <= li.start) {
            Piece p{};
            p.vreg = li.vreg;
            p.start = li.start;
            p.end = li.end;
            p.loop_depth = li.loop_depth;
            p.is_pyobject = li.is_pyobject;
            p.is_fpr = piece_is_fpr;
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
                piece.is_fpr = piece_is_fpr;
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
        tail.is_fpr = piece_is_fpr;
        tail.parent_interval = i;
        pieces.push_back(tail);
        (void)split_any;
    }

    // ---- 4. priority linear scan --------------------------------------------------
    // Sort by start asc; for equal starts, prefer higher loop depth first
    // (hot values claim registers). Ties after that: GPR-class before FPR
    // class (no semantic effect on a single pool, but stabilizes the sort
    // across runs — the rule 34 determinism guard). The original sort used
    // a vreg tiebreak, which is fine; we add the reg-class tiebreak ahead
    // of it so a tiny int-IV and a float-IV starting at the same position
    // have a deterministic relative order.
    std::sort(pieces.begin(), pieces.end(), [](const Piece& a, const Piece& b) noexcept {
        if (a.start != b.start) return a.start < b.start;
        if (a.loop_depth != b.loop_depth) return a.loop_depth > b.loop_depth;
        if (a.is_fpr != b.is_fpr) return !a.is_fpr;
        return a.vreg < b.vreg;
    });

    const std::uint32_t available_gpr = target.allocatable_gprs;
    // LSRA->XMM extension: two independent occupancy bitmasks, one per
    // pool. A GPR-piece's `assigned` index is into target.allocatable[]
    // (the GPR pool); an FPR-piece's `assigned` index is into
    // target.allocatable_fpr[] (the FPR pool). The active list carries
    // both kinds mixed; the per-piece `is_fpr` flag routes expiry/eviction
    // to the right bitmask. The same victim-selection heuristic (lowest
    // loop_depth, then furthest end) is applied per-pool: a high-pressure
    // FPR workload evicts from the FPR pool only, never stealing a GPR
    // that's holding a loop-IV integer.
    stdx::small_vector<Piece*, 16> active;   // currently holding a register
    std::uint32_t in_use_gpr = 0;            // bitmask of allocated GPR slots
    std::uint32_t in_use_fpr = 0;            // bitmask of allocated FPR slots

    auto free_gpr = [&]() noexcept -> std::int32_t {
        for (std::uint32_t r = 0; r < available_gpr; ++r) {
            if (!(in_use_gpr & (1u << r))) {
                in_use_gpr |= 1u << r;
                return static_cast<std::int32_t>(r);
            }
        }
        return -1;
    };
    auto free_fpr = [&]() noexcept -> std::int32_t {
        for (std::uint32_t r = 0; r < available_fpr; ++r) {
            if (!(in_use_fpr & (1u << r))) {
                in_use_fpr |= 1u << r;
                return static_cast<std::int32_t>(r);
            }
        }
        return -1;
    };

    std::uint32_t pressure = 0;
    for (Piece& p : pieces) {
        // expire: release slots held by pieces whose lifetime ended before
        // this piece's start. The pool the piece drew from is encoded on
        // the piece itself via `is_fpr`.
        for (std::size_t k = 0; k < active.size();) {
            if (active[k]->end < p.start) {
                std::uint32_t& in_use = active[k]->is_fpr ? in_use_fpr : in_use_gpr;
                const std::uint32_t pool_sz = active[k]->is_fpr ? available_fpr
                                                                 : available_gpr;
                for (std::uint32_t r = 0; r < pool_sz; ++r) {
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

        // LSRA->XMM: pick the right pool for this piece. An FPR pool size
        // of zero (the legacy contract when the descriptor doesn't expose
        // one) forces FPR-class pieces onto the spill-to-home path —
        // equivalent to the always-spill discipline the GPR cache
        // retired for GPR-class vregs. Backward compatible.
        const bool use_fpr_pool = p.is_fpr && available_fpr > 0;
        std::int32_t reg = use_fpr_pool ? free_fpr() : free_gpr();
        if (reg < 0) {
            // Evict: lowest loop depth, then furthest end. Same heuristic
            // as before but constrained to the SAME pool as p (a GPR-piece
            // cannot evict an FPR-piece, and vice versa — cross-pool
            // eviction would silently steal a register from the wrong
            // register file, which the codegen's per-class cache model
            // doesn't track).
            Piece* victim = nullptr;
            for (Piece* a : active) {
                if (a->is_fpr != p.is_fpr) continue;   // different pool
                if (!victim) { victim = a; continue; }
                if (a->loop_depth < victim->loop_depth) { victim = a; continue; }
                if (a->loop_depth == victim->loop_depth && a->end > victim->end) victim = a;
            }
            if (victim && (victim->loop_depth < p.loop_depth ||
                           (victim->loop_depth == p.loop_depth && victim->end > p.end))) {
                // Mark the whole vreg spilled. The codegen's resolve
                // falls back to home; the legacy contract.
                if (victim->is_fpr) {
                    intervals_out[victim->parent_interval].phys_reg = -1;
                    intervals_out[victim->parent_interval].assigned = true;
                    // (no INCREF/DECREF traffic for spilled FP vregs; only
                    // PyObject-class spills do — and FP vregs are not
                    // PyObjects by construction.)
                } else {
                    intervals_out[victim->parent_interval].phys_reg = -1;
                    intervals_out[victim->parent_interval].assigned = true;
                    if (victim->is_pyobject) ++out.spills;   // INCREF/DECREF boundary
                }
                std::uint32_t& in_use = victim->is_fpr ? in_use_fpr : in_use_gpr;
                const std::uint32_t pool_sz = victim->is_fpr ? available_fpr
                                                              : available_gpr;
                for (std::uint32_t r = 0; r < pool_sz; ++r) {
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
                reg = use_fpr_pool ? free_fpr() : free_gpr();
                // victim reload happens at its next piece boundary (codegen
                // inserts MOV from home slot).
            } else {
                // p itself spills: its vreg reads/writes go through home.
                if (p.is_fpr) {
                    intervals_out[p.parent_interval].phys_reg = -1;
                    intervals_out[p.parent_interval].assigned = true;
                } else {
                    intervals_out[p.parent_interval].phys_reg = -1;
                    intervals_out[p.parent_interval].assigned = true;
                    if (p.is_pyobject) ++out.spills;
                }
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

    // Map piece assignments onto vreg assignment.
    //
    // Soundness: if ANY piece of a vreg spilled, the whole vreg must
    // read/write through home slots — otherwise the codegen (which
    // queries a single physreg per vreg) would use a stale register
    // value at the boundary where the piece was evicted. The previous
    // version overwrote `out.assignment[p.vreg]` with each piece's
    // assignment, so the LAST piece won and earlier spills were
    // silently discarded. That was unsound.
    //
    // The current rule: a vreg gets a physreg ONLY if every piece of
    // that vreg was assigned the SAME physreg. If pieces disagree (e.g.
    // piece 1 in R1, piece 2 in R2 because of eviction across a loop
    // header), the vreg spills to home. The codegen then reads/writes
    // home slots uniformly — no per-piece move insertion required.
    //
    // LSRA->XMM extension: the rule now applies to BOTH pools in
    // parallel. A GPR-class vreg goes into out.assignment[]; an FPR-class
    // vreg goes into out.assignment_fpr[]. The two arrays are sized in
    // lockstep at the top of the function (the FPR array stays empty
    // when no FPR pool exists — the legacy contract). The cross-class
    // spill check (a vreg with mixed GPR/FPR pieces) is impossible
    // because `is_fpr` is a property of the vreg's MIR `rc`, constant
    // across all pieces.
    stdx::flat_map<std::uint32_t, std::int32_t, 64> first_assignment;
    stdx::flat_map<std::uint32_t, bool, 64> any_spill;
    for (const Piece& p : pieces) {
        if (p.assigned < 0) {
            any_spill.insert_or_assign(p.vreg, true);
        } else {
            const std::int32_t* cur = first_assignment.get(p.vreg);
            if (!cur) {
                first_assignment.insert(p.vreg, p.assigned);
            } else if (*cur != p.assigned) {
                // Pieces disagree: spill the whole vreg.
                any_spill.insert_or_assign(p.vreg, true);
            }
        }
    }
    for (const Piece& p : pieces) {
        const bool* spill = any_spill.get(p.vreg);
        if (spill && *spill) {
            // Spill to home: -1 in whichever pool the vreg belongs to.
            if (p.is_fpr) {
                out.assignment_fpr[p.vreg] = -1;
            } else {
                out.assignment[p.vreg] = -1;
            }
        } else {
            const std::int32_t* a = first_assignment.get(p.vreg);
            if (a) {
                if (p.is_fpr) {
                    out.assignment_fpr[p.vreg] = *a;
                } else {
                    out.assignment[p.vreg] = *a;
                }
            }
        }
    }
    // Spilled vregs keep assignment -1; their operands read/write home slots.
    const std::uint32_t max_pool = available_gpr > available_fpr ? available_gpr
                                                                  : available_fpr;
    out.max_pressure = pressure < max_pool ? pressure : max_pool;
    return out;
}

}  // namespace abi_v1
}  // namespace vortex::backend
