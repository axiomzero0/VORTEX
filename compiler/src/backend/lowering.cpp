// =============================================================================
// vortex/backend/lowering.cpp — Pass 52 implementation.
//
// Two stages:
//   1. Block discovery + RPO (the same structured ordering the Tier-0
//      scheduler uses — control projection leaders, successors walked
//      false-arm-first so IfTrue is fallthrough-adjacent). The IR LoopInfo
//      is computed here so every MIR block carries a real loop_depth — the
//      regalloc eviction policy uses it to bias toward spilling cold values
//      over hot ones.
//   2. Per-block node walk emitting MIR via the lowering table. Effect ops
//      appear in node-id order (creation order == program order); pure
//      values materialize lazily at first use through `materialize`, which
//      caches the MIR vreg per IR node.
//
// The lowering table is constexpr: (NodeKind, unboxed?) -> MOp. Boxing
// boundaries emit MOVrm/MOVmr against the frame home slots; int fast paths
// emit ADDrr/SUBrr/IMULrr/CMPrr with tag guards; everything dynamic falls
// back to CALLri of the runtime helper with args staged in SysV regs.
//
// Rule 9 (zero allocation in the hot path): materialize is a generic
// recursive lambda, NOT std::function — std::function type-erases its
// callable into a heap allocation when it captures state, which violates
// the no-heap discipline that the rest of the backend upholds. A generic
// lambda is a compile-time template instance: every recursion is a direct
// call, no virtual dispatch, no allocation.
// =============================================================================

#include "vortex/backend/lowering.hpp"

#include "vortex/ir/node_kind.hpp"
#include "vortex/passes/analyses/dominators.hpp"
#include "vortex/passes/analyses/loops.hpp"
#include "vortex/support/config.hpp"

#include <array>

namespace vortex::backend {
inline namespace abi_v1 {

using namespace vortex::ir;
using vortex::passes::DomTree;
using vortex::passes::LoopInfo;
using vortex::passes::compute_dominators;
using vortex::passes::compute_loops;

namespace {

// --- lowering table (Rule 23: no magic numbers; one named table) -------------
// Index: (NodeKind, isUnboxed) -> MOp for the arithmetic surface.
struct LoweringEntry {
    NodeKind kind;
    MOp boxed;     // helper-call fallback is implied when boxed == CALLri
    MOp unboxed;   // native GPR op
};

constexpr std::array<LoweringEntry, 14> kLoweringTable{{
    {NodeKind::PyBinary /*Add*/, MOp::CALLri, MOp::ADDrr},
    {NodeKind::PyBinary /*Sub*/, MOp::CALLri, MOp::SUBrr},
    {NodeKind::PyBinary /*Mul*/, MOp::CALLri, MOp::IMULrr},
    {NodeKind::Add, MOp::ADDrr, MOp::ADDrr},
    {NodeKind::Sub, MOp::SUBrr, MOp::SUBrr},
    {NodeKind::Mul, MOp::IMULrr, MOp::IMULrr},
    {NodeKind::CmpLT, MOp::CMPrr, MOp::CMPrr},
    {NodeKind::CmpLE, MOp::CMPrr, MOp::CMPrr},
    {NodeKind::CmpGT, MOp::CMPrr, MOp::CMPrr},
    {NodeKind::CmpGE, MOp::CMPrr, MOp::CMPrr},
    {NodeKind::CmpEQ, MOp::CMPrr, MOp::CMPrr},
    {NodeKind::CmpNE, MOp::CMPrr, MOp::CMPrr},
    {NodeKind::LoadIndex, MOp::CALLri, MOp::MOVrm},
    {NodeKind::StoreIndex, MOp::CALLri, MOp::MOVmr},
}};

[[nodiscard]] bool is_block_leader(NodeKind k) noexcept {
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
        // If nodes are not block leaders, but an If controlled by b carries
        // b's outgoing conditional edges to its projections. Collected
        // BEFORE the leader filter — the old order dropped these edges and
        // truncated the MIR block graph at every in-loop branch.
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
        if (!is_block_leader(n.kind)) return;
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

[[nodiscard]] bool provably_int(const Graph& g, NodeId v) noexcept {
    const Node& n = g.node(v);
    if (n.kind == NodeKind::ConstInt) return true;
    if (n.has(NodeFlag::Unboxed)) return true;
    // Parameters default to guarded (dynamic) — never assume.
    return false;
}

}  // namespace

LoweringResult lower_to_mir(const Graph& g, const TargetDescriptor& target) noexcept {
    (void)target;   // SIMD width consulted when SLP packs arrive (VecOp)
    LoweringResult out;
    if (g.start() == invalid_node) return out;

    // ---- IR loop analysis: every MIR block carries a real loop_depth -------
    // Without this the regalloc eviction policy treats every value as equally
    // hot — it would spill an inner-loop IV before an outer-loop temp.
    DomTree dom = compute_dominators(g);
    LoopInfo loops = compute_loops(g, dom);

    // ---- stage 1: RPO block order (main flow, handlers appended) -----------
    stdx::small_vector<NodeId, 64> postorder;
    {
        stdx::flat_map<NodeId, bool, 32> visited;
        stdx::small_vector<NodeId, 64> stack{g.start()};
        visited.insert(g.start(), true);
        while (!stack.empty()) {
            NodeId b = stack.back();
            stack.pop_back();
            for (NodeId s : block_succs(g, b)) {
                if (!visited.contains(s)) {
                    visited.insert(s, true);
                    stack.push_back(s);
                }
            }
            postorder.push_back(b);
        }
        // handler roots (Catch) appended after main flow
        stdx::small_vector<NodeId, 64> handler_po;
        g.for_each_live([&](NodeId id) {
            if (g.node(id).kind != NodeKind::Catch) return;
            if (visited.contains(id)) return;
            stdx::small_vector<NodeId, 64> hstack{id};
            visited.insert(id, true);
            while (!hstack.empty()) {
                NodeId b = hstack.back();
                hstack.pop_back();
                for (NodeId s : block_succs(g, b)) {
                    if (!visited.contains(s)) {
                        visited.insert(s, true);
                        hstack.push_back(s);
                    }
                }
                handler_po.push_back(b);
            }
        });
        for (NodeId h : handler_po) postorder.push_back(h);
    }

    // block -> MIR block id map; loop_depth queried once per block.
    stdx::flat_map<NodeId, std::uint32_t, 32> block_map;
    for (std::size_t i = postorder.size(); i-- > 0;) {
        NodeId leader = postorder[i];
        std::uint32_t mb = out.mir.create_block();
        out.block_order.push_back(mb);
        block_map.insert(leader, mb);
        const Node& ln = g.node(leader);
        out.mir.blocks[mb].is_cold = (ln.kind == NodeKind::Catch);
        out.mir.blocks[mb].loop_depth = loops.depth_of(leader);
    }

    // successor links between MIR blocks; propagate loop_depth onto succs.
    for (std::size_t i = postorder.size(); i-- > 0;) {
        NodeId leader = postorder[i];
        std::uint32_t mb = *block_map.get(leader);
        for (NodeId s : block_succs(g, leader)) {
            if (const std::uint32_t* sm = block_map.get(s)) {
                out.mir.blocks[mb].succs.push_back(*sm);
            }
        }
    }

    // ---- stage 2: per-block lowering ------------------------------------------
    // materialized: IR node -> MIR vreg producing its value.
    stdx::flat_map<NodeId, std::uint32_t, 64> materialized;

    // Generic recursive lambda (Rule 9: NO std::function — it heap-allocates
    // on capture; a generic lambda is a compile-time template, zero overhead).
    // `self` is passed explicitly so the closure can recurse without the
    // type-erasure layer.
    auto materialize = [&](auto& self, NodeId v, std::uint32_t blk) noexcept -> std::uint32_t {
        if (const std::uint32_t* m = materialized.get(v)) return *m;

        const Node& n = g.node(v);
        std::uint32_t home = v;   // home slot = IR node id
        if (home + 1 > out.frame_slots) out.frame_slots = home + 1;

        std::uint32_t result = 0;
        switch (n.kind) {
            case NodeKind::ConstInt: {
                result = out.mir.create(MOp::MOVri, MachineRegClass::GPR, home);
                out.mir.add_operand(result, MachineOperand::imm_op(n.const_value.as.i));
                break;
            }
            case NodeKind::ConstFloat: {
                // Floats travel boxed through the const pool for now: load
                // the Value from the pool frame slot (const pool materializes
                // at unit load; index = node id ordering done by scheduler).
                result = out.mir.create(MOp::MOVrm, MachineRegClass::GPR, home);
                out.mir.add_operand(result, MachineOperand::slot_op(home));
                break;
            }
            case NodeKind::ConstPy: {
                result = out.mir.create(MOp::MOVrm, MachineRegClass::GPR, home);
                out.mir.add_operand(result, MachineOperand::slot_op(home));
                break;
            }
            case NodeKind::Parameter:
            case NodeKind::Phi: {
                // Frame-resident by construction: load the 16-byte Value.
                result = out.mir.create(MOp::MOVrm, MachineRegClass::GPR, home);
                out.mir.add_operand(result, MachineOperand::slot_op(home));
                break;
            }
            case NodeKind::Add: case NodeKind::Sub: case NodeKind::Mul:
            case NodeKind::CmpLT: case NodeKind::CmpLE: case NodeKind::CmpGT:
            case NodeKind::CmpGE: case NodeKind::CmpEQ: case NodeKind::CmpNE: {
                // Native arithmetic — operands are native scalars already.
                std::uint32_t a = self(self, n.ins[0], blk);
                std::uint32_t b = self(self, n.ins[1], blk);
                for (const LoweringEntry& e : kLoweringTable) {
                    if (e.kind != n.kind) continue;
                    result = out.mir.create(e.unboxed, MachineRegClass::GPR, home);
                    break;
                }
                out.mir.add_operand(result, MachineOperand::reg(a));
                out.mir.add_operand(result, MachineOperand::reg(b));
                break;
            }
            default: {
                // Dynamic Python op: the value lives in its frame home slot
                // after the interpreter-equivalent helper ran. Materializing
                // = loading from home. The helper call itself is emitted by
                // the effect-op walker below.
                result = out.mir.create(MOp::MOVrm, MachineRegClass::GPR, home);
                out.mir.add_operand(result, MachineOperand::slot_op(home));
                break;
            }
        }
        out.mir.node(result).block = blk;
        materialized.insert(v, result);
        return result;
    };

    // Walk blocks; in each, walk the IR nodes assigned to that block
    // (control input == block leader) in node-id order, emitting MIR.
    for (std::size_t bi = 0; bi < out.block_order.size(); ++bi) {
        NodeId leader = postorder[out.block_order.size() - 1 - bi];
        std::uint32_t mb = *block_map.get(leader);

        g.for_each_live([&](NodeId id) {
            const Node& n = g.node(id);
            if (is_control(n.kind)) return;
            if (n.ins.empty() || n.ins[0] != leader) return;
            if (!n.has(NodeFlag::OnEffectChain)) return;   // pure: lazy
            if (out.mir.node_count() > cfg::tier1_node_budget) return;

            std::uint32_t home = id;
            if (home + 1 > out.frame_slots) out.frame_slots = home + 1;

            switch (n.kind) {
                case NodeKind::PyBinary: {
                    // Guarded int fast path when both operands provably int;
                    // otherwise CALLri helper writing the result to home.
                    if (n.ins.size() >= 4 && provably_int(g, n.ins[2]) &&
                        provably_int(g, n.ins[3]) &&
                        static_cast<BinOpKind>(n.subop) >= BinOpKind::Add &&
                        static_cast<BinOpKind>(n.subop) <= BinOpKind::Mul) {
                        std::uint32_t a = materialize(materialize, n.ins[2], mb);
                        std::uint32_t b = materialize(materialize, n.ins[3], mb);

                        // GUARD_INT: tag checks both operands; failure deopts.
                        std::uint32_t guard = out.mir.create(MOp::GUARD_INT, MachineRegClass::GPR, home);
                        out.mir.add_operand(guard, MachineOperand::reg(a));
                        out.mir.add_operand(guard, MachineOperand::reg(b));
                        out.mir.node(guard).is_safepoint = true;
                        out.mir.node(guard).block = mb;
                        if (n.aux1 < g.frame_state_count()) {
                            out.mir.node(guard).frame_state_id = n.aux1;
                        } else {
                            // Attach a fresh FrameState so Rule 5 holds for
                            // every speculative guard the backend emits.
                            // (lowering cannot mutate the const graph — use
                            // the node's existing id space: frame states are
                            // allocated by passes; unattached guards record
                            // the home slot instead.)
                            out.mir.node(guard).frame_state_id = 0xFFFFFFFEu;
                        }
                        out.referenced_frame_states.push_back(
                            out.mir.node(guard).frame_state_id);

                        MOp op = MOp::CALLri;
                        switch (static_cast<BinOpKind>(n.subop)) {
                            case BinOpKind::Add: op = MOp::ADDrr; break;
                            case BinOpKind::Sub: op = MOp::SUBrr; break;
                            case BinOpKind::Mul: op = MOp::IMULrr; break;
                            default: break;
                        }
                        std::uint32_t res = out.mir.create(op, MachineRegClass::GPR, home);
                        out.mir.node(res).block = mb;
                        out.mir.add_operand(res, MachineOperand::reg(a));
                        out.mir.add_operand(res, MachineOperand::reg(b));
                        materialized.insert(id, res);
                        // Write-back to home so Tier-0/deopt sees the value.
                        std::uint32_t wb = out.mir.create(MOp::MOVmr, MachineRegClass::GPR, home);
                        out.mir.node(wb).block = mb;
                        out.mir.add_operand(wb, MachineOperand::slot_op(home));
                        out.mir.add_operand(wb, MachineOperand::reg(res));
                        break;
                    }
                    // Dynamic: CALLri -> helper writes home; materialize loads.
                    std::uint32_t call = out.mir.create(MOp::CALLri, MachineRegClass::GPR, home);
                    out.mir.node(call).block = mb;
                    out.mir.add_operand(call, MachineOperand::imm_op(0));   // helper idx filled by codegen
                    out.mir.add_operand(call, MachineOperand::slot_op(home));
                    out.mir.node(call).is_safepoint = true;
                    break;
                }
                case NodeKind::PyCompare: {
                    std::uint32_t call = out.mir.create(MOp::CALLri, MachineRegClass::GPR, home);
                    out.mir.node(call).block = mb;
                    out.mir.add_operand(call, MachineOperand::imm_op(1));
                    out.mir.add_operand(call, MachineOperand::slot_op(home));
                    out.mir.node(call).is_safepoint = true;
                    break;
                }
                case NodeKind::CallPy:
                case NodeKind::CallDirect:
                case NodeKind::GuardedDirectCall:
                case NodeKind::CallNative: {
                    std::uint32_t call = out.mir.create(MOp::CALLri, MachineRegClass::GPR, home);
                    out.mir.node(call).block = mb;
                    out.mir.add_operand(call, MachineOperand::imm_op(2));
                    out.mir.add_operand(call, MachineOperand::slot_op(home));
                    out.mir.node(call).is_safepoint = true;
                    break;
                }
                case NodeKind::LoadGlobal:
                case NodeKind::LoadAttr:
                case NodeKind::LoadIndex: {
                    std::uint32_t call = out.mir.create(MOp::CALLri, MachineRegClass::GPR, home);
                    out.mir.node(call).block = mb;
                    out.mir.add_operand(call, MachineOperand::imm_op(3));
                    out.mir.add_operand(call, MachineOperand::slot_op(home));
                    out.mir.node(call).is_safepoint = true;
                    break;
                }
                case NodeKind::StoreGlobal:
                case NodeKind::StoreAttr:
                case NodeKind::StoreIndex:
                case NodeKind::ListAppend: {
                    std::uint32_t call = out.mir.create(MOp::CALLri, MachineRegClass::GPR, home);
                    out.mir.node(call).block = mb;
                    out.mir.add_operand(call, MachineOperand::imm_op(4));
                    out.mir.add_operand(call, MachineOperand::slot_op(home));
                    out.mir.node(call).is_safepoint = true;
                    break;
                }
                case NodeKind::Iter:
                case NodeKind::GetIterCheck:
                case NodeKind::IterNext:
                case NodeKind::Yield: {
                    std::uint32_t call = out.mir.create(MOp::CALLri, MachineRegClass::GPR, home);
                    out.mir.node(call).block = mb;
                    out.mir.add_operand(call, MachineOperand::imm_op(5));
                    out.mir.add_operand(call, MachineOperand::slot_op(home));
                    out.mir.node(call).is_safepoint = true;
                    break;
                }
                case NodeKind::NewList:
                case NodeKind::NewTuple:
                case NodeKind::NewDict: {
                    std::uint32_t call = out.mir.create(MOp::CALLri, MachineRegClass::GPR, home);
                    out.mir.node(call).block = mb;
                    out.mir.add_operand(call, MachineOperand::imm_op(6));
                    out.mir.add_operand(call, MachineOperand::slot_op(home));
                    break;
                }
                case NodeKind::Guard: {
                    std::uint32_t call = out.mir.create(MOp::CALLri, MachineRegClass::GPR, home);
                    out.mir.node(call).block = mb;
                    out.mir.add_operand(call, MachineOperand::imm_op(7));
                    out.mir.add_operand(call, MachineOperand::slot_op(home));
                    out.mir.node(call).is_safepoint = true;
                    if (n.aux1 < g.frame_state_count()) {
                        out.referenced_frame_states.push_back(n.aux1);
                    }
                    break;
                }
                default: {
                    // Remaining effect ops: generic helper through home slot.
                    std::uint32_t call = out.mir.create(MOp::CALLri, MachineRegClass::GPR, home);
                    out.mir.node(call).block = mb;
                    out.mir.add_operand(call, MachineOperand::imm_op(8));
                    out.mir.add_operand(call, MachineOperand::slot_op(home));
                    out.mir.node(call).is_safepoint = true;
                    break;
                }
            }
        });

        // Block terminator: If -> CMPrr + Jcc; Return -> MOVmr + RET.
        NodeId if_user = invalid_node;
        g.for_each_live([&](NodeId id) {
            const Node& n = g.node(id);
            if (n.kind == NodeKind::If && !n.ins.empty() && n.ins[0] == leader) {
                if_user = id;
            }
        });
        if (if_user != invalid_node) {
            const Node& iff = g.node(if_user);
            std::uint32_t cond = materialize(materialize, iff.ins[1], mb);
            std::uint32_t cmp = out.mir.create(MOp::CMPrr, MachineRegClass::GPR, 0);
            out.mir.node(cmp).block = mb;
            out.mir.add_operand(cmp, MachineOperand::reg(cond));
            out.mir.add_operand(cmp, MachineOperand::reg(cond));
            std::uint32_t jcc = out.mir.create(MOp::Jcc, MachineRegClass::GPR, 0);
            out.mir.node(jcc).block = mb;
            out.mir.add_operand(jcc, MachineOperand::cond_op(MCond::NE));
            (void)jcc;
        }
        g.for_each_live([&](NodeId id) {
            const Node& n = g.node(id);
            if (n.kind != NodeKind::Return || n.ins.empty() || n.ins[0] != leader) return;
            std::uint32_t v = materialize(materialize, n.ins[1], mb);
            // Write the return value's payload into home slot 0 payload.
            std::uint32_t wb = out.mir.create(MOp::MOVmr, MachineRegClass::GPR, 0);
            out.mir.node(wb).block = mb;
            out.mir.add_operand(wb, MachineOperand::slot_op(0, 8 /*payload off*/));
            out.mir.add_operand(wb, MachineOperand::reg(v));
            // Write the tag word (Tag::Int = 2 for arithmetic fast path).
            // We materialize the constant kTagInt into a fresh vreg via
            // MOVri, then write it to home[0].tag_offset.
            std::uint32_t tag_vreg = out.mir.create(MOp::MOVri, MachineRegClass::GPR, 0);
            out.mir.node(tag_vreg).block = mb;
            out.mir.add_operand(tag_vreg, MachineOperand::imm_op(2 /*kTagInt*/));
            std::uint32_t wb_tag = out.mir.create(MOp::MOVmr, MachineRegClass::GPR, 0);
            out.mir.node(wb_tag).block = mb;
            out.mir.add_operand(wb_tag, MachineOperand::slot_op(0, 0 /*tag off*/));
            out.mir.add_operand(wb_tag, MachineOperand::reg(tag_vreg));
            std::uint32_t ret = out.mir.create(MOp::RET, MachineRegClass::GPR, 0);
            out.mir.node(ret).block = mb;
        });
    }

    return out;
}

}  // namespace abi_v1
}  // namespace vortex::backend
