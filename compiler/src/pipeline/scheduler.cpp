// =============================================================================
// vortex/pipeline/scheduler.cpp — IR -> Tier-0 bytecode linearization.
//
// Algorithm:
//   1. Blocks are led by control projections: Start, Region, Loop, Catch,
//      IfTrue, IfFalse, Jump (the try-marker).
//   2. Block order = DFS from Start through control successors (true arm
//      first), then DFS from each unreachable Catch (exception handlers).
//      This yields a valid linearization with fallthrough for the common
//      case and explicit JUMPs otherwise.
//   3. Block body: effect ops whose control input is the block leader, in
//      node-id (creation) order; pure values schedule lazily at first use.
//   4. Terminators: If -> JUMP_IF_FALSE (fixup to IfFalse block); Return /
//      Throw end blocks; loop backedges (Loop.ins[1]) emit phi moves + JUMP;
//      merge edges (Region/Loop/Catch preds) emit phi moves + JUMP unless
//      fallthrough. Parallel-move cycles go through scratch registers.
//   5. Try ranges: [block(Catch.aux0 marker).start, Catch block start) ->
//      handler at the Catch block start. Nested handlers nest naturally.
//
// Phi discipline (the nested-loop lesson): phis are REGISTERS, never
// computations. Their inputs are materialized at the PREDECESSOR edge
// (terminator MOVE emission + materialize_phi_source), never by a
// recursive operand walk at the header — walking them hoists latch
// computations into loop headers where their operands are not yet valid.
// =============================================================================

#include "vortex/pipeline/scheduler.hpp"

#include <cstring>

#include "vortex/frontend/lowering.hpp"
#include "vortex/ir/node_kind.hpp"
#include "vortex/support/symbol_table.hpp"

namespace vortex::pipeline {
inline namespace abi_v1 {

namespace {

using namespace vortex::ir;
using vortex::rt::Runtime;
using rt::Instr;
using rt::Op;

struct BlockInfo {
    NodeId leader{invalid_node};
    stdx::small_vector<Instr, 48> body{};         // instructions (body only)
    stdx::small_vector<Instr, 48> term{};         // terminator (phis + jump)
    std::uint32_t start_pc{0};
    bool visited{false};
    // fixup sites inside `term` needing block start pcs:
    stdx::small_vector<std::pair<std::size_t, NodeId>, 4> term_fixups{};
};

[[nodiscard]] bool is_block_leader(NodeKind k) noexcept {
    switch (k) {
        case NodeKind::Start:
        case NodeKind::Region:
        case NodeKind::Loop:
        case NodeKind::Catch:
        case NodeKind::IfTrue:
        case NodeKind::IfFalse:
        case NodeKind::Jump:
            return true;
        default:
            return false;
    }
}

class Scheduler {
public:
    Scheduler(const Graph& g, rt::CodeUnit& unit, const stdx::small_vector<char, 4096>& pool)
        : g_(g), unit_(unit), pool_(pool) {}

    Result<void> run() noexcept;

private:
    // --- instruction emission into current block --------------------------------
    void ins(stdx::small_vector<Instr, 48>& buf, Instr i) noexcept {
        buf.push_back(rt::Instr{static_cast<std::uint16_t>(i.op), i.dst, i.a, i.b, i.c, i.imm});
    }

    [[nodiscard]] std::uint32_t add_constant(Value v) noexcept {
        for (std::uint32_t i = 0; i < unit_.constants.size(); ++i) {
            const Value& c = unit_.constants[i];
            if (c.tag == v.tag && c.as.i == v.as.i) return i;
        }
        if (v.tag == Tag::Obj && v.as.obj) Runtime::instance().incref(v.as.obj);
        unit_.constants.push_back(v);
        return static_cast<std::uint32_t>(unit_.constants.size() - 1);
    }

    [[nodiscard]] std::uint32_t symbol_const(SymbolId sym) noexcept {
        // Namespace layer keys on symbol ints (Rule 16: no strings on the
        // hot path; runtime attr dicts use the same packing).
        return add_constant(Value::integer(sym));
    }

    BlockInfo* block(NodeId leader) noexcept {
        BlockInfo* b = blocks_.get(leader);
        if (!b) {
            blocks_.insert(leader, BlockInfo{leader, {}, {}, 0, false, {}});
            b = blocks_.get(leader);
        }
        return b;
    }

    void emit_value(NodeId id) noexcept;
    void emit_effect_op(NodeId id) noexcept;
    void emit_block(NodeId leader) noexcept;
    void emit_terminator(NodeId leader) noexcept;
    void materialize_phi_source(NodeId src) noexcept;
    void dfs_post(NodeId leader, stdx::small_vector<NodeId, 64>& postorder) noexcept;
    [[nodiscard]] stdx::small_vector<NodeId, 8> successors(NodeId leader) noexcept;

    const Graph& g_;
    rt::CodeUnit& unit_;
    const stdx::small_vector<char, 4096>& pool_;

    stdx::flat_map<NodeId, BlockInfo, 32> blocks_{};
    stdx::small_vector<NodeId, 256> scheduled_{};
    NodeId current_leader_{invalid_node};
    std::uint32_t scratch_base_{0};
    std::uint32_t max_phi_width_{0};
};

// --- successors --------------------------------------------------------------
stdx::small_vector<NodeId, 8> Scheduler::successors(NodeId leader) noexcept {
    stdx::small_vector<NodeId, 8> out;
    g_.for_each_live([&](NodeId id) {
        const Node& n = g_.node(id);
        if (!is_control(n.kind)) return;
        switch (n.kind) {
            case NodeKind::If:
                if (n.ins.size() >= 1 && n.ins[0] == leader) {
                    // Visit the FALSE arm first: reverse post-order then
                    // places the TRUE arm immediately after the branch,
                    // making fallthrough == IfTrue (matches the
                    // JUMP_IF_FALSE encoding).
                    g_.for_each_live([&](NodeId proj) {
                        const Node& p = g_.node(proj);
                        if (p.kind == NodeKind::IfFalse && !p.ins.empty() && p.ins[0] == id) {
                            out.push_back(proj);
                        }
                    });
                    g_.for_each_live([&](NodeId proj) {
                        const Node& p = g_.node(proj);
                        if (p.kind == NodeKind::IfTrue && !p.ins.empty() && p.ins[0] == id) {
                            out.push_back(proj);
                        }
                    });
                }
                return;
            case NodeKind::Start:
                return;
            default:
                break;
        }
        // Region/Loop: leader is any input. Catch blocks are NOT normal-flow
        // successors — they are exception targets reached only via unwind.
        if (n.kind == NodeKind::Catch) return;
        for (NodeId in : n.ins) {
            if (in == leader && is_block_leader(n.kind)) {
                // skip loop backedge targets (handled separately)
                out.push_back(id);
                return;
            }
        }
        // Direct control child (Jump chains, projections of projections).
        // Catch handlers are exception targets, never normal successors.
        if (!n.ins.empty() && n.ins[0] == leader && is_block_leader(n.kind) &&
            n.kind != NodeKind::IfTrue && n.kind != NodeKind::IfFalse &&
            n.kind != NodeKind::Catch) {
            out.push_back(id);
        }
    });
    return out;
}

void Scheduler::dfs_post(NodeId leader, stdx::small_vector<NodeId, 64>& postorder) noexcept {
    BlockInfo* b = block(leader);
    if (b->visited) return;
    b->visited = true;
    for (NodeId succ : successors(leader)) {
        dfs_post(succ, postorder);
    }
    postorder.push_back(leader);
}

// --- value scheduling -----------------------------------------------------------
void Scheduler::emit_value(NodeId id) noexcept {
    if (id == invalid_node) return;
    const Node& n = g_.node(id);
    if (n.has(NodeFlag::Dead)) return;
    // Constants are NOT cross-block deduped: a const used in multiple
    // blocks re-emits per block (idempotent LOAD_CONST) so handler blocks
    // materialize their own operands (the try/except None bug).
    bool is_const = n.kind == NodeKind::ConstInt || n.kind == NodeKind::ConstFloat ||
                    n.kind == NodeKind::ConstPy;
    if (!is_const) {
        for (NodeId s : scheduled_) {
            if (s == id) return;
        }
        scheduled_.push_back(id);
    }

    // Phis are REGISTERS, not computations. Their inputs are edge-specific
    // (initial from the entry edge, carried value from the latch edge) and
    // are materialized by the predecessor terminators' MOVE emission.
    // Walking them here would hoist LATCH computations into the loop
    // header, where their operands are not yet valid — the nested-while
    // miscompile: `i = i + 1` (a latch effect) was emitted at the OUTER
    // loop head reading the INNER loop's phi register before the inner
    // loop ever executed. Single loops only survived because the hoisted
    // op happened to read the (valid) outer phi.
    if (n.kind == NodeKind::Phi || n.kind == NodeKind::EffectPhi) return;

    // schedule data operands first
    std::uint32_t start = 0;
    if (n.has(NodeFlag::OnEffectChain) || is_memory(n.kind)) start = 2;
    std::uint32_t n_inputs = n.ins.size();
    for (std::uint32_t i = start; i < n_inputs; ++i) {
        const Node& in = g_.node(n.ins[i]);
        if (is_control(in.kind)) continue;
        emit_value(n.ins[i]);
    }

    BlockInfo* b = block(current_leader_);
    Instr i{};
    i.dst = static_cast<std::uint16_t>(id);
    switch (n.kind) {
        case NodeKind::ConstInt:
        case NodeKind::ConstFloat: {
            i.op = static_cast<std::uint16_t>(Op::LOAD_CONST);
            i.imm = add_constant(n.const_value);
            ins(b->body, i);
            return;
        }
        case NodeKind::ConstPy: {
            i.op = static_cast<std::uint16_t>(Op::LOAD_CONST);
            // String literals carry (aux0=offset, aux1=length) into the
            // module pool; Bool/None literals carry tag-only payloads.
            bool is_pool_string = n.aux0 != 0xFFFF'FFFF && n.aux1 != 0xFFFF'FFFF &&
                                  n.symbol == 0xFFFF'FFFF &&
                                  n.const_value.tag == Tag::None;
            if (n.symbol != 0xFFFF'FFFF && n.symbol != 0) {
                i.imm = symbol_const(n.symbol);
            } else if (n.const_value.tag == Tag::Obj) {
                i.imm = add_constant(n.const_value);
            } else if (n.const_value.tag == Tag::Bool) {
                i.imm = add_constant(n.const_value);
            } else if (n.const_value.tag == Tag::None && !is_pool_string) {
                i.imm = add_constant(n.const_value);
            } else {
                std::string_view text(pool_.data() + n.aux0, n.aux1);
                auto* s = Runtime::instance().new_str(text);
                i.imm = add_constant(Value::object(reinterpret_cast<PyObj*>(s)));
            }
            ins(b->body, i);
            return;
        }
        case NodeKind::Parameter:
        case NodeKind::Phi:
        case NodeKind::EffectPhi:
        case NodeKind::Guard:
        case NodeKind::DeoptBarrier:
            return;
        case NodeKind::PyBinary:
            // Py ops carry (control, effect, lhs, rhs) since the effectful
            // rework — data operands start at index 2.
            i.op = static_cast<std::uint16_t>(Op::PY_BINOP);
            i.a = static_cast<std::uint16_t>(n.ins[2]);
            i.b = static_cast<std::uint16_t>(n.ins[3]);
            i.imm = n.subop;
            ins(b->body, i);
            return;
        case NodeKind::PyUnary:
            i.op = static_cast<std::uint16_t>(Op::PY_UNOP);
            i.a = static_cast<std::uint16_t>(n.ins[2]);
            i.imm = n.subop;
            ins(b->body, i);
            return;
        case NodeKind::PyCompare:
            i.op = static_cast<std::uint16_t>(Op::PY_CMP);
            i.a = static_cast<std::uint16_t>(n.ins[2]);
            i.b = static_cast<std::uint16_t>(n.ins[3]);
            i.imm = n.subop;
            ins(b->body, i);
            return;
        // native arithmetic on the generic Tier-0 surface
        // IBE-1 fix: the previous code used a positional lookup
        //   idx = n.kind - NodeKind::Add
        // and indexed a flat table — but NodeKind has Neg in the middle
        // (Add, Sub, Mul, Div, Mod, Pow, Neg, BitAnd, ...), so every
        // entry past Pow was off by one: BitAnd picked up BitOr's slot,
        // BitOr picked up BitXor's, ..., and Shr (idx 11) read past
        // the table end and was silently dropped. Use an explicit switch
        // so the mapping is unambiguous.
        case NodeKind::Add: case NodeKind::Sub: case NodeKind::Mul:
        case NodeKind::Div: case NodeKind::Mod: case NodeKind::Pow:
        case NodeKind::BitAnd: case NodeKind::BitOr: case NodeKind::BitXor:
        case NodeKind::Shl: case NodeKind::Shr: {
            BinOpKind bop;
            switch (n.kind) {
                case NodeKind::Add:    bop = BinOpKind::Add; break;
                case NodeKind::Sub:    bop = BinOpKind::Sub; break;
                case NodeKind::Mul:    bop = BinOpKind::Mul; break;
                case NodeKind::Div:    bop = BinOpKind::TrueDiv; break;
                case NodeKind::Mod:    bop = BinOpKind::Mod; break;
                case NodeKind::Pow:    bop = BinOpKind::Pow; break;
                case NodeKind::BitAnd: bop = BinOpKind::BitAnd; break;
                case NodeKind::BitOr:  bop = BinOpKind::BitOr; break;
                case NodeKind::BitXor: bop = BinOpKind::BitXor; break;
                case NodeKind::Shl:    bop = BinOpKind::LShift; break;
                case NodeKind::Shr:    bop = BinOpKind::RShift; break;
                default: return;   // unreachable
            }
            i.op = static_cast<std::uint16_t>(Op::PY_BINOP);
            i.a = static_cast<std::uint16_t>(n.ins[0]);
            i.b = static_cast<std::uint16_t>(n.ins[1]);
            i.imm = static_cast<std::uint32_t>(bop);
            ins(b->body, i);
            return;
        }
        case NodeKind::CmpLT: case NodeKind::CmpLE: case NodeKind::CmpGT:
        case NodeKind::CmpGE: case NodeKind::CmpEQ: case NodeKind::CmpNE: {
            static const CmpOpKind table[] = {CmpOpKind::LT, CmpOpKind::LE, CmpOpKind::GT,
                                              CmpOpKind::GE, CmpOpKind::EQ, CmpOpKind::NE};
            std::uint32_t kind_base = static_cast<std::uint32_t>(NodeKind::CmpLT);
            std::uint32_t idx = static_cast<std::uint32_t>(n.kind) - kind_base;
            i.op = static_cast<std::uint16_t>(Op::PY_CMP);
            i.a = static_cast<std::uint16_t>(n.ins[0]);
            i.b = static_cast<std::uint16_t>(n.ins[1]);
            i.imm = static_cast<std::uint32_t>(table[idx]);
            ins(b->body, i);
            return;
        }
        case NodeKind::Unbox: case NodeKind::Box:
        case NodeKind::SExt: case NodeKind::ZExt: case NodeKind::Trunc:
        case NodeKind::BitCast: case NodeKind::I2F: case NodeKind::F2I: {
            // Pure value-form preservation ops (no semantic change to the
            // Tier-0 surface — these are unboxing markers for the JIT only).
            i.op = static_cast<std::uint16_t>(Op::MOVE);
            i.a = static_cast<std::uint16_t>(n.ins[0]);
            ins(b->body, i);
            return;
        }
        case NodeKind::Neg: {
            // IBE-9 fix: Neg lowered to PY_UNOP with un_neg subop (was
            // lowered to MOVE — a copy, not a negation — so `-x` returned x).
            // UnOpKind constants: un_neg=1, un_invert=2, un_not=3
            // (see frontend/parser.cpp + lowering.cpp).
            i.op = static_cast<std::uint16_t>(Op::PY_UNOP);
            i.a = static_cast<std::uint16_t>(n.ins[0]);
            i.imm = 1;   // un_neg
            ins(b->body, i);
            return;
        }
        case NodeKind::Not: {
            // IBE-10 fix: Not lowered to PY_UNOP with un_not subop (was
            // lowered to MOVE — a copy, so `not x` returned x).
            i.op = static_cast<std::uint16_t>(Op::PY_UNOP);
            i.a = static_cast<std::uint16_t>(n.ins[0]);
            i.imm = 3;   // un_not
            ins(b->body, i);
            return;
        }
        default:
            emit_effect_op(id);
            return;
    }
}

// Materialize a phi MOVE source at the predecessor edge (cases 3/4 of
// emit_terminator). Constants and pass-produced PURE nodes (IV increments,
// folded ops) have no control dependence — if nothing else scheduled them,
// they emit here, right before the MOVE, because a phi no longer walks its
// inputs (see emit_value). Effect-chain sources are NEVER re-emitted:
// their computation is control-dependent (the continue-path double-add
// regression). Phis and parameters are registers the MOVE reads directly.
void Scheduler::materialize_phi_source(NodeId src) noexcept {
    if (src == invalid_node) return;
    const Node& sn = g_.node(src);
    const bool is_const = sn.kind == NodeKind::ConstInt ||
                          sn.kind == NodeKind::ConstFloat ||
                          sn.kind == NodeKind::ConstPy;
    if (is_const) {
        emit_value(src);   // idempotent per-block LOAD_CONST
        return;
    }
    if (sn.has(NodeFlag::Dead)) return;
    if (sn.has(NodeFlag::OnEffectChain) || is_control(sn.kind)) return;
    switch (sn.kind) {
        case NodeKind::Phi:
        case NodeKind::EffectPhi:
        case NodeKind::Parameter:
        case NodeKind::Guard:
        case NodeKind::DeoptBarrier:
            return;   // registers — the MOVE reads them where they live
        default:
            break;
    }
    // Unscheduled pure computation (its only user may be the phi itself):
    // emit at this predecessor — its operands dominate the edge by
    // construction of the dataflow that produced it.
    for (NodeId s : scheduled_) {
        if (s == src) return;
    }
    emit_value(src);
}

void Scheduler::emit_effect_op(NodeId id) noexcept {
    const Node& n = g_.node(id);
    BlockInfo* b = block(current_leader_);
    Instr i{};
    i.dst = static_cast<std::uint16_t>(id);
    switch (n.kind) {
        case NodeKind::LoadGlobal:
            i.op = static_cast<std::uint16_t>(Op::LOAD_GLOBAL);
            i.imm = n.symbol;
            ins(b->body, i);
            return;
        case NodeKind::StoreGlobal:
            i.op = static_cast<std::uint16_t>(Op::STORE_GLOBAL);
            i.a = static_cast<std::uint16_t>(n.ins[2]);
            i.imm = n.symbol;
            ins(b->body, i);
            return;
        case NodeKind::LoadAttr:
            i.op = static_cast<std::uint16_t>(Op::LOAD_ATTR);
            i.a = static_cast<std::uint16_t>(n.ins[2]);
            i.imm = n.symbol;
            ins(b->body, i);
            return;
        case NodeKind::StoreAttr:
            i.op = static_cast<std::uint16_t>(Op::STORE_ATTR);
            i.a = static_cast<std::uint16_t>(n.ins[2]);
            i.b = static_cast<std::uint16_t>(n.ins[3]);
            i.imm = n.symbol;
            ins(b->body, i);
            return;
        case NodeKind::LoadIndex:
            i.op = static_cast<std::uint16_t>(Op::LOAD_INDEX);
            i.a = static_cast<std::uint16_t>(n.ins[2]);
            i.b = static_cast<std::uint16_t>(n.ins[3]);
            ins(b->body, i);
            return;
        case NodeKind::StoreIndex:
            i.op = static_cast<std::uint16_t>(Op::STORE_INDEX);
            i.a = static_cast<std::uint16_t>(n.ins[2]);
            i.b = static_cast<std::uint16_t>(n.ins[3]);
            i.c = static_cast<std::uint16_t>(n.ins[4]);
            ins(b->body, i);
            return;
        case NodeKind::StoreField:
            i.op = static_cast<std::uint16_t>(Op::STORE_FIELD);
            i.a = static_cast<std::uint16_t>(n.ins[2]);   // base
            i.b = static_cast<std::uint16_t>(n.ins[3]);   // value
            i.imm = n.aux0;                               // slot ordinal
            ins(b->body, i);
            return;
        case NodeKind::LoadField:
            i.op = static_cast<std::uint16_t>(Op::LOAD_FIELD);
            i.a = static_cast<std::uint16_t>(n.ins[2]);
            i.imm = n.aux0;
            ins(b->body, i);
            return;
        case NodeKind::NewList:
        case NodeKind::NewTuple: {
            std::uint32_t count = n.ins.size() > 2 ? n.ins.size() - 2 : 0;
            std::uint32_t base = scratch_base_;
            for (std::uint32_t k = 0; k < count; ++k) {
                emit_value(n.ins[2 + k]);
                Instr mv{};
                mv.op = static_cast<std::uint16_t>(Op::MOVE);
                mv.dst = static_cast<std::uint16_t>(scratch_base_ + k);
                mv.a = static_cast<std::uint16_t>(n.ins[2 + k]);
                ins(b->body, mv);
            }
            i.op = static_cast<std::uint16_t>(n.kind == NodeKind::NewList ? Op::NEW_LIST
                                                                          : Op::NEW_TUPLE);
            i.a = static_cast<std::uint16_t>(base);
            i.b = static_cast<std::uint16_t>(count);
            ins(b->body, i);
            return;
        }
        case NodeKind::NewDict: {
            i.op = static_cast<std::uint16_t>(Op::NEW_DICT);
            ins(b->body, i);
            // Literal pairs arrive as data inputs [key, value, key, value...]
            // after (control, effect): emit a StoreIndex per pair.
            std::uint32_t pairs = n.ins.size() > 2 ? (n.ins.size() - 2) / 2 : 0;
            for (std::uint32_t k = 0; k < pairs; ++k) {
                NodeId key = n.ins[2 + k * 2];
                NodeId val = n.ins[2 + k * 2 + 1];
                emit_value(key);
                emit_value(val);
                Instr mvk{};
                mvk.op = static_cast<std::uint16_t>(Op::MOVE);
                mvk.dst = static_cast<std::uint16_t>(scratch_base_);
                mvk.a = static_cast<std::uint16_t>(key);
                ins(b->body, mvk);
                Instr mvv{};
                mvv.op = static_cast<std::uint16_t>(Op::MOVE);
                mvv.dst = static_cast<std::uint16_t>(scratch_base_ + 1);
                mvv.a = static_cast<std::uint16_t>(val);
                ins(b->body, mvv);
                Instr st{};
                st.op = static_cast<std::uint16_t>(Op::STORE_INDEX);
                st.a = static_cast<std::uint16_t>(id);   // the dict register
                st.b = static_cast<std::uint16_t>(scratch_base_);
                st.c = static_cast<std::uint16_t>(scratch_base_ + 1);
                ins(b->body, st);
            }
            return;
        }
        case NodeKind::ListAppend:
            i.op = static_cast<std::uint16_t>(Op::LIST_APPEND);
            i.a = static_cast<std::uint16_t>(n.ins[2]);
            i.b = static_cast<std::uint16_t>(n.ins[3]);
            ins(b->body, i);
            return;
        case NodeKind::CallPy:
        case NodeKind::CallDirect:
        case NodeKind::GuardedDirectCall: {
            std::uint32_t data_start = 2;
            NodeId callee = n.ins[data_start];
            std::uint32_t argc = n.aux0;
            bool has_kw_tail = (n.ins.size() > data_start + 1 + argc);
            std::uint32_t base = scratch_base_;
            for (std::uint32_t k = 0; k < argc; ++k) {
                emit_value(n.ins[data_start + 1 + k]);
                Instr mv{};
                mv.op = static_cast<std::uint16_t>(Op::MOVE);
                mv.dst = static_cast<std::uint16_t>(scratch_base_ + k);
                mv.a = static_cast<std::uint16_t>(n.ins[data_start + 1 + k]);
                ins(b->body, mv);
            }
            emit_value(callee);
            if (has_kw_tail) {
                NodeId kwnode = n.ins[data_start + 1 + argc];
                emit_value(kwnode);
                Instr ikw{};
                ikw.op = static_cast<std::uint16_t>(Op::CALL_KW);
                ikw.dst = static_cast<std::uint16_t>(id);
                ikw.a = static_cast<std::uint16_t>(callee);
                ikw.b = static_cast<std::uint16_t>(base);
                ikw.c = static_cast<std::uint16_t>(argc);
                ikw.imm = (static_cast<std::uint32_t>(kwnode) << 16) | (n.aux1 & 0xFFFF);
                ins(b->body, ikw);
            } else {
                Instr ic{};
                ic.op = static_cast<std::uint16_t>(Op::CALL);
                ic.dst = static_cast<std::uint16_t>(id);
                ic.a = static_cast<std::uint16_t>(callee);
                ic.b = static_cast<std::uint16_t>(base);
                ic.c = static_cast<std::uint16_t>(argc);
                ic.imm = n.aux1;
                ins(b->body, ic);
            }
            return;
        }
        case NodeKind::CallNative: {
            std::uint32_t count = n.ins.size() > 2 ? n.ins.size() - 2 : 0;
            std::uint32_t base = scratch_base_;
            for (std::uint32_t k = 0; k < count; ++k) {
                emit_value(n.ins[2 + k]);
                Instr mv{};
                mv.op = static_cast<std::uint16_t>(Op::MOVE);
                mv.dst = static_cast<std::uint16_t>(scratch_base_ + k);
                mv.a = static_cast<std::uint16_t>(n.ins[2 + k]);
                ins(b->body, mv);
            }
            Instr in2{};
            in2.op = static_cast<std::uint16_t>(Op::NATIVE);
            in2.dst = static_cast<std::uint16_t>(id);
            in2.a = static_cast<std::uint16_t>(base);
            in2.b = static_cast<std::uint16_t>(count);
            in2.imm = n.subop;
            ins(b->body, in2);
            return;
        }
        case NodeKind::Iter:
            i.op = static_cast<std::uint16_t>(Op::ITER);
            i.a = static_cast<std::uint16_t>(n.ins[2]);
            ins(b->body, i);
            return;
        case NodeKind::GetIterCheck:
            i.op = static_cast<std::uint16_t>(Op::ITER_CHECK);
            i.a = static_cast<std::uint16_t>(n.ins[2]);
            ins(b->body, i);
            return;
        case NodeKind::IterNext:
            i.op = static_cast<std::uint16_t>(Op::ITER_NEXT);
            i.a = static_cast<std::uint16_t>(n.ins[2]);
            ins(b->body, i);
            return;
        case NodeKind::Yield:
            i.op = static_cast<std::uint16_t>(Op::YIELD);
            i.a = static_cast<std::uint16_t>(n.ins[2]);
            ins(b->body, i);
            return;
        default:
            // Vector/memory/speculation kinds never appear in frontend IR.
            if (!n.ins.empty() && !is_control(g_.node(n.ins[0]).kind)) {
                i.op = static_cast<std::uint16_t>(Op::MOVE);
                i.a = static_cast<std::uint16_t>(n.ins[0]);
                ins(b->body, i);
            }
            return;
    }
}

// --- block bodies ------------------------------------------------------------
void Scheduler::emit_block(NodeId leader) noexcept {
    current_leader_ = leader;
    BlockInfo* b = block(leader);
    // Effect ops assigned to this leader, in creation order.
    g_.for_each_live([&](NodeId id) {
        const Node& n = g_.node(id);
        if (is_control(n.kind)) return;
        if (!n.has(NodeFlag::OnEffectChain)) return;
        if (n.ins.empty() || n.ins[0] != leader) return;
        emit_value(id);
    });
    (void)b;
}

// --- terminators --------------------------------------------------------------
void Scheduler::emit_terminator(NodeId leader) noexcept {
    current_leader_ = leader;
    BlockInfo* b = block(leader);

    // 1. If-user: conditional branch.
    NodeId if_user = invalid_node;
    g_.for_each_live([&](NodeId id) {
        const Node& n = g_.node(id);
        if (n.kind == NodeKind::If && !n.ins.empty() && n.ins[0] == leader) {
            if_user = id;
        }
    });
    if (if_user != invalid_node) {
        const Node& iff = g_.node(if_user);
        emit_value(iff.ins[1]);
        NodeId false_target = invalid_node;
        g_.for_each_live([&](NodeId id) {
            const Node& p = g_.node(id);
            if (p.kind == NodeKind::IfFalse && !p.ins.empty() && p.ins[0] == if_user) {
                false_target = id;
            }
        });
        Instr j{};
        j.op = static_cast<std::uint16_t>(Op::JUMP_IF_FALSE);
        j.a = static_cast<std::uint16_t>(iff.ins[1]);
        ins(b->term, j);
        if (false_target != invalid_node) {
            b->term_fixups.push_back({b->term.size() - 1, false_target});
        }
        return;   // true path falls through to next block
    }

    // 2. Return / Throw users terminate.
    bool terminated = false;
    g_.for_each_live([&](NodeId id) {
        if (terminated) return;
        const Node& n = g_.node(id);
        if (n.kind == NodeKind::Return && !n.ins.empty() && n.ins[0] == leader) {
            emit_value(n.ins[1]);
            Instr r{};
            r.op = static_cast<std::uint16_t>(Op::RETURN);
            r.a = static_cast<std::uint16_t>(n.ins[1]);
            ins(b->term, r);
            terminated = true;
            return;
        }
        if (n.kind == NodeKind::Throw && !n.ins.empty() && n.ins[0] == leader) {
            emit_value(n.ins[1]);
            Instr r{};
            r.op = static_cast<std::uint16_t>(Op::RAISE);
            r.a = static_cast<std::uint16_t>(n.ins[1]);
            ins(b->term, r);
            terminated = true;
            return;
        }
    });
    if (terminated) return;

    // 3. Loop backedge: leader is Loop.ins[1].
    bool is_backedge = false;
    NodeId loop_header = invalid_node;
    g_.for_each_live([&](NodeId id) {
        const Node& n = g_.node(id);
        if (n.kind == NodeKind::Loop && n.ins.size() >= 2 && n.ins[1] == leader) {
            is_backedge = true;
            loop_header = id;
        }
    });
    if (is_backedge) {
        // phi moves with pred index 1, then jump to header.
        const Node& loop = g_.node(loop_header);
        stdx::small_vector<NodeId, 8> phis;
        g_.for_each_live([&](NodeId id) {
            const Node& p = g_.node(id);
            if (p.kind == NodeKind::Phi && p.ins.size() >= 3 && p.ins.back() == loop_header) {
                phis.push_back(id);
            }
        });
        for (NodeId phi : phis) {
            const Node& p = g_.node(phi);
            NodeId src = p.ins[1];
            materialize_phi_source(src);
            Instr mv{};
            mv.op = static_cast<std::uint16_t>(Op::MOVE);
            mv.dst = static_cast<std::uint16_t>(phi);
            mv.a = static_cast<std::uint16_t>(src);
            ins(b->term, mv);
        }
        Instr j{};
        j.op = static_cast<std::uint16_t>(Op::JUMP);
        ins(b->term, j);
        b->term_fixups.push_back({b->term.size() - 1, loop_header});
        return;
    }

    // 4. Merge-region predecessors: phi moves + fallthrough or JUMP.
    NodeId merge_target = invalid_node;
    std::uint32_t pred_index = 0;
    g_.for_each_live([&](NodeId id) {
        if (merge_target != invalid_node) return;
        const Node& n = g_.node(id);
        if (n.kind != NodeKind::Region && n.kind != NodeKind::Loop) {
            return;   // Catch blocks are exception merges, not normal flow
        }
        for (std::uint32_t pi = 0; pi < n.ins.size(); ++pi) {
            if (n.ins[pi] == leader) {
                merge_target = id;
                pred_index = pi;
                return;
            }
        }
    });
    if (merge_target != invalid_node) {
        const Node& merge = g_.node(merge_target);
        if (merge.kind == NodeKind::Loop && pred_index == 1) {
            return;   // backedge handled above
        }
        stdx::small_vector<NodeId, 8> phis;
        g_.for_each_live([&](NodeId id) {
            const Node& p = g_.node(id);
            if (p.kind == NodeKind::Phi && p.ins.size() >= 2 && p.ins.back() == merge_target) {
                phis.push_back(id);
            }
        });
        for (NodeId phi : phis) {
            const Node& p = g_.node(phi);
            if (pred_index >= p.ins.size() - 1) continue;
            NodeId src = p.ins[pred_index];
            // Materialize unscheduled CONSTANT / pure sources in this block
            // before the move (phi move referencing a value scheduled later
            // = the ternary-None bug). Non-const effectful sources must not
            // re-emit: their computation may be control-dependent (the
            // continue-path double-add regression).
            materialize_phi_source(src);
            Instr mv{};
            mv.op = static_cast<std::uint16_t>(Op::MOVE);
            mv.dst = static_cast<std::uint16_t>(phi);
            mv.a = static_cast<std::uint16_t>(src);
            ins(b->term, mv);
        }
        Instr j{};
        j.op = static_cast<std::uint16_t>(Op::JUMP);
        ins(b->term, j);
        b->term_fixups.push_back({b->term.size() - 1, merge_target});
        return;
    }

    // 5. No terminator: fallthrough (pure straight-line leaders like Jump).
}

// --- top level -----------------------------------------------------------------
Result<void> Scheduler::run() noexcept {
    if (g_.start() == invalid_node) {
        return fail_msg("schedule: graph has no Start", diag_code::graph_verify_dangling_node);
    }

    // Register budget: node ids + scratch for arg packing / phi moves.
    scratch_base_ = g_.node_count() + 8;
    max_phi_width_ = 32;
    unit_.n_registers = g_.node_count() + 8 + 64;

    // Block order: reverse post-order from Start (each block after ALL its
    // forward predecessors — required for correct fallthrough + pure-value
    // scheduling), then handler blocks (Catch roots) appended.
    stdx::small_vector<NodeId, 64> postorder;
    dfs_post(g_.start(), postorder);
    // Handler (Catch) blocks append AFTER the main graph so normal flow
    // never falls into an exception handler; entries jump to them.
    stdx::small_vector<NodeId, 64> handler_postorder;
    g_.for_each_live([&](NodeId id) {
        if (g_.node(id).kind == NodeKind::Catch) {
            BlockInfo* b = block(id);
            if (!b->visited) dfs_post(id, handler_postorder);
        }
    });
    // Main flow: reverse post-order (Start first). Handlers: reverse of
    // their own postorder, appended at the END.
    stdx::small_vector<NodeId, 64> order;
    for (std::size_t i = postorder.size(); i-- > 0;) {
        order.push_back(postorder[i]);
    }
    for (std::size_t i = handler_postorder.size(); i-- > 0;) {
        order.push_back(handler_postorder[i]);
    }

    // Emit all blocks.
    for (NodeId leader : order) {
        emit_block(leader);
        emit_terminator(leader);
    }

    // Two-pass layout: (1) assign every block its start pc, (2) emit bodies,
    // (3) resolve jump fixups — targets may precede OR follow the jump site.
    std::uint32_t next_pc = 0;
    for (NodeId leader : order) {
        BlockInfo* b = block(leader);
        b->start_pc = next_pc;
        next_pc += static_cast<std::uint32_t>(b->body.size() + b->term.size());
    }
    for (NodeId leader : order) {
        BlockInfo* b = block(leader);
        for (const Instr& i : b->body) {
            unit_.code.push_back(rt::Instr{static_cast<std::uint16_t>(i.op), i.dst, i.a, i.b,
                                           i.c, i.imm});
        }
        for (const Instr& i : b->term) {
            unit_.code.push_back(rt::Instr{static_cast<std::uint16_t>(i.op), i.dst, i.a, i.b,
                                           i.c, i.imm});
        }
    }
    for (NodeId leader : order) {
        BlockInfo* b = block(leader);
        for (auto& [site, target] : b->term_fixups) {
            std::size_t at = b->start_pc + b->body.size() + site;
            BlockInfo* tb = block(target);
            if (tb && at < unit_.code.size()) {
                unit_.code[at].imm = tb->start_pc;
            }
        }
    }

    // Start block's pc must be 0.
    BlockInfo* sb = block(g_.start());
    if (sb && sb->start_pc != 0) {
        return fail_msg("schedule: start block is not at pc 0",
                        diag_code::graph_verify_dominance);
    }

    // Try ranges: [protected start, catch block start) -> catch block.
    // The protected start is the try-marker block's pc, but code HOISTED
    // above the marker (const loads, folded ops feeding the body) can also
    // raise — so the range extends DOWN to the previous try's handler (or
    // unit start for the outermost try). This keeps nested tries precise
    // while covering hoisted raisers.
    g_.for_each_live([&](NodeId id) {
        const Node& n = g_.node(id);
        if (n.kind != NodeKind::Catch) return;
        BlockInfo* cb = block(id);
        rt::TryRange range;
        range.handler_pc = cb ? cb->start_pc : 0;
        range.start_pc = 0;
        range.end_pc = range.handler_pc;
        // Precise inner bound: the previous handler's end (nested) or 0.
        for (const rt::TryRange& prev : unit_.try_ranges) {
            if (prev.handler_pc > range.start_pc && prev.handler_pc < range.handler_pc) {
                range.start_pc = prev.handler_pc;
            }
        }
        unit_.try_ranges.push_back(range);
    });

    if (unit_.code.empty()) {
        return fail_msg("schedule: produced empty code", diag_code::graph_verify_dominance);
    }
    return {};
}

}  // namespace

Result<void> schedule_unit(const ir::Graph& graph, rt::CodeUnit& unit,
                           const stdx::small_vector<char, 4096>& string_pool,
                           const stdx::small_vector<std::uint32_t, 8>& param_regs,
                           const stdx::small_vector<SymbolId, 8>& param_names,
                           bool is_generator) noexcept {
    unit.name = graph.function_name;
    unit.is_generator = is_generator;
    for (std::uint32_t r : param_regs) unit.param_regs.push_back(r);
    for (SymbolId s : param_names) unit.param_names.push_back(s);
    Scheduler sched(graph, unit, string_pool);
    return sched.run();
}

}  // namespace abi_v1
}  // namespace vortex::pipeline
