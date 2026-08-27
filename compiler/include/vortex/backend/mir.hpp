// =============================================================================
// vortex/backend/mir.hpp — Machine IR: Sea of Nodes lowered to machine form
//
// Purpose:
//   Pass 52's output. Every MIR node is a machine operation with a register
//   class, at most 3 operands (std::inplace_vector semantics via stdx —
//   Rule 19: zero heap allocation for 99% of instructions), and a home slot
//   in the Tier-0 register file (its originating IR NodeId) — which is what
//   makes deoptimization reconstruction free.
//
// Design:
//   - Identifiers are dense u32 indices into a bump-allocated arena; no
//     pointers in edges (Rule 15 discipline carried to the backend).
//   - Virtual registers ARE MIR node ids: single-def SSA at machine level.
//   - The cold flag drives Pass 55's hot/cold partitioning.
//   - Safepoints carry the (machinePC, frameState, live vreg->physreg map)
//     record the deoptimizer consumes.
// =============================================================================

#pragma once

#include <cstdint>

#include "vortex/backend/target.hpp"
#include "vortex/ir/node.hpp"
#include "vortex/stdx/small_vector.hpp"

namespace vortex::backend {

inline namespace abi_v1 {

/// Machine opcodes — arch-neutral MIR surface. Distinct sequential values
/// (a switch must be exhaustive and unambiguous); the x86/ARM encoding and
/// per-cost-class latencies are derived at emission from tables owned by
/// the emitters and the TargetDescriptor — NEVER overloaded onto the opcode
/// value.
enum class MOp : std::uint8_t {
    MOVri = 1,      // dst <- imm64
    MOVrm,          // dst <- [frame slot]
    MOVmr,          // [frame slot] <- src
    MOVrr,          // dst <- src (reg-reg)
    ADDrr,
    SUBrr,
    IMULrr,
    NEGrr,
    CMPrr,
    CMPri,
    TESTrr,
    JMP,
    Jcc,
    CALLri,
    CALLrr,
    RET,
    INCREF,
    DECREF,
    GUARD_TAG,
    GUARD_INT,
    GUARD_FLOAT,    // tag == Tag::Float (3) on both operands; else deopt
    PUSH,
    POP,
    LEA,
    VADDrr,
    VMOVUPS,
    VPINSRQ,
    // --- scalar float (SSE2 / A64 scalar) -------------------------------
    // 64-bit IEEE float in the low lane of an XMM/D-register. Distinct from
    // VADDrr (full-lane SIMD): the cost model treats them differently
    // (VecAlu vs Move+Alu) and the emitter selects ADDSD vs VPADDQ.
    FMOVrr,         // movsd xmm_dst, xmm_src
    FMOVrm,         // movsd xmm_dst, [base + slot*16 + off]
    FMOVmr,         // movsd [base + slot*16 + off], xmm_src
    FADDrr,         // addsd xmm_dst, xmm_src
    FSUBrr,         // subsd xmm_dst, xmm_src
    FMULrr,         // mulsd xmm_dst, xmm_src
    FDIVrr,         // divsd xmm_dst, xmm_src
    FNEGrr,         // xorpd with sign-mask (sign-flip; no FP subtract needed)
    // FP constant materialization — writes the IEEE 754 bit pattern (an
    // int64 payload) AND tag=Tag::Float to the home slot. Distinct from
    // MOVri which writes payload + tag=Tag::Int. Without this op the
    // ConstFloat lowering would have to compose a MOVri (writes
    // payload+kTagInt) followed by a tag-only write — but no MOp exists
    // for "write imm directly to the tag offset of a home slot" without
    // risking payload clobber on a spill, so we fold the two writes into
    // one node.
    FCONSTri,
    // Bool set-on-condition — consumes the flags from a preceding CMPrr.
    // Operand[0] = MCond immediate (the condition to test). The codegen
    // emits XOR+SETcc+MOVmr (write 0/1 to home.payload) and writes tag=
    // Tag::Bool. Distinct from CMPrr (which only sets flags without
    // producing a Python-typed result). The fast path covers PyCompare
    // with both operands provably_int and subop in {LT,LE,GT,GE,EQ,NE};
    // Is/IsNot/In/NotIn still bridge (object identity / __contains__).
    SETCCri,
    SAFEPOINT,
    DEOPT_TRAP,
};

/// Branch predicates — arch-neutral. The machine condition encoding (x86
/// Jcc low nibble, AArch64 cond field) is emitter-side table data; the old
/// version of this enum carried raw x86 opcode bytes as its values, which
/// silently leaked one ISA into the neutral MIR.
enum class MCond : std::uint8_t {
    EQ = 0, NE, LT, GE, LE, GT,
    Count,   // sentinel: keeps the value space dense and future additions
             // append-only (the dispatch-table lesson from Op).
};

/// Arch-neutral cost classification (Rule 45): maps a machine op onto the
/// TargetDescriptor's latency row. Scheduling and peephole cost decisions
/// go through this — never through opcode-shaped literals.
[[nodiscard]] constexpr CostClass cost_class(MOp op) noexcept {
    switch (op) {
        case MOp::MOVri:
        case MOp::MOVrr:
            return CostClass::Move;
        case MOp::MOVrm:
        case MOp::VMOVUPS:
            return CostClass::Load;
        case MOp::MOVmr:
            return CostClass::Store;
        case MOp::ADDrr:
        case MOp::SUBrr:
        case MOp::NEGrr:
        case MOp::CMPrr:
        case MOp::CMPri:
        case MOp::TESTrr:
        case MOp::LEA:
            return CostClass::Alu;
        case MOp::IMULrr:
            return CostClass::Mul;
        case MOp::VADDrr:
        case MOp::VPINSRQ:
            return CostClass::VecAlu;
        case MOp::FMOVrr:
        case MOp::FMOVrm:
        case MOp::FMOVmr:
            return CostClass::Move;   // FP moves are throughput-1 like GP moves
        case MOp::FADDrr:
        case MOp::FSUBrr:
        case MOp::FMULrr:
        case MOp::FDIVrr:
        case MOp::FNEGrr:
            return CostClass::Alu;     // FP adds throughput-1; mul ~4c, div ~13c, but Alu is the closest class
        case MOp::FCONSTri:
            return CostClass::Move;   // 1x mov + 1x store-imm32, throughput-bound
        case MOp::SETCCri:
            return CostClass::Alu;   // xor+setcc+store, ~3 ALU ops
        case MOp::JMP:
        case MOp::Jcc:
            return CostClass::Branch;
        case MOp::CALLri:
        case MOp::CALLrr:
        case MOp::RET:
            return CostClass::Call;
        case MOp::PUSH:
        case MOp::POP:
            return CostClass::Move;
        case MOp::INCREF:
        case MOp::DECREF:
        case MOp::GUARD_TAG:
        case MOp::GUARD_INT:
        case MOp::GUARD_FLOAT:
            return CostClass::Alu;   // tag load + compare, branch folded
        case MOp::SAFEPOINT:
        case MOp::DEOPT_TRAP:
            return CostClass::Branch;
    }
    return CostClass::Alu;
}

/// Arch-neutral cost classification for IR-level binary ops (Rule 45).
/// Mid-level passes (SLP, vectorization) that need to weigh a packet's
/// scalar-vs-vector cost go through this — never through opcode-shaped
/// literals in pass logic. The BinOpKind→CostClass mapping is semantic
/// (Mul is always CostClass::Mul on every architecture); the actual
/// cycle cost is queried separately via TargetDescriptor::latency().
[[nodiscard]] constexpr CostClass cost_class(ir::BinOpKind op) noexcept {
    switch (op) {
        case ir::BinOpKind::Mul:
        case ir::BinOpKind::Pow:
        case ir::BinOpKind::MatMul:
            return CostClass::Mul;
        case ir::BinOpKind::TrueDiv:
        case ir::BinOpKind::FloorDiv:
        case ir::BinOpKind::Mod:
            return CostClass::Div;
        case ir::BinOpKind::Add:
        case ir::BinOpKind::Sub:
        case ir::BinOpKind::LShift:
        case ir::BinOpKind::RShift:
        case ir::BinOpKind::BitAnd:
        case ir::BinOpKind::BitOr:
        case ir::BinOpKind::BitXor:
            return CostClass::Alu;
    }
    return CostClass::Alu;
}

/// Operand kinds: virtual register, immediate, or frame-relative memory
/// (the Tier-0 home slot). Memory operands only reference the frame base
/// — real addressing mode selection happens at emission.
struct MachineOperand {
    enum Kind : std::uint8_t { None = 0, VReg, Imm, FrameSlot };
    Kind kind{None};
    std::uint32_t vreg{0};       // VReg: MIR node id
    std::int64_t imm{0};         // Imm payload / condition code
    std::uint32_t slot{0};       // FrameSlot: Tier-0 register index
    std::uint8_t tag_off{0};     // byte offset within the 16-byte Value

    [[nodiscard]] static MachineOperand reg(std::uint32_t v) noexcept {
        return MachineOperand{VReg, v, 0, 0, 0};
    }
    [[nodiscard]] static MachineOperand imm_op(std::int64_t i) noexcept {
        return MachineOperand{Imm, 0, i, 0, 0};
    }
    [[nodiscard]] static MachineOperand slot_op(std::uint32_t s,
                                                std::uint8_t off = 8) noexcept {
        return MachineOperand{FrameSlot, 0, 0, s, off};
    }
    [[nodiscard]] static MachineOperand cond_op(MCond c) noexcept {
        return MachineOperand{Imm, 0, static_cast<std::int64_t>(c), 0, 0};
    }
};

struct MachineNode {
    MOp op{MOp::MOVrr};
    MachineRegClass rc{MachineRegClass::GPR};
    std::uint32_t id{0};
    std::uint32_t home_slot{0xFFFFFFFF};   // Tier-0 reg index (deopt home)
    stdx::small_vector<MachineOperand, 3> operands{};
    bool is_cold{false};       // Pass 55: emit into the cold partition
    bool is_safepoint{false};  // emission records a SafepointRecord here
    std::uint32_t frame_state_id{0xFFFFFFFF};
    /// Block id for interval computation; 0 = entry block.
    std::uint32_t block{0};
    /// Position within the block (for interval ordering).
    std::uint32_t pos{0};
};

/// A basic block in the MIR: successor indices for interval + layout.
struct MachineBlock {
    std::uint32_t id{0};
    stdx::small_vector<std::uint32_t, 2> succs{};
    std::uint32_t loop_depth{0};
    bool is_cold{false};
};

/// The Machine IR graph: arena of nodes + blocks. Index == id.
struct MachineGraph {
    stdx::small_vector<MachineNode, 128> nodes{};   // node 0 reserved
    stdx::small_vector<MachineBlock, 16> blocks{};

    MachineGraph() noexcept {
        // Reserve node 0 as a sentinel so the first real create() returns
        // id=1 — the codegen iterates `for (id=1; id <= node_count(); ++id)`
        // and would silently skip a real node 0. Without this reservation
        // the lowering's first MIR node (typically the entry MOVrm/MOVri)
        // is never emitted, producing a broken prologue that reads
        // uninitialized home slots.
        nodes.push_back(MachineNode{});
        nodes.back().op = MOp::RET;   // never executed; safe sentinel
    }

    [[nodiscard]] std::uint32_t create_block() noexcept {
        blocks.push_back(MachineBlock{});
        blocks.back().id = static_cast<std::uint32_t>(blocks.size()) - 1;
        return blocks.back().id;
    }

    [[nodiscard]] std::uint32_t create(MOp op, MachineRegClass rc,
                                       std::uint32_t home_slot) noexcept {
        nodes.push_back(MachineNode{});
        MachineNode& n = nodes.back();
        n.op = op;
        n.rc = rc;
        n.id = static_cast<std::uint32_t>(nodes.size()) - 1;
        n.home_slot = home_slot;
        return n.id;
    }

    void add_operand(std::uint32_t node, MachineOperand op) noexcept {
        nodes[node].operands.push_back(op);
    }

    [[nodiscard]] std::size_t node_count() const noexcept { return nodes.size() - 1; }
    [[nodiscard]] MachineNode& node(std::uint32_t id) noexcept { return nodes[id]; }
    [[nodiscard]] const MachineNode& node(std::uint32_t id) const noexcept {
        return nodes[id];
    }
};

/// Live interval for linear-scan allocation (Pass 53).
struct LiveInterval {
    std::uint32_t vreg{0};
    std::uint32_t start{0};
    std::uint32_t end{0};
    std::uint32_t loop_depth{0};
    std::int32_t phys_reg{-1};      // assigned; -1 = spilled
    std::uint32_t spill_slot{0};    // frame home when spilled
    bool is_pyobject{false};        // spills insert refcount traffic
    bool assigned{false};
};

/// Safepoint record — the .vortex_unwind entry format (Phase V).
#pragma pack(push, 1)
struct SafepointRecord {
    std::uint32_t pc_offset{0};        // from function start
    std::uint32_t frame_state_id{0};   // IR FrameState pool index
    std::uint16_t live_regs{0};        // count of (physreg, slot) pairs
    std::uint16_t stack_depth{0};      // Tier-0 reg count at this point
    // Followed inline: live_regs x { u8 physreg, u8 pad, u16 slot }
};
#pragma pack(pop)

struct SafepointMapping {
    std::uint8_t physreg{0};
    std::uint16_t slot{0};
};

}  // namespace abi_v1
}  // namespace vortex::backend
