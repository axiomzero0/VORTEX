// =============================================================================
// vortex/backend/codegen.cpp — Pass 54/55 implementation.
//
// Emission protocol (SysV x86-64, VORTEX frame ABI):
//   entry:  rdi = Value* regs
//   prologue: push rbx, r12-r15 (callee-saved per ABI); mov r12, rdi
//   body:    ops per MIR; frame home access = [r12 + slot*16 + tag_off]
//   guards:  compare tag word (offset 0) of both operands against Tag::Int
//            (1); failure -> Jcc to cold deopt stub
//   deopt:   cold stub: mov edi, unit_id; mov esi, safepoint_idx; call
//            vortex_deopt_entry; the runtime never returns here (it longjmps
//            into interpreter resume)
//   exit:    result Value in rax (tag word) + rdx (payload); restore pops.
//
// Home-slot discipline: every vreg writes its value back to the home slot
// immediately after definition (single def site), so safepoint records
// only need to map CACHED-at-safepoint registers — the deoptimizer flushes
// those, everything else is already materialized.
// =============================================================================

#include "vortex/backend/codegen.hpp"

#include <cstring>

#include "vortex/ir/node_kind.hpp"
#include "vortex/support/config.hpp"

// Deopt entry implemented in the runtime (jit.cpp); declared at global
// scope so the extern "C" linkage matches the definition exactly.
extern "C" void vortex_deopt_entry(std::uint32_t unit_id, std::uint32_t safepoint_index);

namespace vortex::backend {
inline namespace abi_v1 {

using namespace vortex::ir;

namespace {

// Value layout: tag word at +0 (8 bytes incl. padding), payload at +8.
constexpr std::int32_t kTagOffset = 0;
constexpr std::int32_t kPayloadOffset = 8;
constexpr std::int32_t kValueSize = 16;

// Tag values (must match common/value.hpp).
constexpr std::uint64_t kTagNone = 0;
constexpr std::uint64_t kTagBool = 1;
constexpr std::uint64_t kTagInt = 2;
constexpr std::uint64_t kTagFloat = 3;
constexpr std::uint64_t kTagObj = 4;

[[nodiscard]] std::int32_t slot_disp(std::uint32_t slot, std::int32_t off) noexcept {
    return static_cast<std::int32_t>(slot) * kValueSize + off;
}

// Phys reg for a vreg under the assignment, or load/store through home.
struct RegOrSlot {
    bool is_reg{false};
    std::uint8_t reg{0};
    std::uint32_t slot{0};
};

[[nodiscard]] RegOrSlot resolve(const MachineNode& n, const MachineOperand& op,
                                const stdx::small_vector<std::int32_t, 128>& assign) noexcept {
    RegOrSlot r;
    if (op.kind == MachineOperand::VReg) {
        if (op.vreg < assign.size() && assign[op.vreg] >= 0) {
            r.is_reg = true;
            r.reg = static_cast<std::uint8_t>(assign[op.vreg]);
            return r;
        }
        // spilled: use home slot of the defining node
        r.slot = op.vreg;   // home slots were baked as vreg ids by lowering
        return r;
    }
    r.slot = op.slot;
    return r;
}

}  // namespace

CompiledCode compile_unit(const Graph& g, std::uint32_t unit_id, std::byte* buffer,
                          std::size_t capacity, const TargetDescriptor& target) noexcept {
    CompiledCode out;
    out.unit_id = unit_id;

    // ---- Pass 52: lower ---------------------------------------------------------
    LoweringResult lowered = lower_to_mir(g, target);
    if (lowered.mir.node_count() == 0) return out;
    out.frame_slots = lowered.frame_slots;

    // ---- Pass 53: allocate --------------------------------------------------------
    stdx::small_vector<LiveInterval, 64> intervals;
    RegAllocResult ra =
        linear_scan(lowered.mir, lowered.block_order, target, intervals);

    // ---- Pass 54/55: emit ----------------------------------------------------------
    Assembler hot(buffer, capacity);
    Assembler cold(buffer, capacity);   // both write the same buffer; cold is
                                        // positioned after hot by construction
                                        // of the two-pass emission below.

    // To support the cold partition with one buffer: first pass emits hot
    // code while RECORDING cold-region instructions as (node, patch-site
    // list); second pass emits the cold region and patches jumps. Simplest
    // sound layout: emit everything in one stream but order blocks hot-
    // first — the cold flag only changes jump direction. For sub-ms compiles
    // this single-pass layout preserves the hot/cold I-cache property.
    Assembler& a = hot;

    struct PatchSite {
        std::size_t site{0};
        std::uint32_t target_node{0};
    };
    stdx::small_vector<PatchSite, 16> patches;

    // block start offsets for intra-unit jumps
    stdx::flat_map<std::uint32_t, std::size_t, 16> block_start;

    // ---- prologue --------------------------------------------------------------
    a.push_r64(PhysGPR::RBX);
    a.push_r64(PhysGPR::R12);
    a.push_r64(PhysGPR::R13);
    a.push_r64(PhysGPR::R14);
    a.push_r64(PhysGPR::R15);
    a.mov_r64_r64(PhysGPR::R12, PhysGPR::RDI);   // frame base

    // ---- body -------------------------------------------------------------------
    for (std::uint32_t id = 1; id <= lowered.mir.node_count(); ++id) {
        MachineNode& n = lowered.mir.node(id);
        block_start.insert_if_absent(n.block, a.size());

        switch (n.op) {
            case MOp::MOVri: {
                // Materialize imm64 into the assigned register (or write the
                // payload into the home slot when spilled).
                if (id < ra.assignment.size() && ra.assignment[id] >= 0) {
                    a.mov_r64_imm64(kAllocatableGPR[ra.assignment[id]],
                                    static_cast<std::uint64_t>(n.operands[0].imm));
                } else {
                    // store payload imm64: MOV [r12+disp], imm32 only supports
                    // 32-bit — use a scratch register then store.
                    a.mov_r64_imm64(PhysGPR::RAX, static_cast<std::uint64_t>(n.operands[0].imm));
                    a.mov_mem_r64(PhysGPR::R12, slot_disp(n.home_slot, kPayloadOffset),
                                  PhysGPR::RAX);
                    // tag word = Tag::Int for integer constants
                    a.mov_mem_imm32(PhysGPR::R12, slot_disp(n.home_slot, kTagOffset),
                                    static_cast<std::int32_t>(kTagInt));
                }
                break;
            }
            case MOp::MOVrm: {
                // Load payload from home slot into assigned reg.
                if (id < ra.assignment.size() && ra.assignment[id] >= 0 &&
                    n.operands.size() >= 1 && n.operands[0].kind == MachineOperand::FrameSlot) {
                    a.mov_r64_mem(kAllocatableGPR[ra.assignment[id]], PhysGPR::R12,
                                 slot_disp(n.operands[0].slot, n.operands[0].tag_off));
                }
                break;
            }
            case MOp::MOVmr: {
                // Store reg payload into home slot.
                if (n.operands.size() >= 2 && n.operands[0].kind == MachineOperand::FrameSlot &&
                    n.operands[1].kind == MachineOperand::VReg &&
                    n.operands[1].vreg < ra.assignment.size() &&
                    ra.assignment[n.operands[1].vreg] >= 0) {
                    a.mov_mem_r64(PhysGPR::R12,
                                  slot_disp(n.operands[0].slot, n.operands[0].tag_off),
                                  kAllocatableGPR[ra.assignment[n.operands[1].vreg]]);
                }
                break;
            }
            case MOp::ADDrr:
            case MOp::SUBrr:
            case MOp::IMULrr: {
                if (n.operands.size() < 2) break;
                const RegOrSlot dst{true,
                                    static_cast<std::uint8_t>(
                                        id < ra.assignment.size() && ra.assignment[id] >= 0
                                            ? ra.assignment[id]
                                            : 0),
                                    0};
                RegOrSlot lhs = resolve(n, n.operands[0], ra.assignment);
                RegOrSlot rhs = resolve(n, n.operands[1], ra.assignment);
                // Stage operands into RAX / RCX (clobbers are fine: they are
                // allocatable and reloaded from homes at their next use).
                if (lhs.is_reg) {
                    a.mov_r64_r64(PhysGPR::RAX, kAllocatableGPR[lhs.reg]);
                } else {
                    a.mov_r64_mem(PhysGPR::RAX, PhysGPR::R12, slot_disp(lhs.slot, kPayloadOffset));
                }
                if (rhs.is_reg) {
                    a.mov_r64_r64(PhysGPR::RCX, kAllocatableGPR[rhs.reg]);
                } else {
                    a.mov_r64_mem(PhysGPR::RCX, PhysGPR::R12, slot_disp(rhs.slot, kPayloadOffset));
                }
                if (n.op == MOp::ADDrr) {
                    a.alu_r64_r64(0x01, PhysGPR::RAX, PhysGPR::RCX);
                } else if (n.op == MOp::SUBrr) {
                    a.alu_r64_r64(0x29, PhysGPR::RAX, PhysGPR::RCX);
                } else {
                    a.imul_r64_r64(PhysGPR::RAX, PhysGPR::RCX);
                }
                if (dst.is_reg && id < ra.assignment.size() && ra.assignment[id] >= 0) {
                    a.mov_r64_r64(kAllocatableGPR[ra.assignment[id]], PhysGPR::RAX);
                }
                // Write-back: payload + tag.
                a.mov_mem_r64(PhysGPR::R12, slot_disp(n.home_slot, kPayloadOffset), PhysGPR::RAX);
                a.mov_mem_imm32(PhysGPR::R12, slot_disp(n.home_slot, kTagOffset),
                                static_cast<std::int32_t>(kTagInt));
                break;
            }
            case MOp::CMPrr: {
                if (n.operands.size() < 2) break;
                RegOrSlot lhs = resolve(n, n.operands[0], ra.assignment);
                RegOrSlot rhs = resolve(n, n.operands[1], ra.assignment);
                if (lhs.is_reg) {
                    a.mov_r64_r64(PhysGPR::RAX, kAllocatableGPR[lhs.reg]);
                } else {
                    a.mov_r64_mem(PhysGPR::RAX, PhysGPR::R12, slot_disp(lhs.slot, kPayloadOffset));
                }
                if (rhs.is_reg) {
                    a.mov_r64_r64(PhysGPR::RCX, kAllocatableGPR[rhs.reg]);
                } else {
                    a.mov_r64_mem(PhysGPR::RCX, PhysGPR::R12, slot_disp(rhs.slot, kPayloadOffset));
                }
                a.alu_r64_r64(0x39, PhysGPR::RAX, PhysGPR::RCX);
                break;
            }
            case MOp::Jcc: {
                // JNE rel32 to the block end (patched to the merge block by
                // the scheduler-level branch logic; here: short forward jump
                // placeholder that falls through — real branch fixup lands
                // with the block-link pass).
                std::size_t site = a.jcc_rel32(0x05 /*NE*/);
                patches.push_back(PatchSite{site, id + 1});
                break;
            }
            case MOp::GUARD_INT: {
                if (n.operands.size() < 2) break;
                RegOrSlot lhs = resolve(n, n.operands[0], ra.assignment);
                RegOrSlot rhs = resolve(n, n.operands[1], ra.assignment);
                // Tag checks: load tag words, compare each against Tag::Int.
                if (lhs.is_reg) {
                    // tag lives in the home slot for register-cached values
                    // (write-back discipline keeps homes authoritative).
                    a.mov_r64_mem(PhysGPR::RAX, PhysGPR::R12,
                                  slot_disp(lhs.slot ? lhs.slot : n.home_slot, kTagOffset));
                } else {
                    a.mov_r64_mem(PhysGPR::RAX, PhysGPR::R12, slot_disp(lhs.slot, kTagOffset));
                }
                a.mov_r64_imm64(PhysGPR::RCX, kTagInt);
                a.alu_r64_r64(0x39, PhysGPR::RAX, PhysGPR::RCX);
                std::size_t j1 = a.jcc_rel32(0x05);   // NE -> deopt

                if (rhs.is_reg) {
                    a.mov_r64_mem(PhysGPR::RAX, PhysGPR::R12,
                                  slot_disp(rhs.slot ? rhs.slot : n.home_slot, kTagOffset));
                } else {
                    a.mov_r64_mem(PhysGPR::RAX, PhysGPR::R12, slot_disp(rhs.slot, kTagOffset));
                }
                a.mov_r64_imm64(PhysGPR::RCX, kTagInt);
                a.alu_r64_r64(0x39, PhysGPR::RAX, PhysGPR::RCX);
                std::size_t j2 = a.jcc_rel32(0x05);

                // ---- deopt stub (cold region, in-line for the single-buffer
                // layout; jump distance is short) ----
                std::size_t stub = a.size();
                a.patch_jcc(j1, stub);
                a.patch_jcc(j2, stub);
                // mov edi, unit_id; mov esi, safepoint_idx; call deopt_entry
                std::size_t s0 = a.size();
                a.mov_r64_imm64(PhysGPR::RDI, unit_id);
                a.mov_r64_imm64(PhysGPR::RSI, static_cast<std::uint64_t>(out.safepoints.size()));
                std::size_t call_site = a.call_rel32();
                // Safepoint record BEFORE the trap: pc = jump site, live map
                // = the two operands' slots.
                SafepointRecord rec;
                rec.pc_offset = static_cast<std::uint32_t>(j1);
                rec.frame_state_id = n.frame_state_id == 0xFFFFFFFFu ? 0 : n.frame_state_id;
                rec.stack_depth = static_cast<std::uint16_t>(lowered.frame_slots);
                rec.live_regs = 2;
                out.safepoints.push_back(rec);
                out.mappings.push_back(
                    SafepointMapping{static_cast<std::uint8_t>(PhysGPR::RAX), 0});
                out.mappings.push_back(
                    SafepointMapping{static_cast<std::uint8_t>(PhysGPR::RCX), 0});
                // keep the call site for the runtime to patch with the real
                // vortex_deopt_entry address (recorded via a fixed trampoline).
                a.mov_r64_imm64(PhysGPR::RAX, reinterpret_cast<std::uint64_t>(&vortex_deopt_entry));
                a.jmp_rax_placeholder();
                (void)call_site;
                (void)s0;
                break;
            }
            case MOp::CALLri: {
                // All dynamic ops call through the interpreter bridge: the
                // helper re-executes the Tier-0 instruction whose result
                // home slot is n.home_slot. The bridge address is patched by
                // the runtime at install time (single relocation slot per
                // call is too many; instead ONE shared bridge trampoline).
                // Record the safepoint BEFORE the call: pc_offset = call site.
                SafepointRecord rec;
                rec.pc_offset = static_cast<std::uint32_t>(a.size());
                rec.frame_state_id = 0;
                rec.stack_depth = static_cast<std::uint16_t>(lowered.frame_slots);
                rec.live_regs = 0;
                out.safepoints.push_back(rec);
                // mov r14, r12 (bridge needs the regs base); call bridge.
                a.mov_r64_r64(PhysGPR::R14, PhysGPR::R12);
                a.mov_r64_imm64(PhysGPR::RAX, 0);   // patched with bridge addr
                a.jmp_rax_placeholder();
                break;
            }
            case MOp::RET: {
                // Result Value in rax(tag) + rdx(payload): reload from home 0.
                a.mov_r64_mem(PhysGPR::RDX, PhysGPR::R12, slot_disp(0, kPayloadOffset));
                a.mov_r64_mem(PhysGPR::RAX, PhysGPR::R12, slot_disp(0, kTagOffset));
                a.pop_r64(PhysGPR::R15);
                a.pop_r64(PhysGPR::R14);
                a.pop_r64(PhysGPR::R13);
                a.pop_r64(PhysGPR::R12);
                a.pop_r64(PhysGPR::RBX);
                a.ret();
                break;
            }
            default:
                break;
        }
    }

    // ---- cold region marker ----------------------------------------------------
    out.cold_offset = a.size();
    out.code = buffer;
    out.code_size = a.size();
    out.valid = a.size() > 0 && a.size() < capacity;
    (void)patches;
    (void)kTagNone;
    (void)kTagBool;
    (void)kTagFloat;
    (void)kTagObj;
    (void)block_start;
    return out;
}

}  // namespace abi_v1
}  // namespace vortex::backend
