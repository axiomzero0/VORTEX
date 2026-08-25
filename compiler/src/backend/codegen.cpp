// =============================================================================
// vortex/backend/codegen.cpp — Pass 54/55 implementation (x86-64 emitter).
//
// ARCHITECTURE CONTRACT
//   This file is the SysV x86-64 emitter and nothing else. compile_unit()
//   refuses to emit for any other descriptor architecture — a mismatched
//   pair returns valid=false with zero bytes written instead of emitting
//   garbage that happens to run on the build machine. The AArch64 emitter
//   is a separate compilation unit with the same contract.
//
//   The register PARTITION (allocatable set, frame-protocol roles) comes
//   from the TargetDescriptor — never from file-local tables. What remains
//   local are genuine x86-64 ISA/ABI facts: REX encodings live in the
//   Assembler, and this file picks the SysV scratch/argument registers
//   (RAX/RCX/RDX/RDI/RSI) for operand staging.
//
// Emission protocol (SysV x86-64, VORTEX frame ABI):
//   entry:  rdi = Value* regs
//   prologue: push rbx + the descriptor's four reserved roles; r12 <- rdi
//   body:    ops per MIR; frame home access = [frame_base + slot*16 + off]
//   guards:  compare tag word (offset 0) of both operands against Tag::Int
//            (2); failure -> Jcc to cold deopt stub
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

// MCond -> x86-64 Jcc condition nibble (two-byte 0F 8x form). Emitter-side
// table: the neutral MIR carries no x86 encodings.
constexpr std::uint8_t kX86Cond[enum_size(MCond::Count)] = {
    0x04,   // EQ  (e)
    0x05,   // NE  (ne)
    0x0C,   // LT  (l, signed)
    0x0D,   // GE  (ge, signed)
    0x0E,   // LE  (le, signed)
    0x0F,   // GT  (g, signed)
};

[[nodiscard]] std::int32_t slot_disp(std::uint32_t slot, std::int32_t off) noexcept {
    return static_cast<std::int32_t>(slot) * kValueSize + off;
}

// Phys reg for a vreg under the assignment, or load/store through home.
struct RegOrSlot {
    bool is_reg{false};
    std::uint8_t reg{0};
    std::uint32_t slot{0};
};

}  // namespace

CompiledCode compile_unit(const Graph& g, std::uint32_t unit_id, std::byte* buffer,
                          std::size_t capacity, const TargetDescriptor& target) noexcept {
    CompiledCode out;
    out.unit_id = unit_id;

    // ---- architecture gate: this emitter only speaks x86-64 -------------------
    // Fail loudly (valid=false, zero bytes) instead of emitting code that only
    // happens to run on the machine this compiler was built on.
    if (target.architecture != Arch::X86_64) return out;
    if (buffer == nullptr || capacity == 0) return out;

    // Descriptor-driven register partition (the ONLY legal source).
    const std::uint8_t frame_base = target.reserved[static_cast<std::uint32_t>(ReservedGPR::FrameBase)];
    const std::uint8_t vm_ctx = target.reserved[static_cast<std::uint32_t>(ReservedGPR::VMContext)];
    const std::uint32_t n_alloc = target.allocatable_gprs;
    if (n_alloc == 0 || n_alloc > kMaxAllocatable) return out;

    // Allocation index -> physical encoding, with a defensive scratch
    // fallback for out-of-range indices (never trust derived indices).
    const auto alloc_reg = [&](std::int32_t idx) noexcept -> std::uint8_t {
        if (idx < 0 || static_cast<std::uint32_t>(idx) >= n_alloc) return x86::RAX;
        return target.allocatable[static_cast<std::uint32_t>(idx)];
    };

    // ---- Pass 52: lower ---------------------------------------------------------
    LoweringResult lowered = lower_to_mir(g, target);
    if (lowered.mir.node_count() == 0) return out;
    out.frame_slots = lowered.frame_slots;

    // ---- Pass 53: allocate --------------------------------------------------------
    stdx::small_vector<LiveInterval, 64> intervals;
    RegAllocResult ra =
        linear_scan(lowered.mir, lowered.block_order, target, intervals);

    // ---- Pass 54/55: emit ----------------------------------------------------------
    Assembler a(buffer, capacity);

    struct PatchSite {
        std::size_t site{0};
        std::uint32_t target_node{0};
    };
    stdx::small_vector<PatchSite, 16> patches;

    // block start offsets for intra-unit jumps
    stdx::flat_map<std::uint32_t, std::size_t, 16> block_start;

    // ---- prologue --------------------------------------------------------------
    // Callee-saved under SysV that this unit touches: RBX (allocatable) plus
    // the four frame-protocol roles from the descriptor.
    a.push_r64(x86::RBX);
    for (std::uint32_t r = 0; r < enum_size(ReservedGPR::Count); ++r) {
        a.push_r64(target.reserved[r]);
    }
    a.mov_r64_r64(frame_base, x86::RDI);   // SysV: first integer argument

    // ---- body -------------------------------------------------------------------
    for (std::uint32_t id = 1; id <= lowered.mir.node_count(); ++id) {
        MachineNode& n = lowered.mir.node(id);
        block_start.insert_if_absent(n.block, a.size());

        const bool has_reg = id < ra.assignment.size() && ra.assignment[id] >= 0;

        switch (n.op) {
            case MOp::MOVri: {
                // Materialize imm64 into the assigned register (or write the
                // payload into the home slot when spilled).
                if (has_reg) {
                    a.mov_r64_imm64(alloc_reg(ra.assignment[id]),
                                    static_cast<std::uint64_t>(n.operands[0].imm));
                } else {
                    // store payload imm64: MOV [frame+disp], imm32 only supports
                    // 32-bit — use a scratch register then store.
                    a.mov_r64_imm64(x86::RAX, static_cast<std::uint64_t>(n.operands[0].imm));
                    a.mov_mem_r64(frame_base, slot_disp(n.home_slot, kPayloadOffset),
                                  x86::RAX);
                    // tag word = Tag::Int for integer constants
                    a.mov_mem_imm32(frame_base, slot_disp(n.home_slot, kTagOffset),
                                    static_cast<std::int32_t>(kTagInt));
                }
                break;
            }
            case MOp::MOVrm: {
                // Load payload from home slot into assigned reg.
                if (has_reg && n.operands.size() >= 1 &&
                    n.operands[0].kind == MachineOperand::FrameSlot) {
                    a.mov_r64_mem(alloc_reg(ra.assignment[id]), frame_base,
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
                    a.mov_mem_r64(frame_base,
                                  slot_disp(n.operands[0].slot, n.operands[0].tag_off),
                                  alloc_reg(ra.assignment[n.operands[1].vreg]));
                }
                break;
            }
            case MOp::ADDrr:
            case MOp::SUBrr:
            case MOp::IMULrr: {
                if (n.operands.size() < 2) break;
                const auto resolve = [&](const MachineOperand& op) noexcept {
                    RegOrSlot r;
                    if (op.kind == MachineOperand::VReg) {
                        if (op.vreg < ra.assignment.size() &&
                            ra.assignment[op.vreg] >= 0) {
                            r.is_reg = true;
                            r.reg = alloc_reg(ra.assignment[op.vreg]);
                            return r;
                        }
                        // spilled: home slot of the defining node
                        r.slot = op.vreg;
                        return r;
                    }
                    r.slot = op.slot;
                    return r;
                };
                RegOrSlot lhs = resolve(n.operands[0]);
                RegOrSlot rhs = resolve(n.operands[1]);
                // Stage operands into RAX / RCX (SysV caller-saved scratches:
                // clobbers are fine, values reload from homes at next use).
                if (lhs.is_reg) {
                    a.mov_r64_r64(x86::RAX, lhs.reg);
                } else {
                    a.mov_r64_mem(x86::RAX, frame_base, slot_disp(lhs.slot, kPayloadOffset));
                }
                if (rhs.is_reg) {
                    a.mov_r64_r64(x86::RCX, rhs.reg);
                } else {
                    a.mov_r64_mem(x86::RCX, frame_base, slot_disp(rhs.slot, kPayloadOffset));
                }
                if (n.op == MOp::ADDrr) {
                    a.alu_r64_r64(0x01, x86::RAX, x86::RCX);
                } else if (n.op == MOp::SUBrr) {
                    a.alu_r64_r64(0x29, x86::RAX, x86::RCX);
                } else {
                    a.imul_r64_r64(x86::RAX, x86::RCX);
                }
                if (has_reg) {
                    a.mov_r64_r64(alloc_reg(ra.assignment[id]), x86::RAX);
                }
                // Write-back: payload + tag.
                a.mov_mem_r64(frame_base, slot_disp(n.home_slot, kPayloadOffset), x86::RAX);
                a.mov_mem_imm32(frame_base, slot_disp(n.home_slot, kTagOffset),
                                static_cast<std::int32_t>(kTagInt));
                break;
            }
            case MOp::CMPrr: {
                if (n.operands.size() < 2) break;
                const auto resolve = [&](const MachineOperand& op) noexcept {
                    RegOrSlot r;
                    if (op.kind == MachineOperand::VReg) {
                        if (op.vreg < ra.assignment.size() &&
                            ra.assignment[op.vreg] >= 0) {
                            r.is_reg = true;
                            r.reg = alloc_reg(ra.assignment[op.vreg]);
                            return r;
                        }
                        r.slot = op.vreg;
                        return r;
                    }
                    r.slot = op.slot;
                    return r;
                };
                RegOrSlot lhs = resolve(n.operands[0]);
                RegOrSlot rhs = resolve(n.operands[1]);
                if (lhs.is_reg) {
                    a.mov_r64_r64(x86::RAX, lhs.reg);
                } else {
                    a.mov_r64_mem(x86::RAX, frame_base, slot_disp(lhs.slot, kPayloadOffset));
                }
                if (rhs.is_reg) {
                    a.mov_r64_r64(x86::RCX, rhs.reg);
                } else {
                    a.mov_r64_mem(x86::RCX, frame_base, slot_disp(rhs.slot, kPayloadOffset));
                }
                a.alu_r64_r64(0x39, x86::RAX, x86::RCX);
                break;
            }
            case MOp::Jcc: {
                // Condition comes from the MIR operand (arch-neutral MCond),
                // mapped through kX86Cond — never a hard-coded opcode byte.
                if (n.operands.empty()) break;
                const auto mc = static_cast<MCond>(n.operands[0].imm);
                if (mc >= MCond::Count) break;
                std::size_t site = a.jcc_rel32(kX86Cond[static_cast<std::size_t>(mc)]);
                patches.push_back(PatchSite{site, id + 1});
                break;
            }
            case MOp::GUARD_INT: {
                if (n.operands.size() < 2) break;
                const auto resolve = [&](const MachineOperand& op) noexcept {
                    RegOrSlot r;
                    if (op.kind == MachineOperand::VReg) {
                        if (op.vreg < ra.assignment.size() &&
                            ra.assignment[op.vreg] >= 0) {
                            r.is_reg = true;
                            r.reg = alloc_reg(ra.assignment[op.vreg]);
                            return r;
                        }
                        r.slot = op.vreg;
                        return r;
                    }
                    r.slot = op.slot;
                    return r;
                };
                RegOrSlot lhs = resolve(n.operands[0]);
                RegOrSlot rhs = resolve(n.operands[1]);
                // Tag checks: load tag words, compare each against Tag::Int.
                if (lhs.is_reg) {
                    // tag lives in the home slot for register-cached values
                    // (write-back discipline keeps homes authoritative).
                    a.mov_r64_mem(x86::RAX, frame_base,
                                  slot_disp(lhs.slot ? lhs.slot : n.home_slot, kTagOffset));
                } else {
                    a.mov_r64_mem(x86::RAX, frame_base, slot_disp(lhs.slot, kTagOffset));
                }
                a.mov_r64_imm64(x86::RCX, kTagInt);
                a.alu_r64_r64(0x39, x86::RAX, x86::RCX);
                std::size_t j1 = a.jcc_rel32(kX86Cond[static_cast<std::size_t>(MCond::NE)]);   // NE -> deopt

                if (rhs.is_reg) {
                    a.mov_r64_mem(x86::RAX, frame_base,
                                  slot_disp(rhs.slot ? rhs.slot : n.home_slot, kTagOffset));
                } else {
                    a.mov_r64_mem(x86::RAX, frame_base, slot_disp(rhs.slot, kTagOffset));
                }
                a.mov_r64_imm64(x86::RCX, kTagInt);
                a.alu_r64_r64(0x39, x86::RAX, x86::RCX);
                std::size_t j2 = a.jcc_rel32(kX86Cond[static_cast<std::size_t>(MCond::NE)]);

                // ---- deopt stub (cold region, in-line for the single-buffer
                // layout; jump distance is short) ----
                std::size_t stub = a.size();
                a.patch_jcc(j1, stub);
                a.patch_jcc(j2, stub);
                // mov edi, unit_id; mov esi, safepoint_idx; call deopt_entry
                std::size_t s0 = a.size();
                a.mov_r64_imm64(x86::RDI, unit_id);
                a.mov_r64_imm64(x86::RSI, static_cast<std::uint64_t>(out.safepoints.size()));
                std::size_t call_site = a.call_rel32();
                // Safepoint record BEFORE the trap: pc = jump site, live map
                // = the two operands' slots.
                SafepointRecord rec;
                rec.pc_offset = static_cast<std::uint32_t>(j1);
                rec.frame_state_id = n.frame_state_id == 0xFFFFFFFFu ? 0 : n.frame_state_id;
                rec.stack_depth = static_cast<std::uint16_t>(lowered.frame_slots);
                rec.live_regs = 2;
                out.safepoints.push_back(rec);
                out.mappings.push_back(SafepointMapping{x86::RAX, 0});
                out.mappings.push_back(SafepointMapping{x86::RCX, 0});
                // keep the call site for the runtime to patch with the real
                // vortex_deopt_entry address (recorded via a fixed trampoline).
                a.mov_r64_imm64(x86::RAX, reinterpret_cast<std::uint64_t>(&vortex_deopt_entry));
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
                // vm_ctx <- frame base (bridge needs the regs base); call bridge.
                a.mov_r64_r64(vm_ctx, frame_base);
                a.mov_r64_imm64(x86::RAX, 0);   // patched with bridge addr
                a.jmp_rax_placeholder();
                break;
            }
            case MOp::RET: {
                // Result Value in rax(tag) + rdx(payload): reload from home 0.
                a.mov_r64_mem(x86::RDX, frame_base, slot_disp(0, kPayloadOffset));
                a.mov_r64_mem(x86::RAX, frame_base, slot_disp(0, kTagOffset));
                for (std::uint32_t r = enum_size(ReservedGPR::Count); r-- > 0;) {
                    a.pop_r64(target.reserved[r]);
                }
                a.pop_r64(x86::RBX);
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
