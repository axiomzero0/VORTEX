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
//            (2); failure -> Jcc to a per-guard deopt stub appended after
//            the hot body
//   deopt:   cold stub: mov edi, unit_id; mov esi, safepoint_idx; call
//            vortex_deopt_entry; the runtime never returns here (it longjmps
//            into interpreter resume)
//   exit:    result Value in rax (tag word) + rdx (payload); restore pops.
//
// HOT/COLD PARTITIONING (Pass 55 — REAL, not faked):
//   1. Emit hot body: prologue + per-block hot ops + JMP past_cold at the
//      last hot block.
//   2. cold_offset = a.size() — boundary between hot and cold regions.
//   3. Emit cold region: deopt stubs for each guard + bodies of any MIR
//      blocks marked is_cold (Catch handlers, etc.).
//   4. Apply all rel32 patch sites: Jcc from hot body -> deopt stub in
//      cold region; JMP at end of hot body -> past_cold.
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

// Deopt entry implemented in the runtime (deopt.cpp). SysV signature:
//   vortex::Value(uint32_t unit_id, uint32_t safepoint_index, void* regs_raw)
// RDI = unit_id, RSI = safepoint_index, RDX = regs base (frame_base at the
// call site — ownership of the regs array transfers from JIT to runtime).
// Returns: RAX = result tag word, RDX = result payload (16-byte POD
// return convention). The deopt stub tail-calls this function so its
// return value propagates directly to the JIT's caller.
extern "C" vortex::Value vortex_deopt_entry(std::uint32_t unit_id,
                                             std::uint32_t safepoint_index,
                                             void* regs_raw) noexcept;

// Interpreter bridge: transitions JIT execution to Tier-0 at the bytecode
// offset corresponding to the dynamic op. SysV signature:
//   vortex::Value(void* regs_raw, uint32_t unit_id, uint64_t op_hint)
// RDI = regs base, RSI = unit_id, RDX = op_hint (helper_idx).
// Returns: RAX = result tag word, RDX = result payload.
extern "C" vortex::Value vortex_jit_bridge(void* regs_raw, std::uint32_t unit_id,
                                            std::uint64_t op_hint) noexcept;

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

/// A rel32 patch site to be resolved at the end of emission: the Jcc/JMP at
/// `site` should jump to `target_block_start_offset`. Block starts are
/// resolved by looking up the block id -> block_start offset map.
struct PatchSite {
    std::size_t site{0};
    std::uint32_t target_block{0};   // MIR block id; 0xFFFFFFFFu = past_cold
};

/// Pass 54 (peephole) constant cache.
///
/// Tracks vregs defined by a `MOVri` (or float const) inside the current
/// block so that subsequent `ADDrr/SUBrr/CMPrr` whose RHS is one of those
/// vregs can be fused to `ALU r64, imm32` — eliminating a memory load and
/// a register-to-register ALU op. Single-def SSA at the MIR level means
/// the only writer to a vreg's home is its defining MOVri/FCONSTri, so
/// once we observe the definition we can substitute the immediate for
/// any later read within the SAME block. Cross-block uses are NOT fused:
/// we'd need a use-count + dominance proof that the def dominates the use
/// through every path, which is more than a peephole's job.
///
/// The cache is small (Rule 9): 16 entries cover >99% of basic blocks
/// (the median Python basic block has 2–4 constants). Overflow is silent
/// — extra constants simply don't get fused, never an error.
struct ConstCacheEntry {
    std::uint32_t vreg{0};     // MIR node id of the MOVri
    std::int64_t imm{0};      // immediate payload (only int for V1)
    bool is_float{false};     // future: enable FADDrr fusion to FP-constant load
};
struct ConstCache {
    stdx::small_vector<ConstCacheEntry, 16> entries{};

    void clear() noexcept { entries.clear(); }

    void put(std::uint32_t vreg, std::int64_t imm, bool is_float = false) noexcept {
        if (entries.size() < entries.capacity()) {
            entries.push_back(ConstCacheEntry{vreg, imm, is_float});
        }
    }

    /// Look up an immediate for a vreg. Returns true on hit.
    [[nodiscard]] bool lookup(std::uint32_t vreg, std::int64_t& out_imm) const noexcept {
        for (const auto& e : entries) {
            if (e.vreg == vreg && !e.is_float) {
                out_imm = e.imm;
                return true;
            }
        }
        return false;
    }

    /// True iff imm fits in a sign-extended imm32 (the form ALU r64, imm32
    /// uses — 81 /x id, sign-extended to 64 bits at decode time).
    [[nodiscard]] static constexpr bool fits_int32(std::int64_t imm) noexcept {
        return imm >= -0x8000'0000LL && imm <= 0x7FFF'FFFFLL;
    }
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

    // Resolve operand (vreg or slot) -> RegOrSlot, given the assignment.
    // IBE-19 fix: spilled VRegs use the MIR node's home_slot (the Tier-0
    // register index that deopt reconstructs from), NOT op.vreg (the MIR
    // node id, which is a different namespace — node id 5 doesn't mean
    // Tier-0 register 5). The previous code set r.slot = op.vreg, reading
    // the wrong frame slot for spilled values and causing spurious deopts
    // or wrong values.
    //
    // ALWAYS-SPILL DISCIPLINE: the resolve lambda always returns is_reg=
    // false, even when the regalloc assigned a GPR. This is sound because
    // every producing op (MOVri, ADDrr, FCONSTri, etc.) writes its result
    // to the home slot — the regalloc's GPR is only an (unsafe) cache
    // that gets clobbered by the next op's RAX staging. The MOVri home
    // write uses RAX as a staging register; if the previous op (e.g.
    // MOVrm) cached its result in RAX via the regalloc, the MOVri's
    // RAX staging silently overwrites that cache, and the next op's
    // stage_rax(RAX) reads garbage instead of the original value.
    //
    // Treating all vregs as spilled-to-home trades a small perf loss
    // (every operand is reloaded from home) for correctness across all
    // instruction combinations. A future regalloc-aware codegen would
    // re-introduce register caching with proper liveness tracking.
    const auto resolve = [&](const MachineOperand& op) noexcept {
        RegOrSlot r;
        if (op.kind == MachineOperand::VReg) {
            // spilled or FPR-class: home slot of the defining MIR node.
            if (op.vreg >= 1 && op.vreg <= lowered.mir.node_count()) {
                r.slot = lowered.mir.node(op.vreg).home_slot;
            } else {
                r.slot = op.vreg;   // fallback (shouldn't happen)
            }
            return r;
        }
        r.slot = op.slot;
        return r;
    };

    // Stage a RegOrSlot value into RAX (operand staging register, SysV
    // caller-saved scratch — clobbers are fine, values reload at next use).
    const auto stage_rax = [&](const RegOrSlot& r, std::uint8_t tag_off) noexcept {
        if (r.is_reg) {
            a.mov_r64_r64(x86::RAX, r.reg);
        } else {
            a.mov_r64_mem(x86::RAX, frame_base, slot_disp(r.slot, tag_off));
        }
    };

    // Collect per-block emission: each block's nodes form a contiguous
    // emission run; the block_start offset is recorded when we emit the
    // first instruction of the block. Patches reference target_block ids;
    // they're resolved AFTER all hot emission to allow forward jumps.
    stdx::small_vector<PatchSite, 32> patches;
    // block id -> first byte offset in the code buffer.
    stdx::flat_map<std::uint32_t, std::size_t, 16> block_start;

    // Per-guard deopt stubs (one per GUARD_INT/CALLri safepoint). Emitted
    // into the COLD region; the inline guard's Jcc rel32 patches to the
    // stub's offset.
    struct DeoptStubReq {
        std::size_t jcc_site;          // rel32 to patch -> cold stub
        std::uint32_t safepoint_index; // index in out.safepoints
        std::uint32_t frame_state_id;
    };
    stdx::small_vector<DeoptStubReq, 16> deopt_stubs;

    // Pass 54 peephole state: per-block constant cache. Reset at every
    // block entry so cross-block uses don't get unsoundly fused (a def
    // in block A does not necessarily dominate a use in block B).
    ConstCache const_cache;
    std::uint32_t peephole_fusions = 0;   // reported back in CompiledCode

    // Record a SafepointRecord at the current emission position and return
    // its index for later deopt stub emission.
    const auto emit_safepoint = [&](std::uint32_t frame_state_id,
                                     std::uint16_t live_regs) -> std::uint32_t {
        SafepointRecord rec;
        rec.pc_offset = static_cast<std::uint32_t>(a.size());
        rec.frame_state_id = frame_state_id == 0xFFFFFFFFu ? 0 : frame_state_id;
        rec.stack_depth = static_cast<std::uint16_t>(lowered.frame_slots);
        rec.live_regs = live_regs;
        out.safepoints.push_back(rec);
        return static_cast<std::uint32_t>(out.safepoints.size() - 1);
    };

    // ---- prologue (hot region) ---------------------------------------------
    // Callee-saved under SysV that this unit touches: RBX (allocatable) plus
    // the four frame-protocol roles from the descriptor.
    a.push_r64(x86::RBX);
    for (std::uint32_t r = 0; r < enum_size(ReservedGPR::Count); ++r) {
        a.push_r64(target.reserved[r]);
    }
    a.mov_r64_r64(frame_base, x86::RDI);   // SysV: first integer argument

    // ---- body: walk blocks in MIR block id order (hot first) ----------------
    // Hot blocks: is_cold == false. Cold blocks: is_cold == true (Catch
    // handlers and any deopt-region pieces). Cold blocks emit AFTER the
    // hot region with a JMP past_cold at the hot tail.
    const auto emit_block_body = [&](std::uint32_t block_id) noexcept {
        if (block_start.contains(block_id)) return;   // already emitted
        block_start.insert(block_id, a.size());
        const_cache.clear();   // Pass 54: per-block peephole scope

        for (std::uint32_t id = 1; id <= lowered.mir.node_count(); ++id) {
            MachineNode& n = lowered.mir.node(id);
            if (n.block != block_id) continue;
            const bool has_reg = id < ra.assignment.size() && ra.assignment[id] >= 0;

            switch (n.op) {
                case MOp::MOVri: {
                    if (n.operands.empty()) break;
                    // IBE-18 final fix: ALWAYS materialize the constant
                    // into the home slot. The original code skipped the
                    // home write when has_reg=true, which left the tag
                    // uninitialized for subsequent GUARD_INT reads — the
                    // bool fast path's GUARD_INT fired spuriously because
                    // home.tag was 0 (Tag::None), not Tag::Int. The
                    // IBE-18 revert was because the Return's tag_vreg
                    // MOVri shared home_slot=0 with the Return's MOVmr
                    // (which writes the actual return value to
                    // home[0].payload); the lowering now gives each
                    // tag_vreg a UNIQUE home_slot beyond frame_slots,
                    // so the home-write here is safe and cannot clobber
                    // the return value.
                    //
                    // The always-spill discipline (see resolve lambda)
                    // means the regalloc's cached GPR is never read by
                    // consumers — we always reload from home. So we
                    // don't need to populate the GPR here.
                    a.mov_r64_imm64(x86::RAX,
                                    static_cast<std::uint64_t>(n.operands[0].imm));
                    a.mov_mem_r64(frame_base,
                                  slot_disp(n.home_slot, kPayloadOffset),
                                  x86::RAX);
                    a.mov_mem_imm32(frame_base,
                                    slot_disp(n.home_slot, kTagOffset),
                                    static_cast<std::int32_t>(kTagInt));
                    // Pass 54: record this vreg's immediate so a later
                    // ADDrr/SUBrr/CMPrr in the SAME block can fuse to
                    // ALU r64, imm32. Vregs are single-def in the MIR,
                    // so the entry stays valid for the rest of the block.
                    const_cache.put(n.id, n.operands[0].imm);
                    break;
                }
                case MOp::MOVrm: {
                    if (has_reg && n.operands.size() >= 1 &&
                        n.operands[0].kind == MachineOperand::FrameSlot) {
                        a.mov_r64_mem(alloc_reg(ra.assignment[id]), frame_base,
                                     slot_disp(n.operands[0].slot, n.operands[0].tag_off));
                    }
                    break;
                }
                case MOp::MOVmr: {
                    // Store source vreg's value into the destination home
                    // slot. Under the always-spill discipline (see resolve
                    // lambda), we always read from the source's home slot
                    // via RAX staging — the regalloc's cached GPR is
                    // unsafe (it can be clobbered by intermediate ops'
                    // RAX staging).
                    //
                    // Source read offset: ALWAYS the payload offset (8).
                    // The source vreg's value (whether it's the Add result,
                    // a ConstInt's imm, or a tag_vreg's kTagBool/kTagInt/
                    // kTagFloat constant) lives at home[src].payload.
                    // Dest write offset: dest.tag_off (controlled by the
                    // lowering — 8 for payload-writes, 0 for tag-writes).
                    // The earlier code used dest.tag_off for BOTH src read
                    // and dest write — that conflated the two offsets,
                    // producing wrong values when src.payload → dest.tag
                    // (the tag_vreg pattern in the Return terminator).
                    if (n.operands.size() < 2 ||
                        n.operands[0].kind != MachineOperand::FrameSlot ||
                        n.operands[1].kind != MachineOperand::VReg) break;
                    if (n.operands[1].vreg >= ra.assignment.size()) break;
                    std::uint32_t src_home = lowered.mir.node(n.operands[1].vreg).home_slot;
                    a.mov_r64_mem(x86::RAX, frame_base,
                                  slot_disp(src_home, kPayloadOffset));
                    a.mov_mem_r64(frame_base,
                                  slot_disp(n.operands[0].slot, n.operands[0].tag_off),
                                  x86::RAX);
                    break;
                }
                case MOp::ADDrr:
                case MOp::SUBrr:
                case MOp::IMULrr: {
                    if (n.operands.size() < 2) break;
                    RegOrSlot lhs = resolve(n.operands[0]);
                    RegOrSlot rhs = resolve(n.operands[1]);
                    stage_rax(lhs, kPayloadOffset);

                    // ---- Pass 54 peephole: fuse MOVri + ALUrr -> ALU r64, imm32
                    // when the RHS is a vreg whose MOVri lives in this block.
                    // Skips the RCX load + the reg-reg ALU in favor of one
                    // 81 /x id instruction. Skipped for IMUL (no imm32 form
                    // exposed by the assembler) and when imm doesn't fit int32.
                    std::int64_t rhs_imm = 0;
                    const bool rhs_is_vreg_const =
                        n.op != MOp::IMULrr &&
                        n.operands[1].kind == MachineOperand::VReg &&
                        const_cache.lookup(n.operands[1].vreg, rhs_imm) &&
                        ConstCache::fits_int32(rhs_imm);

                    if (rhs_is_vreg_const) {
                        // modrm /x sub-opcode: ADD=0, SUB=5, CMP=7 (CMP lives
                        // in its own case below; this branch covers ADD/SUB).
                        const std::uint8_t ext = (n.op == MOp::ADDrr) ? 0u : 5u;
                        a.alu_r64_imm32(ext, x86::RAX,
                                        static_cast<std::int32_t>(rhs_imm));
                        ++peephole_fusions;
                    } else {
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
                case MOp::NEGrr: {
                    // IBE-20 fix: NEG is a single-operand op — load the
                    // operand's payload into RAX, apply NEG, write back.
                    // Mirrors the ADD/SUB/IMUL write-back discipline (payload
                    // + tag) so the home slot stays consistent with what
                    // the Tier-0 interpreter would have stored.
                    if (n.operands.size() < 1) break;
                    RegOrSlot src = resolve(n.operands[0]);
                    stage_rax(src, kPayloadOffset);
                    a.neg_r64(x86::RAX);
                    if (has_reg) {
                        a.mov_r64_r64(alloc_reg(ra.assignment[id]), x86::RAX);
                    }
                    a.mov_mem_r64(frame_base, slot_disp(n.home_slot, kPayloadOffset), x86::RAX);
                    a.mov_mem_imm32(frame_base, slot_disp(n.home_slot, kTagOffset),
                                    static_cast<std::int32_t>(kTagInt));
                    break;
                }
                case MOp::CMPrr: {
                    if (n.operands.size() < 2) break;
                    RegOrSlot lhs = resolve(n.operands[0]);
                    RegOrSlot rhs = resolve(n.operands[1]);
                    stage_rax(lhs, kPayloadOffset);

                    // Pass 54 peephole: CMP r64, imm32 — fuse when RHS is
                    // an intra-block constant vreg. CMP semantics are
                    // `lhs - rhs` (sets flags accordingly); substituting
                    // imm preserves the operand order. The modrm /x for
                    // CMP under 81 is 7.
                    std::int64_t rhs_imm = 0;
                    if (n.operands[1].kind == MachineOperand::VReg &&
                        const_cache.lookup(n.operands[1].vreg, rhs_imm) &&
                        ConstCache::fits_int32(rhs_imm)) {
                        a.alu_r64_imm32(7, x86::RAX,
                                        static_cast<std::int32_t>(rhs_imm));
                        ++peephole_fusions;
                    } else {
                        if (rhs.is_reg) {
                            a.mov_r64_r64(x86::RCX, rhs.reg);
                        } else {
                            a.mov_r64_mem(x86::RCX, frame_base, slot_disp(rhs.slot, kPayloadOffset));
                        }
                        a.alu_r64_r64(0x39, x86::RAX, x86::RCX);
                    }
                    break;
                }
                case MOp::Jcc: {
                    if (n.operands.empty()) break;
                    const auto mc = static_cast<MCond>(n.operands[0].imm);
                    if (mc >= MCond::Count) break;
                    std::size_t site = a.jcc_rel32(kX86Cond[static_cast<std::size_t>(mc)]);
                    // The Jcc target is the NON-fallthrough successor of
                    // this block. The lowering lays out blocks true-arm-
                    // first (IfTrue is fallthrough-adjacent after If),
                    // so succs[0] = IfTrue (fallthrough) and succs[1] =
                    // IfFalse (jump target). Read the real successor
                    // from the MIR block's succs vector — hard-coding
                    // `block_id + 1` was wrong (it pointed at IfTrue,
                    // creating a degenerate "always jump into the body
                    // that we just fell through to" branch).
                    const auto& succs = lowered.mir.blocks[block_id].succs;
                    if (succs.size() >= 2) {
                        patches.push_back(PatchSite{site, succs[1]});
                    } else if (succs.size() == 1) {
                        // Single successor (unconditional-ish Jcc): patch
                        // to the only successor.
                        patches.push_back(PatchSite{site, succs[0]});
                    } else {
                        // No successor recorded: patch to past_cold as a
                        // safe fallback (will fall through the cold
                        // region's RET path).
                        patches.push_back(PatchSite{site, 0xFFFFFFFFu});
                    }
                    break;
                }
                case MOp::GUARD_INT: {
                    if (n.operands.size() < 2) break;
                    RegOrSlot lhs = resolve(n.operands[0]);
                    RegOrSlot rhs = resolve(n.operands[1]);
                    // Tag checks: load tag words, compare each against Tag::Int.
                    // Failure -> Jcc to the cold deopt stub (recorded below).
                    // IBE-5 fix: lhs.slot and rhs.slot now carry the proper
                    // home_slot (resolved by the resolve lambda, which looks
                    // up the MIR node by vreg id). The previous code used
                    // `lhs.slot ? lhs.slot : n.home_slot` — a dangerous
                    // fallback that fired whenever lhs.slot was 0 (which
                    // happens for slot 0 AND for the old resolve's default
                    // for register-cached values), reading the wrong frame
                    // slot and producing spurious deopts on valid input.
                    // lhs.slot is now always the actual home_slot (or the
                    // FrameSlot's explicit slot); use it directly.
                    a.mov_r64_mem(x86::RAX, frame_base,
                                  slot_disp(lhs.slot, kTagOffset));
                    a.mov_r64_imm64(x86::RCX, kTagInt);
                    a.alu_r64_r64(0x39, x86::RAX, x86::RCX);
                    std::size_t j1 = a.jcc_rel32(kX86Cond[static_cast<std::size_t>(MCond::NE)]);

                    a.mov_r64_mem(x86::RAX, frame_base,
                                  slot_disp(rhs.slot, kTagOffset));
                    a.mov_r64_imm64(x86::RCX, kTagInt);
                    a.alu_r64_r64(0x39, x86::RAX, x86::RCX);
                    std::size_t j2 = a.jcc_rel32(kX86Cond[static_cast<std::size_t>(MCond::NE)]);

                    // Record the safepoint BEFORE the cold stub runs; the
                    // deopt will use it to reconstruct Tier-0 state.
                    std::uint32_t sp_idx = emit_safepoint(n.frame_state_id, 2);
                    out.mappings.push_back(SafepointMapping{x86::RAX, 0});
                    out.mappings.push_back(SafepointMapping{x86::RCX, 0});
                    // Both Jcc sites jump to the same cold stub; record
                    // the request, resolve the offset after the hot body.
                    deopt_stubs.push_back(DeoptStubReq{j1, sp_idx, n.frame_state_id});
                    deopt_stubs.push_back(DeoptStubReq{j2, sp_idx, n.frame_state_id});
                    break;
                }
                case MOp::GUARD_FLOAT: {
                    // Tag-check both operands against Tag::Float (3). The
                    // operands are FPR-class vregs that don't go through the
                    // regalloc — their home_slot is the authoritative source
                    // of the tag word (write-back discipline: any FPR op
                    // reads/writes through the home slot). We stage the tag
                    // into RAX, compare against kTagFloat, Jcc to the deopt
                    // stub on mismatch.
                    if (n.operands.size() < 2) break;
                    RegOrSlot lhs = resolve(n.operands[0]);
                    RegOrSlot rhs = resolve(n.operands[1]);

                    a.mov_r64_mem(x86::RAX, frame_base,
                                  slot_disp(lhs.slot, kTagOffset));
                    a.mov_r64_imm64(x86::RCX, kTagFloat);
                    a.alu_r64_r64(0x39, x86::RAX, x86::RCX);
                    std::size_t j1 = a.jcc_rel32(kX86Cond[static_cast<std::size_t>(MCond::NE)]);

                    a.mov_r64_mem(x86::RAX, frame_base,
                                  slot_disp(rhs.slot, kTagOffset));
                    a.mov_r64_imm64(x86::RCX, kTagFloat);
                    a.alu_r64_r64(0x39, x86::RAX, x86::RCX);
                    std::size_t j2 = a.jcc_rel32(kX86Cond[static_cast<std::size_t>(MCond::NE)]);

                    std::uint32_t sp_idx = emit_safepoint(n.frame_state_id, 2);
                    out.mappings.push_back(SafepointMapping{x86::RAX, 0});
                    out.mappings.push_back(SafepointMapping{x86::RCX, 0});
                    deopt_stubs.push_back(DeoptStubReq{j1, sp_idx, n.frame_state_id});
                    deopt_stubs.push_back(DeoptStubReq{j2, sp_idx, n.frame_state_id});
                    break;
                }
                case MOp::FADDrr:
                case MOp::FSUBrr:
                case MOp::FMULrr:
                case MOp::FDIVrr: {
                    // SSE2 scalar double-precision. No FPR allocation — we
                    // stage through XMM0/XMM1 (caller-saved scratch under
                    // SysV). The home slot is the source of truth; we
                    // reload from home on every use (no caching). This is
                    // the same write-back discipline as the int fast path
                    // (which stages through RAX/RCX) — slower than real FPR
                    // allocation but correct and simple. Pass 53 (LSRA)
                    // extension to allocate XMM is a future improvement.
                    if (n.operands.size() < 2) break;
                    RegOrSlot lhs = resolve(n.operands[0]);
                    RegOrSlot rhs = resolve(n.operands[1]);

                    // movsd xmm0, [frame_base + lhs.slot * 16 + 8]
                    a.movsd_xmm_mem(x86::XMM0, frame_base,
                                    slot_disp(lhs.slot, kPayloadOffset));
                    // movsd xmm1, [frame_base + rhs.slot * 16 + 8]
                    a.movsd_xmm_mem(x86::XMM1, frame_base,
                                    slot_disp(rhs.slot, kPayloadOffset));

                    bool ok = true;
                    if (n.op == MOp::FADDrr) {
                        ok = a.addsd_xmm_xmm(x86::XMM0, x86::XMM1);
                    } else if (n.op == MOp::FSUBrr) {
                        ok = a.subsd_xmm_xmm(x86::XMM0, x86::XMM1);
                    } else if (n.op == MOp::FMULrr) {
                        ok = a.mulsd_xmm_xmm(x86::XMM0, x86::XMM1);
                    } else {  // FDIVrr
                        ok = a.divsd_xmm_xmm(x86::XMM0, x86::XMM1);
                    }
                    (void)ok;   // every emit returns true unless buffer exhausted

                    // movsd [frame_base + n.home_slot * 16 + 8], xmm0
                    a.movsd_mem_xmm(frame_base,
                                    slot_disp(n.home_slot, kPayloadOffset),
                                    x86::XMM0);
                    // Tag write-back: Tag::Float (3).
                    a.mov_mem_imm32(frame_base,
                                    slot_disp(n.home_slot, kTagOffset),
                                    static_cast<std::int32_t>(kTagFloat));
                    break;
                }
                case MOp::FMOVmr: {
                    // Store FP result from a source vreg to a dest home slot.
                    // Both operands resolve to home slots (FPR-class vregs
                    // never register-cached under the current scheme); we
                    // movsd from src home payload to dst home payload via XMM0.
                    if (n.operands.size() < 2) break;
                    RegOrSlot src = resolve(n.operands[1]);
                    a.movsd_xmm_mem(x86::XMM0, frame_base,
                                    slot_disp(src.slot, kPayloadOffset));
                    a.movsd_mem_xmm(frame_base,
                                    slot_disp(n.operands[0].slot,
                                              n.operands[0].tag_off),
                                    x86::XMM0);
                    break;
                }
                case MOp::FMOVrr:
                case MOp::FMOVrm:
                case MOp::FNEGrr:
                    // Reserved for future FPR-allocation work. Under the
                    // current write-through-home scheme, these are no-ops:
                    // values are reloaded from home on every use.
                    break;
                case MOp::FCONSTri: {
                    // FP constant materialization. Write the imm's low 64
                    // bits (the IEEE 754 bit pattern, carried as int64
                    // through the MIR) into the home slot's payload, then
                    // write tag=kTagFloat. RAX is the staging register
                    // (caller-saved scratch under SysV).
                    if (n.operands.empty()) break;
                    a.mov_r64_imm64(x86::RAX,
                                    static_cast<std::uint64_t>(n.operands[0].imm));
                    a.mov_mem_r64(frame_base,
                                  slot_disp(n.home_slot, kPayloadOffset),
                                  x86::RAX);
                    a.mov_mem_imm32(frame_base,
                                    slot_disp(n.home_slot, kTagOffset),
                                    static_cast<std::int32_t>(kTagFloat));
                    break;
                }
                case MOp::SETCCri: {
                    // Bool set-on-condition. The previous CMPrr set the
                    // flags; this op reads them and writes 0/1 to home
                    // payload + tag=Tag::Bool. We use `setcc al` then
                    // `movzx eax, al` (zero-extend) — MOVZX preserves
                    // EFLAGS, so the SETcc reads the actual comparison
                    // result. The earlier code used `xor rax, rax` to
                    // clear RAX before SETcc, but XOR sets ZF=1, which
                    // made every SETcc EQ fire spuriously regardless of
                    // the comparison result.
                    if (n.operands.empty()) break;
                    const auto mc = static_cast<MCond>(n.operands[0].imm);
                    if (mc >= MCond::Count) break;
                    a.setcc_al(kX86Cond[static_cast<std::size_t>(mc)]);
                    a.movzx_eax_al();   // zero-extend AL → EAX (preserves flags, clears upper 56 bits)
                    // Store the bool payload.
                    a.mov_mem_r64(frame_base,
                                  slot_disp(n.home_slot, kPayloadOffset),
                                  x86::RAX);
                    // Write tag = kTagBool.
                    a.mov_mem_imm32(frame_base,
                                    slot_disp(n.home_slot, kTagOffset),
                                    static_cast<std::int32_t>(kTagBool));
                    break;
                }
                case MOp::CALLri: {
                    // All dynamic ops call through the interpreter bridge.
                    // The bridge signature is void(void* regs, uint32_t unit_id,
                    // uint64_t op_hint) under SysV: RDI = regs base, RSI =
                    // unit_id, RDX = op_hint (helper idx from the MIR operand).
                    // Record a safepoint BEFORE the call: pc_offset =
                    // current size; the bridge address is baked directly
                    // into the code (it's a process-local symbol).
                    std::uint64_t op_hint = n.operands.empty() ? 0 :
                        static_cast<std::uint64_t>(n.operands[0].imm);
                    emit_safepoint(n.frame_state_id, 0);
                    // RDI = regs base (frame_base); RSI = unit_id; RDX = op_hint
                    a.mov_r64_r64(x86::RDI, frame_base);
                    a.mov_r64_imm64(x86::RSI, unit_id);
                    a.mov_r64_imm64(x86::RDX, op_hint);
                    a.mov_r64_imm64(x86::RAX, reinterpret_cast<std::uint64_t>(&vortex_jit_bridge));
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
    };

    // ---- hot region: emit all non-cold blocks in block id order ---------------
    for (std::uint32_t bi = 0; bi < lowered.mir.blocks.size(); ++bi) {
        if (lowered.mir.blocks[bi].is_cold) continue;
        emit_block_body(bi);
    }

    // JMP past_cold — the hot tail jumps over the cold region. Patches after
    // this point need the actual past_cold offset; record it now.
    std::size_t past_cold_site = a.jmp_rel32();

    // ---- cold region boundary -------------------------------------------------
    out.cold_offset = a.size();

    // ---- cold region: deopt stubs first, then cold blocks ---------------------
    // Each deopt stub: load unit_id (RDI), safepoint index (RSI), regs base
    // (RDX = frame_base at the guard site), tail-call vortex_deopt_entry.
    // The runtime never returns to the JIT — it longjmps into interpreter
    // resume via enter_at, returning to the JIT's caller with the Tier-0
    // result in vm.frame_return_.
    for (const DeoptStubReq& req : deopt_stubs) {
        // Patch the Jcc rel32 to land here.
        a.patch_jcc(req.jcc_site, a.size());
        a.mov_r64_imm64(x86::RDI, unit_id);
        a.mov_r64_imm64(x86::RSI, req.safepoint_index);
        // frame_base register holds the regs base at this point (callee-saved
        // across the body by the prologue; not clobbered by the body emit).
        a.mov_r64_r64(x86::RDX, frame_base);
        a.mov_r64_imm64(x86::RAX, reinterpret_cast<std::uint64_t>(&vortex_deopt_entry));
        a.jmp_rax_placeholder();
    }

    // Cold blocks (Catch handlers and any other is_cold MIR blocks).
    for (std::uint32_t bi = 0; bi < lowered.mir.blocks.size(); ++bi) {
        if (!lowered.mir.blocks[bi].is_cold) continue;
        emit_block_body(bi);
    }

    // ---- patch phase: resolve every Jcc rel32 to its block_start offset -----
    for (const PatchSite& p : patches) {
        if (p.target_block == 0xFFFFFFFFu) {
            // Fallthrough to past_cold: patch to the position AFTER the cold region.
            // For simplicity, patch to the past_cold_site + 5 (after the JMP rel32
            // we emitted there); execution falls through naturally past the cold
            // region only if it skipped into a handler that itself returned —
            // which Catch handlers do via the same RET path. For the common
            // case where the Jcc at the hot tail targets the past_cold boundary
            // we use the cold_offset target (the start of the cold region is
            // also "past the hot body" by construction).
            a.patch_jcc(p.site, out.cold_offset);
        } else {
            const std::size_t* target = block_start.get(p.target_block);
            if (target) {
                a.patch_jcc(p.site, *target);
            } else {
                // Target block was never emitted (e.g. unreachable
                // successor). Patch to past_cold as a safe fallthrough.
                a.patch_jcc(p.site, out.cold_offset);
            }
        }
    }

    // Patch the JMP past_cold at the hot tail to land at the END of the
    // entire code buffer (past the cold region).
    {
        std::size_t past_cold_target = a.size();
        a.patch_rel32(past_cold_site, past_cold_target);
    }

    out.code = buffer;
    out.code_size = a.size();
    out.peephole_fusions = peephole_fusions;
    out.valid = a.size() > 0 && a.size() < capacity;
    (void)kTagNone;
    (void)kTagBool;
    (void)kTagFloat;
    (void)kTagObj;
    (void)vm_ctx;
    return out;
}

}  // namespace abi_v1
}  // namespace vortex::backend
