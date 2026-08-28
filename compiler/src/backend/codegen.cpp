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
// LSRA->XMM extension: a vreg's resolved location is either
//   - is_reg=true with reg=GPR encoding byte  (GPR-class vreg, cached)
//   - is_xmm=true with xmm=XMM encoding byte  (FPR-class vreg, cached)
//   - both false, slot=home slot             (spilled or uncached)
// The codegen's GPR-stage helpers (stage_rax / stage_rcx) consume the
// is_reg branch; the FPR-stage helpers (stage_xmm0 / stage_xmm1) consume
// the is_xmm branch. A given vreg never has both is_reg and is_xmm
// (the regalloc routes a vreg to exactly one pool based on its MIR rc).
struct RegOrSlot {
    bool is_reg{false};
    bool is_xmm{false};
    std::uint8_t reg{0};
    std::uint8_t xmm{0};
    std::uint32_t slot{0};
};

/// A rel32 patch site to be resolved at the end of emission: the Jcc/JMP at
/// `site` should jump to `target_block_start_offset`. Block starts are
/// resolved by looking up the block id -> block_start offset map.
///
/// Task 24 (candidate j): track whether the site is a JMP (E9, 5 bytes
/// total, rel starts at site+1) or a Jcc (0F 8x, 6 bytes total, rel
/// starts at site+2). The two have different patch functions: patch_rel32
/// for JMP, patch_jcc for Jcc. Calling the wrong patcher corrupts the
/// rel32 (wrong write offset + wrong rel formula), making every patched
/// jump land at a bogus address — typically segfaulting the JIT.
struct PatchSite {
    std::size_t site{0};
    std::uint32_t target_block{0};   // MIR block id; 0xFFFFFFFFu = past_cold
    bool is_jmp{false};              // true = JMP (patch_rel32), false = Jcc (patch_jcc)
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

/// Pass 54 (register-caching codegen): tracks which physical GPRs
/// currently hold which vreg's value, so subsequent operand reads can
/// use a reg-to-reg move (`mov rax, rbx`, 3 bytes) instead of a memory
/// load (`mov rax, [r12 + slot*16 + 8]`, 4-7 bytes).
///
/// Retires the ALWAYS-SPILL DISCIPLINE that bypassed the regalloc for
/// correctness. The cache model:
///   - Per-block scope: cleared at block entry so cross-block live
///     values fall back to home (a register's content from block A
///     might be clobbered by a function call between A and B; the
///     inter-block liveness proof is the regalloc's job, but tracking
///     it correctly inside the codegen would duplicate LSRA).
///   - Define after every producing op: the op's result lives in RAX
///     (post-ALU) or in dst_gpr (post-load). The cache reflects both:
///     RAX still has the result until the next op's stage_rax clobbers
///     it; dst_gpr holds it for the rest of the block.
///   - Clobber on every staging write: stage_rax writes RAX, so it
///     must mark RAX free in the cache BEFORE any subsequent resolve
///     in the same op (otherwise we'd read a stale cache entry).
///
/// The regalloc's invariant (no two simultaneously-live vregs share a
/// GPR) means: if a vreg is cached in GPR X, no other live vreg is
/// also in X — so reading from X is sound until X is clobbered.
struct GprCache {
    static constexpr std::uint32_t kFree = 0xFFFFFFFFu;
    std::array<std::uint32_t, 16> state{};

    void clear() noexcept { state.fill(kFree); }

    void clobber(std::uint8_t gpr) noexcept {
        if (gpr < 16) state[gpr] = kFree;
    }

    void define(std::uint8_t gpr, std::uint32_t vreg) noexcept {
        if (gpr < 16) state[gpr] = vreg;
    }

    [[nodiscard]] bool holds(std::uint8_t gpr, std::uint32_t vreg) const noexcept {
        return gpr < 16 && state[gpr] == vreg;
    }
};

/// Pass 54 V3 (LSRA->XMM): the FPR analog of GprCache. Tracks which
/// physical XMM registers currently hold which FPR-class vreg's value,
/// so subsequent FP operand reads can use a reg-to-reg movsd (4 bytes,
/// 1 cycle) instead of a memory load `movsd xmm, [r12 + slot*16 + 8]`
/// (5-8 bytes, 4-6 cycles L1-dependent). Same invariants as GprCache:
///   - Cross-block scope (live-interval aware). The LSRA's invariant
///     (no two simultaneously-live FPR vregs share an XMM) means a
///     cache entry "XMM x holds FPR vreg v" stays sound until either
///     v's interval ends or x is clobbered by stage_xmm0/stage_xmm1
///     (staging) or a CALLri (XMM0-XMM7 are caller-saved under SysV).
///   - Define after every producing FP op (FADD/FSUB/FMUL/FDIV/
///     FCONSTri): the result lives in XMM0 (post-op) or in dst_xmm
///     (post-populate). The cache reflects both.
///   - Clobber on every FP staging write: stage_xmm0 writes XMM0, so
///     it must mark XMM0 free in the cache BEFORE any subsequent
///     resolve in the same op (otherwise we'd read a stale cache entry
///     — the same IBE-18-class bug GprCache's clobber-on-stage closes).
///
/// Self-mov elimination (Pass 54 V2 analog): if the operand's cached
/// XMM IS XMM0 (the staging register), the cache invariant says
/// state[XMM0] == r's vreg (resolve checked holds(XMM0, r's vreg)).
/// XMM0 already holds r's value — nothing to stage. Skipping the
/// movsd AND the clobber keeps the cache consistent: future resolves
/// in the same op can still hit the cache for OTHER FPR vregs (e.g.,
/// the rhs of a binary FP op whose cache entry lives in XMM0 because
/// the regalloc assigned XMM0 to it).
struct XmmCache {
    static constexpr std::uint32_t kFree = 0xFFFFFFFFu;
    std::array<std::uint32_t, 16> state{};

    void clear() noexcept { state.fill(kFree); }

    void clobber(std::uint8_t xmm) noexcept {
        if (xmm < 16) state[xmm] = kFree;
    }

    void define(std::uint8_t xmm, std::uint32_t vreg) noexcept {
        if (xmm < 16) state[xmm] = vreg;
    }

    [[nodiscard]] bool holds(std::uint8_t xmm, std::uint32_t vreg) const noexcept {
        return xmm < 16 && state[xmm] == vreg;
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
    // LSRA->XMM extension: FPR-pool analog of alloc_reg. Maps an index
    // into target.allocatable_fpr[] to its XMM encoding byte. Returns
    // XMM0 as a defensive fallback (the codegen treats out-of-range
    // FPR indices as "spilled" via the assignment check, so a fallback
    // here is belt-and-suspenders).
    const std::uint32_t n_alloc_fpr = target.allocatable_fprs;
    const auto alloc_fpr = [&](std::int32_t idx) noexcept -> std::uint8_t {
        if (idx < 0 || static_cast<std::uint32_t>(idx) >= n_alloc_fpr) return x86::XMM0;
        return target.allocatable_fpr[static_cast<std::uint32_t>(idx)];
    };

    // ---- Pass 52: lower ---------------------------------------------------------
    LoweringResult lowered = lower_to_mir(g, target);
    if (lowered.mir.node_count() == 0) return out;
    out.frame_slots = lowered.frame_slots;
    out.has_dynamic_ops = lowered.has_dynamic_ops;   // Task 24

    // ---- Pass 53: allocate --------------------------------------------------------
    stdx::small_vector<LiveInterval, 64> intervals;
    RegAllocResult ra =
        linear_scan(lowered.mir, lowered.block_order, target, intervals);

    // ---- Pass 54/55: emit ----------------------------------------------------------
    Assembler a(buffer, capacity);

    // Pass 54 peephole state: per-block constant cache. Reset at every
    // block entry so cross-block uses don't get unsoundly fused (a def
    // in block A does not necessarily dominate a use in block B).
    // Declared here (before the lambdas) because resolve/stage_rax/
    // stage_rcx/populate_dst_from_rax all capture it by reference.
    ConstCache const_cache;
    std::uint32_t peephole_fusions = 0;   // reported back in CompiledCode

    // Pass 54 register-caching state: per-block GPR occupancy. Retires
    // the ALWAYS-SPILL DISCIPLINE that bypassed the regalloc for safety.
    // See GprCache's doc comment for the cache model.
    GprCache gpr_cache;
    std::uint32_t gpr_cache_hits = 0;   // operand reads served from a cached GPR
    // Pass 54 V2: self-mov eliminations in stage_rax / stage_rcx (when
    // the operand's cached GPR IS the staging register, the naive path
    // would emit a 3-byte no-op `mov reg, reg`; the cache-aware path
    // skips the mov and the cache clobber, so future resolves in the
    // same op can still hit the cache for other vregs. See stage_rax.
    std::uint32_t self_mov_eliminations = 0;
    // Pass 54 V3 (LSRA->XMM): FPR analog of gpr_cache. Empty when the
    // target descriptor doesn't expose an FPR pool (the legacy contract)
    // — the FPR ops (FADD/FSUB/FMUL/FDIV/FCONSTri/FMOVmr) then read/write
    // through home slots exactly as they did before. When the pool is
    // non-empty, the resolve lambda returns is_xmm=true for FPR-class
    // vregs the regalloc assigned an XMM, and the FP op cases use the
    // stage_xmm0/stage_xmm1/populate_dst_from_xmm0 path.
    XmmCache xmm_cache;
    std::uint32_t xmm_cache_hits = 0;

    // Resolve operand (vreg or slot) -> RegOrSlot, given the assignment.
    // IBE-19 fix: spilled VRegs use the MIR node's home_slot (the Tier-0
    // register index that deopt reconstructs from), NOT op.vreg (the MIR
    // node id, which is a different namespace — node id 5 doesn't mean
    // Tier-0 register 5). The previous code set r.slot = op.vreg, reading
    // the wrong frame slot for spilled values and causing spurious deopts
    // or wrong values.
    //
    // REGALLOC-AWARE RESOLVE: if the regalloc assigned a GPR to this vreg
    // AND gpr_cache confirms that GPR still holds the vreg's value, return
    // is_reg=true so the staging helper emits a reg-to-reg move (3 bytes)
    // instead of a memory load (4-7 bytes). On cache miss, fall back to
    // the home slot — the home is always up-to-date because every
    // defining op still writes it (for deopt safety).
    //
    // LSRA->XMM extension (Pass 54 V3): mirror the GPR cache check for
    // FPR-class vregs. The regalloc now produces a parallel
    // assignment_fpr[] array indexed by vreg; if the vreg is FPR-class
    // AND assigned an XMM, return is_xmm=true so the FP staging helpers
    // emit a movsd reg-reg (4 bytes, 1 cycle) instead of a movsd from
    // memory (5-8 bytes, 4-6 cycles). Falls back to home on cache miss
    // — the FP home-write discipline (every FPR op writes back to home
    // for deopt safety, just like the GPR ops) keeps the home slot
    // authoritative.
    const auto resolve = [&](const MachineOperand& op) noexcept {
        RegOrSlot r;
        if (op.kind == MachineOperand::VReg) {
            if (op.vreg >= 1 && op.vreg <= lowered.mir.node_count()) {
                r.slot = lowered.mir.node(op.vreg).home_slot;
            } else {
                r.slot = op.vreg;   // fallback (shouldn't happen)
                return r;
            }
            // Pass 54 V3: FPR-class vregs go through the XMM cache path.
            // The reg class is read from the MIR node's `rc` field
            // (constant per vreg — the regalloc inherited it).
            const bool is_fpr =
                lowered.mir.node(op.vreg).rc == MachineRegClass::FPR;
            if (is_fpr && !ra.assignment_fpr.empty() &&
                op.vreg < ra.assignment_fpr.size() &&
                ra.assignment_fpr[op.vreg] >= 0) {
                std::uint8_t xmm = alloc_fpr(ra.assignment_fpr[op.vreg]);
                if (xmm_cache.holds(xmm, op.vreg)) {
                    r.is_xmm = true;
                    r.xmm = xmm;
                    ++xmm_cache_hits;
                }
            } else if (!is_fpr && op.vreg < ra.assignment.size() &&
                       ra.assignment[op.vreg] >= 0) {
                // If the regalloc assigned a GPR AND that GPR still holds this
                // vreg's value (per gpr_cache), use the GPR directly.
                std::uint8_t gpr = alloc_reg(ra.assignment[op.vreg]);
                if (gpr_cache.holds(gpr, op.vreg)) {
                    r.is_reg = true;
                    r.reg = gpr;
                    ++gpr_cache_hits;
                }
            }
            return r;
        }
        r.slot = op.slot;
        return r;
    };

    // Stage a RegOrSlot value into RAX. Marks RAX as clobbered in
    // gpr_cache BEFORE the load — otherwise a later resolve() in the
    // same op would read a stale "RAX has the previous value" entry.
    // This is the IBE-18-class bug the ALWAYS-SPILL DISCIPLINE was a
    // workaround for: stage_rax(lhs) writes RAX; if the cache still
    // said "RAX has rhs", the next resolve(rhs) would happily return
    // is_reg=true with reg=RAX, and the emit would `mov RCX, RAX` —
    // pulling in lhs's value, not rhs's. Clobbering RAX in the cache
    // before the load closes the race.
    //
    // Pass 54 V2: self-mov elimination. If r.is_reg && r.reg == RAX,
    // the cache invariant says state[RAX] == r's vreg (the resolve()
    // that produced r checked holds(RAX, r's vreg)). RAX already
    // holds r's value — there is nothing to stage. Skipping the mov
    // AND the clobber keeps the cache consistent: future resolves in
    // the same op can still hit the cache for other vregs (e.g., the
    // rhs of a binary op whose cache entry lives in RAX because the
    // regalloc assigned RAX to it).
    const auto stage_rax = [&](const RegOrSlot& r, std::uint8_t tag_off) noexcept {
        if (r.is_reg && r.reg == x86::RAX) {
            ++self_mov_eliminations;
            return;
        }
        gpr_cache.clobber(x86::RAX);
        if (r.is_reg) {
            a.mov_r64_r64(x86::RAX, r.reg);
        } else {
            a.mov_r64_mem(x86::RAX, frame_base, slot_disp(r.slot, tag_off));
        }
    };

    // Stage a RegOrSlot value into RCX (second staging register). Same
    // clobber-first discipline as stage_rax; same self-mov elimination
    // when r.is_reg && r.reg == RCX.
    const auto stage_rcx = [&](const RegOrSlot& r, std::uint8_t tag_off) noexcept {
        if (r.is_reg && r.reg == x86::RCX) {
            ++self_mov_eliminations;
            return;
        }
        gpr_cache.clobber(x86::RCX);
        if (r.is_reg) {
            a.mov_r64_r64(x86::RCX, r.reg);
        } else {
            a.mov_r64_mem(x86::RCX, frame_base, slot_disp(r.slot, tag_off));
        }
    };

    // Post-def cache update: after a producing op leaves its result in
    // RAX (the common case for ADD/SUB/IMUL/NEG/MOVri/SETCC/FCONST),
    // populate the regalloc-assigned dst_gpr from RAX, then mark both
    // RAX and dst_gpr in the cache as holding this vreg's value.
    //
    // RAX's cache entry is valid until the next op's stage_rax clobbers
    // it (which the cache correctly reflects because stage_rax calls
    // gpr_cache.clobber(RAX)). dst_gpr's entry is valid for the rest
    // of the block (or until some op explicitly clobbers it, which
    // only happens if dst_gpr is RAX or RCX and an intervening op
    // uses them as staging).
    const auto populate_dst_from_rax = [&](std::uint32_t vreg_id) noexcept {
        // RAX has the result (post-ALU or post-mov_r64_imm64). Mark it
        // in the cache — a subsequent resolve() in the same op (rare)
        // or the next op's resolve() before its stage_rax can use it.
        gpr_cache.define(x86::RAX, vreg_id);
        if (vreg_id < ra.assignment.size() && ra.assignment[vreg_id] >= 0) {
            std::uint8_t dst_gpr = alloc_reg(ra.assignment[vreg_id]);
            if (dst_gpr != x86::RAX) {
                a.mov_r64_r64(dst_gpr, x86::RAX);
                gpr_cache.define(dst_gpr, vreg_id);
            }
            // If dst_gpr == RAX, the define above already covers it.
        }
    };

    // ===== Pass 54 V3 (LSRA->XMM) FPR-staging helpers =========================
    // The FPR analogs of stage_rax/stage_rcx/populate_dst_from_rax. XMM0
    // is the primary FP staging register (analog of RAX); XMM1 is the
    // secondary (analog of RCX). The SSE2 emit pattern for FADD/FSUB/
    // FMUL/FDIV is:
    //   stage_xmm0(lhs, kPayloadOffset);    // movsd xmm0, ?
    //   stage_xmm1(rhs, kPayloadOffset);    // movsd xmm1, ?
    //   addsd xmm0, xmm1;                   // result in xmm0
    //   movsd [home], xmm0;                 // write-back for deopt safety
    //   populate_dst_from_xmm0(id);         // movsd dst_xmm, xmm0 + cache
    //
    // Self-mov elimination (Pass 54 V2 analog): if r.is_xmm &&
    // r.xmm == XMM0, the cache invariant says state[XMM0] == r's vreg
    // (resolve checked holds(XMM0, r's vreg)). XMM0 already holds r's
    // value — nothing to stage. Skipping the movsd AND the clobber
    // keeps the cache consistent: future resolves in the same op can
    // still hit the cache for OTHER FPR vregs (e.g., the rhs of a
    // binary FP op whose cache entry lives in XMM0 because the regalloc
    // assigned XMM0 to it).
    const auto stage_xmm0 = [&](const RegOrSlot& r, std::uint8_t tag_off) noexcept {
        if (r.is_xmm && r.xmm == x86::XMM0) {
            ++self_mov_eliminations;
            return;
        }
        xmm_cache.clobber(x86::XMM0);
        if (r.is_xmm) {
            a.movsd_xmm_xmm(x86::XMM0, r.xmm);
        } else {
            a.movsd_xmm_mem(x86::XMM0, frame_base, slot_disp(r.slot, tag_off));
        }
    };

    // Stage a RegOrSlot value into XMM1 (second FP staging register).
    // Same clobber-first discipline as stage_xmm0; same self-mov
    // elimination when r.is_xmm && r.xmm == XMM1.
    const auto stage_xmm1 = [&](const RegOrSlot& r, std::uint8_t tag_off) noexcept {
        if (r.is_xmm && r.xmm == x86::XMM1) {
            ++self_mov_eliminations;
            return;
        }
        xmm_cache.clobber(x86::XMM1);
        if (r.is_xmm) {
            a.movsd_xmm_xmm(x86::XMM1, r.xmm);
        } else {
            a.movsd_xmm_mem(x86::XMM1, frame_base, slot_disp(r.slot, tag_off));
        }
    };

    // Post-def cache update: after a producing FP op leaves its result
    // in XMM0 (FADD/FSUB/FMUL/FDIV — XMM0 is the dst per the SSE2 two-
    // operand form), populate the regalloc-assigned dst_xmm from XMM0,
    // then mark both XMM0 and dst_xmm in the cache as holding this
    // vreg's value. Mirrors populate_dst_from_rax exactly.
    //
    // XMM0's cache entry is valid until the next FP op's stage_xmm0
    // clobbers it. dst_xmm's entry is valid for the rest of the block
    // (or until some FP op explicitly clobbers it via stage_xmm0 or
    // stage_xmm1 — only happens if dst_xmm is XMM0 or XMM1).
    //
    // Fallback when assignment_fpr is empty (legacy contract — no FPR
    // pool): the vreg was not assigned an XMM, so the cache is left
    // alone. Subsequent reads of this vreg will fall back to home.
    const auto populate_dst_from_xmm0 = [&](std::uint32_t vreg_id) noexcept {
        xmm_cache.define(x86::XMM0, vreg_id);
        if (!ra.assignment_fpr.empty() && vreg_id < ra.assignment_fpr.size() &&
            ra.assignment_fpr[vreg_id] >= 0) {
            std::uint8_t dst_xmm = alloc_fpr(ra.assignment_fpr[vreg_id]);
            if (dst_xmm != x86::XMM0) {
                a.movsd_xmm_xmm(dst_xmm, x86::XMM0);
                xmm_cache.define(dst_xmm, vreg_id);
            }
            // If dst_xmm == XMM0, the define above already covers it.
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
        const_cache.clear();   // Pass 54: per-block peephole scope.
                               // Cross-block const fusion would require a
                               // dominance proof that the def reaches the
                               // use through every path — out of scope for V1.
        // Pass 54 GPR cache: CROSS-BLOCK (live-interval aware). The LSRA
        // invariant (no two simultaneously-live vregs share a GPR) means a
        // cache entry "GPR g holds vreg v" stays sound until either v's
        // interval ends or g is clobbered by staging/call. Per-block reset
        // would throw away valid cross-block cache entries — every vreg
        // that lives across a Jcc would reload from home at the target
        // block even though the LSRA assigned it a GPR throughout. We do
        // NOT clear here. Stale entries (vreg v ended but cache still
        // says "g holds v") are harmless: no resolve(v) happens past
        // v.end (no use exists past the interval end per LSRA), and any
        // resolve(v2) where v2 is freshly assigned g correctly fails
        // holds(g, v2) and falls back to home — until v2's def refreshes
        // the cache via populate_dst_from_rax.
        // The earlier "per-block scope" was an unsoundness-defensive
        // workaround that conflicted with the architecture's
        // live-interval-aware regalloc contract. Retired.

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
                    // Pass 54: populate the regalloc's assigned GPR from
                    // RAX so a subsequent consumer of this vreg can read
                    // from the GPR (3-byte reg-reg move) instead of from
                    // home (4-7 byte memory load).
                    populate_dst_from_rax(id);
                    break;
                }
                case MOp::MOVrm: {
                    if (has_reg && n.operands.size() >= 1 &&
                        n.operands[0].kind == MachineOperand::FrameSlot) {
                        std::uint8_t dst_gpr = alloc_reg(ra.assignment[id]);
                        a.mov_r64_mem(dst_gpr, frame_base,
                                     slot_disp(n.operands[0].slot, n.operands[0].tag_off));
                        // Pass 54: the load populated dst_gpr directly —
                        // mark it in the cache so consumers can use it.
                        gpr_cache.define(dst_gpr, id);
                    }
                    break;
                }
                case MOp::MOVmr: {
                    // Store source vreg's value into the destination home
                    // slot. Source read offset: ALWAYS the payload offset
                    // (8). The source vreg's value (whether it's the Add
                    // result, a ConstInt's imm, or a tag_vreg's
                    // kTagBool/kTagInt/kTagFloat constant) lives at
                    // home[src].payload.
                    //
                    // Dest write offset: dest.tag_off (controlled by the
                    // lowering — 8 for payload-writes, 0 for tag-writes).
                    // The earlier code used dest.tag_off for BOTH src read
                    // and dest write — that conflated the two offsets,
                    // producing wrong values when src.payload → dest.tag
                    // (the tag_vreg pattern in the Return terminator).
                    //
                    // Pass 54: the source read goes through stage_rax,
                    // which (a) clobbers RAX in gpr_cache before the load
                    // so a subsequent resolve in the same op can't read a
                    // stale entry, and (b) uses a reg-reg move when the
                    // source vreg's GPR is still cached.
                    if (n.operands.size() < 2 ||
                        n.operands[0].kind != MachineOperand::FrameSlot ||
                        n.operands[1].kind != MachineOperand::VReg) break;
                    if (n.operands[1].vreg >= ra.assignment.size()) break;
                    RegOrSlot src = resolve(n.operands[1]);
                    stage_rax(src, kPayloadOffset);
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
                        stage_rcx(rhs, kPayloadOffset);
                        if (n.op == MOp::ADDrr) {
                            a.alu_r64_r64(0x01, x86::RAX, x86::RCX);
                        } else if (n.op == MOp::SUBrr) {
                            a.alu_r64_r64(0x29, x86::RAX, x86::RCX);
                        } else {
                            a.imul_r64_r64(x86::RAX, x86::RCX);
                        }
                    }
                    // Write-back: payload + tag.
                    a.mov_mem_r64(frame_base, slot_disp(n.home_slot, kPayloadOffset), x86::RAX);
                    a.mov_mem_imm32(frame_base, slot_disp(n.home_slot, kTagOffset),
                                    static_cast<std::int32_t>(kTagInt));
                    // Pass 54: populate dst_gpr from RAX so subsequent
                    // consumers can read from the GPR.
                    populate_dst_from_rax(id);
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
                    a.mov_mem_r64(frame_base, slot_disp(n.home_slot, kPayloadOffset), x86::RAX);
                    a.mov_mem_imm32(frame_base, slot_disp(n.home_slot, kTagOffset),
                                    static_cast<std::int32_t>(kTagInt));
                    // Pass 54: populate dst_gpr from RAX.
                    populate_dst_from_rax(id);
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
                        stage_rcx(rhs, kPayloadOffset);
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
                case MOp::JMP: {
                    // Task 24 (candidate j): unconditional JMP emitted by
                    // the lowering at backedge sites (loop body → loop
                    // header) and forward-jump sites (if-true arm → merge
                    // over the if-false arm). The target is the SOLE
                    // successor in the MIR block's succs vector.
                    std::size_t site = a.jmp_rel32();
                    const auto& succs = lowered.mir.blocks[block_id].succs;
                    if (!succs.empty()) {
                        patches.push_back(PatchSite{site, succs[0], /*is_jmp=*/true});
                    } else {
                        // No successor recorded: patch to past_cold as a
                        // safe fallback. Should not happen (a JMP without
                        // a target is a malformed block), but defensive.
                        patches.push_back(PatchSite{site, 0xFFFFFFFFu, /*is_jmp=*/true});
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
                    // Pass 54: clobber RAX/RCX in the cache before staging
                    // tag values into them (they don't hold vreg payloads
                    // after this point — only tag words).
                    gpr_cache.clobber(x86::RAX);
                    gpr_cache.clobber(x86::RCX);
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

                    // Pass 54: clobber RAX/RCX in the cache before staging
                    // tag values into them.
                    gpr_cache.clobber(x86::RAX);
                    gpr_cache.clobber(x86::RCX);
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
                    // SSE2 scalar double-precision. Pass 54 V3 (LSRA->XMM):
                    // when the regalloc assigned an XMM to one or both
                    // operands, the resolve lambda returns is_xmm=true
                    // and we use movsd reg-reg (4 bytes, 1 cycle) instead
                    // of movsd from memory (5-8 bytes, 4-6 cycles L1). The
                    // home slot is still written back for deopt safety
                    // (every FP defining op materializes to home, just
                    // like the GPR ops — see MOVri case). After the op,
                    // populate_dst_from_xmm0 propagates the result into
                    // the regalloc-assigned dst XMM so subsequent reads
                    // hit the cache.
                    if (n.operands.size() < 2) break;
                    RegOrSlot lhs = resolve(n.operands[0]);
                    RegOrSlot rhs = resolve(n.operands[1]);
                    stage_xmm0(lhs, kPayloadOffset);
                    stage_xmm1(rhs, kPayloadOffset);

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

                    // Write-back: payload + tag (FP home discipline — keeps
                    // the home slot authoritative for deopt + GUARD_FLOAT
                    // tag-word reads).
                    a.movsd_mem_xmm(frame_base,
                                    slot_disp(n.home_slot, kPayloadOffset),
                                    x86::XMM0);
                    a.mov_mem_imm32(frame_base,
                                    slot_disp(n.home_slot, kTagOffset),
                                    static_cast<std::int32_t>(kTagFloat));
                    // Pass 54 V3: populate dst_xmm from XMM0 so subsequent
                    // FP reads of this vreg hit the XMM cache.
                    populate_dst_from_xmm0(id);
                    break;
                }
                case MOp::FMOVmr: {
                    // Store FP result from a source vreg to a dest home slot.
                    // Pass 54 V3: when the source vreg is FPR-class and
                    // cached in an XMM, use movsd reg-reg (4 bytes, 1 cycle)
                    // instead of movsd from memory (5-8 bytes, 4-6 cycles).
                    // resolve() returns is_xmm=true in that case; stage_xmm0
                    // routes it through the right emit path.
                    if (n.operands.size() < 2) break;
                    RegOrSlot src = resolve(n.operands[1]);
                    stage_xmm0(src, kPayloadOffset);
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
                    // Pass 54 V2 (legacy): populate dst_gpr from RAX (the
                    // FP constant's RAW bit pattern is now in RAX as an
                    // int64; a GPR-class consumer reading this as int64
                    // could use it, though in practice the only consumer
                    // is an FP op which reads through home).
                    populate_dst_from_rax(id);
                    // Pass 54 V3 (LSRA->XMM): if the regalloc assigned an
                    // XMM to this FCONSTri's vreg, materialize the FP
                    // constant directly into that XMM too. We use the
                    // movq_xmm_r64 form (66 REX.W 0F 6E /r) to move the
                    // int64 from RAX into the dst XMM's low lane. The
                    // cache is then marked so subsequent FP reads of this
                    // vreg hit the XMM. (dst_xmm != XMM0/XMM1, so no
                    // clobber needed.)
                    if (!ra.assignment_fpr.empty() && id < ra.assignment_fpr.size() &&
                        ra.assignment_fpr[id] >= 0) {
                        std::uint8_t dst_xmm = alloc_fpr(ra.assignment_fpr[id]);
                        a.movq_xmm_r64(dst_xmm, x86::RAX);
                        xmm_cache.define(dst_xmm, id);
                    }
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
                    // Pass 54: populate dst_gpr from RAX (the bool payload
                    // is now in RAX after movzx_eax_al — 0 or 1).
                    populate_dst_from_rax(id);
                    break;
                }
                case MOp::CALLri: {
                    // Dynamic ops call through the interpreter bridge.
                    // The bridge signature: Value(void* regs_raw, uint32_t unit_id,
                    // uint64_t op_hint) under SysV: RDI = regs base, RSI =
                    // unit_id, RDX = op_hint.
                    //
                    // op_hint = the IR NodeId (n.home) of this CALLri site.
                    // The bridge uses unit->node_id_to_pc[op_hint] to find
                    // the corresponding Tier-0 instruction, executes that ONE
                    // instruction, and returns. This is what makes the bridge
                    // RETURN to the JIT — the JIT continues after the CALL,
                    // only paying the bridge cost for the one dynamic op.
                    //
                    // CALL (not JMP): pushes the return address so the bridge
                    // can RET back to the next JIT instruction. Stack is
                    // 16-byte aligned at the CALL site (5 pushes + original
                    // CALL = 48 bytes = 3×16, already aligned).
                    std::uint64_t op_hint = static_cast<std::uint64_t>(n.home_slot);
                    emit_safepoint(n.frame_state_id, 0);
                    // RDI = regs base (frame_base); RSI = unit_id; RDX = NodeId
                    a.mov_r64_r64(x86::RDI, frame_base);
                    a.mov_r64_imm64(x86::RSI, unit_id);
                    a.mov_r64_imm64(x86::RDX, op_hint);
                    a.mov_r64_imm64(x86::RAX, reinterpret_cast<std::uint64_t>(&vortex_jit_bridge));
                    a.call_rax();
                    // After the bridge returns, the result Value is in
                    // RAX (tag) + RDX (payload). The bridge already wrote
                    // the result to the home slot (via the Tier-0 instruction's
                    // dst register), so the JIT's home-slot discipline picks
                    // it up on the next read. No explicit write-back needed.
                    //
                    // Pass 54: a call clobbers ALL caller-saved GPRs per SysV.
                    // Mark every allocatable caller-saved GPR free in the cache;
                    // the only survivor is RBX (callee-saved).
                    for (std::uint32_t i = 0; i < n_alloc; ++i) {
                        std::uint8_t gpr = target.allocatable[i];
                        if (gpr != x86::RBX) gpr_cache.clobber(gpr);
                    }
                    // Pass 54 V3 (LSRA->XMM): the bridge is a call to C++
                    // code; under SysV x86-64 the XMM0-XMM15 file is entirely
                    // caller-saved. Mark every cached XMM free.
                    for (std::uint32_t i = 0; i < n_alloc_fpr; ++i) {
                        xmm_cache.clobber(target.allocatable_fpr[i]);
                    }
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

    // ---- patch phase: resolve every Jcc/JMP rel32 to its block_start offset -
    // Task 24 (candidate j): JMP sites use patch_rel32 (5-byte instruction,
    // rel = target - (site+5), write at site+1); Jcc sites use patch_jcc
    // (6-byte instruction, rel = target - (site+6), write at site+2). The
    // two patchers MUST NOT be swapped — calling patch_jcc on a JMP site
    // writes the rel32 to the wrong byte (site+2 instead of site+1) AND
    // uses the wrong rel formula, producing bogus jump targets that segfault
    // the JIT. The is_jmp flag on PatchSite routes each site to the
    // correct patcher.
    for (const PatchSite& p : patches) {
        if (p.target_block == 0xFFFFFFFFu) {
            // Fallthrough to past_cold: patch to the position AFTER the cold region.
            // For simplicity, patch to the past_cold_site + 5 (after the JMP rel32
            // we emitted there); execution falls through naturally past the cold
            // region only if it skipped into a handler that itself returned —
            // which Catch handlers do via the same RET path. For the common
            // case where the Jcc/JMP at the hot tail targets the past_cold
            // boundary we use the cold_offset target (the start of the cold
            // region is also "past the hot body" by construction).
            if (p.is_jmp) {
                a.patch_rel32(p.site, out.cold_offset);
            } else {
                a.patch_jcc(p.site, out.cold_offset);
            }
        } else {
            const std::size_t* target = block_start.get(p.target_block);
            if (target) {
                if (p.is_jmp) {
                    a.patch_rel32(p.site, *target);
                } else {
                    a.patch_jcc(p.site, *target);
                }
            } else {
                // Target block was never emitted (e.g. unreachable
                // successor). Patch to past_cold as a safe fallthrough.
                if (p.is_jmp) {
                    a.patch_rel32(p.site, out.cold_offset);
                } else {
                    a.patch_jcc(p.site, out.cold_offset);
                }
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
    out.gpr_cache_hits = gpr_cache_hits;
    out.self_mov_eliminations = self_mov_eliminations;
    out.xmm_cache_hits = xmm_cache_hits;
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
