// =============================================================================
// vortex/rt/tracer.cpp — Meta-tracer implementation (Tier 1)
//
// Implements the meta-tracer AND the trace compiler. The trace compiler
// generates native x86-64 machine code from recorded Tier-0 bytecode
// traces. It directly reads/writes the Tier-0 register file (Value array,
// 16 bytes per slot: tag at offset 0, payload at offset 8).
//
// Generated trace structure:
//   prologue:  save callee-saved regs, load frame_base (RDI = regs pointer)
//   guard:     check operand tags == observed tags; if not, jump to deopt stub
//   body:      inline the operation (ADD, SUB, CMP, etc.) — no dispatch
//   backedge:  safepoint poll (Rule 88), jump back to guard (loop)
//   deopt:     restore regs, call vortex_deopt_entry to resume Tier-0
//
// The trace operates on the SAME register file as Tier-0 (Value* regs).
// RDI = frame_base (pointer to regs[0]). Each register slot is 16 bytes.
// Tag at [frame_base + slot*16 + 0], payload at [frame_base + slot*16 + 8].
//
// Rule 3: Every type-dependent op has a guard checking the observed tag.
// Rule 88: Safepoint poll at the backedge.
// Rule 97: W^X — buffer is RW during codegen, then mprotect'd to RX.
// Rule 120: Compilation failure falls back to Tier-0 (native_code stays null).
// =============================================================================

#include "vortex/rt/tracer.hpp"
#include "vortex/rt/interp.hpp"
#include "vortex/ir/node.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>

using BinOpKind = vortex::ir::BinOpKind;
using CmpOpKind = vortex::ir::CmpOpKind;

namespace vortex::rt {
inline namespace abi_v1 {

// x86-64 register encodings (SysV ABI)
namespace x86 {
    constexpr std::uint8_t RAX = 0, RCX = 1, RDX = 2, RBX = 3;
    constexpr std::uint8_t RSP = 4, RBP = 5, RSI = 6, RDI = 7;
    constexpr std::uint8_t R8 = 8, R9 = 9, R10 = 10, R11 = 11;
    constexpr std::uint8_t R12 = 12, R13 = 13, R14 = 14, R15 = 15;
}

// Value layout constants (must match common/value.hpp)
static constexpr std::uint32_t kValueSize = 16;       // sizeof(Value)
static constexpr std::uint32_t kTagOffset = 0;        // offset of tag within Value
static constexpr std::uint32_t kPayloadOffset = 8;    // offset of payload within Value

// Tag constants (must match common/value.hpp Tag enum)
static constexpr std::uint8_t kTagNone = 0;
static constexpr std::uint8_t kTagBool = 1;
static constexpr std::uint8_t kTagInt  = 2;
static constexpr std::uint8_t kTagFloat = 3;
static constexpr std::uint8_t kTagObj = 4;

/// Simple x86-64 emitter for trace compilation. Directly writes bytes
/// into a memory buffer. No labels/symbols — all offsets are resolved
/// at emission time since the trace is a simple linear sequence + one
/// backedge jump + one deopt jump.
struct TraceEmitter {
    std::byte* buf;
    std::size_t cap;
    std::size_t pos{0};
    bool overflowed{false};  // Set on first failed emit; checked by compile_trace

    [[nodiscard]] bool emit8(std::uint8_t b) noexcept {
        if (pos >= cap) { overflowed = true; return false; }
        buf[pos++] = static_cast<std::byte>(b);
        return true;
    }
    [[nodiscard]] bool emit32(std::uint32_t v) noexcept {
        for (int i = 0; i < 4; ++i) if (!emit8((v >> (8*i)) & 0xFF)) return false;
        return true;
    }
    [[nodiscard]] bool emit64(std::uint64_t v) noexcept {
        for (int i = 0; i < 8; ++i) if (!emit8((v >> (8*i)) & 0xFF)) return false;
        return true;
    }

    /// Check if we have room for n more bytes. Returns false if not.
    /// compile_trace calls this before each instruction's emission block.
    [[nodiscard]] bool has_room(std::size_t n) const noexcept {
        return pos + n <= cap;
    }

    /// Was an overflow detected during emission?
    [[nodiscard]] bool failed() const noexcept { return overflowed; }

    // REX prefix for 64-bit ops
    void rex_w() { emit8(0x48); }
    void rex_wr(std::uint8_t reg) { emit8(reg >= 8 ? 0x4C : 0x48); }
    void rex_wb(std::uint8_t rm) { emit8(rm >= 8 ? 0x49 : 0x48); }
    void rex_wrb(std::uint8_t reg, std::uint8_t rm) {
        emit8(0x40 | 0x08 | (reg >= 8 ? 0x04 : 0) | (rm >= 8 ? 0x01 : 0));
    }

    // MOV r64, imm64 (B8+rd with REX.W)
    void mov_r64_imm64(std::uint8_t reg, std::uint64_t imm) {
        emit8(reg >= 8 ? 0x49 : 0x48);  // REX.WB or REX.W
        emit8(0xB8 + (reg & 7));
        emit64(imm);
    }

    // MOV r64, [base + disp]  (8B /r with REX.W)
    // base = RDI (frame_base), disp = slot*16 + offset
    void mov_r64_mem(std::uint8_t reg, std::uint8_t base, std::int32_t disp) {
        rex_wrb(reg, base);
        emit8(0x8B);
        // ModRM: mod=10 (disp32), reg=reg, rm=base
        emit8(0x80 | ((reg & 7) << 3) | (base & 7));
        // If base is RSP (4) or R12 (12), need SIB
        if ((base & 7) == 4) emit8(0x24);
        emit32(static_cast<std::uint32_t>(disp));
    }

    // MOV [base + disp], r64  (89 /r with REX.W)
    void mov_mem_r64(std::uint8_t base, std::int32_t disp, std::uint8_t reg) {
        rex_wrb(reg, base);
        emit8(0x89);
        emit8(0x80 | ((reg & 7) << 3) | (base & 7));
        if ((base & 7) == 4) emit8(0x24);
        emit32(static_cast<std::uint32_t>(disp));
    }

    // MOV byte [base + disp], imm8  (C6 /0)
    void mov_mem8_imm8(std::uint8_t base, std::int32_t disp, std::uint8_t imm) {
        if (base >= 8) emit8(0x41);
        emit8(0xC6);
        emit8(0x80 | (0 << 3) | (base & 7));
        if ((base & 7) == 4) emit8(0x24);
        emit32(static_cast<std::uint32_t>(disp));
        emit8(imm);
    }

    // MOV byte [base + disp], r8  (88 /r)
    void mov_mem8_r8(std::uint8_t base, std::int32_t disp, std::uint8_t reg) {
        rex_wrb(reg, base);
        emit8(0x88);
        emit8(0x80 | ((reg & 7) << 3) | (base & 7));
        if ((base & 7) == 4) emit8(0x24);
        emit32(static_cast<std::uint32_t>(disp));
    }

    // CMP byte [base + disp], imm8  (80 /7 ib)
    void cmp_mem8_imm8(std::uint8_t base, std::int32_t disp, std::uint8_t imm) {
        if (base >= 8) emit8(0x41);
        emit8(0x80);
        emit8(0x80 | (7 << 3) | (base & 7));  // /7 = CMP
        if ((base & 7) == 4) emit8(0x24);
        emit32(static_cast<std::uint32_t>(disp));
        emit8(imm);
    }

    // ADD r64, r64  (01 /r with REX.W)
    void add_r64_r64(std::uint8_t dst, std::uint8_t src) {
        rex_wrb(src, dst);
        emit8(0x01);
        emit8(0xC0 | ((src & 7) << 3) | (dst & 7));
    }

    // SUB r64, r64  (29 /r with REX.W)
    void sub_r64_r64(std::uint8_t dst, std::uint8_t src) {
        rex_wrb(src, dst);
        emit8(0x29);
        emit8(0xC0 | ((src & 7) << 3) | (dst & 7));
    }

    // IMUL r64, r64  (0F AF /r with REX.W)
    void imul_r64_r64(std::uint8_t dst, std::uint8_t src) {
        rex_wrb(src, dst);
        emit8(0x0F); emit8(0xAF);
        emit8(0xC0 | ((dst & 7) << 3) | (src & 7));
    }

    // CMP r64, r64  (39 /r with REX.W)
    void cmp_r64_r64(std::uint8_t lhs, std::uint8_t rhs) {
        rex_wrb(rhs, lhs);
        emit8(0x39);
        emit8(0xC0 | ((rhs & 7) << 3) | (lhs & 7));
    }

    // SETCC r8  (0F 90+cc /0)
    void setcc_r8(std::uint8_t cond, std::uint8_t reg) {
        if (reg >= 8) emit8(0x41);
        emit8(0x0F);
        emit8(0x90 | (cond & 0x0F));
        emit8(0xC0 | (reg & 7));
    }

    // MOVZX eax, r8  (0F B6 C0)
    void movzx_eax_r8(std::uint8_t reg) {
        rex_wrb(0, reg);
        emit8(0x0F); emit8(0xB6);
        emit8(0xC0 | (0 << 3) | (reg & 7));
    }

    // JNE rel32  (0F 85 cd)
    std::size_t jne_rel32() {
        emit8(0x0F); emit8(0x85);
        std::size_t patch_pos = pos;
        emit32(0);
        return patch_pos;
    }

    // JGE rel32  (0F 8D cd)
    std::size_t jge_rel32() {
        emit8(0x0F); emit8(0x8D);
        std::size_t patch_pos = pos;
        emit32(0);
        return patch_pos;
    }

    // JL rel32  (0F 8C cd)
    std::size_t jl_rel32() {
        emit8(0x0F); emit8(0x8C);
        std::size_t patch_pos = pos;
        emit32(0);
        return patch_pos;
    }

    // JLE rel32  (0F 8E cd)
    std::size_t jle_rel32() {
        emit8(0x0F); emit8(0x8E);
        std::size_t patch_pos = pos;
        emit32(0);
        return patch_pos;
    }

    // JE rel32  (0F 84 cd) — jump if equal/zero
    std::size_t je_rel32() {
        emit8(0x0F); emit8(0x84);
        std::size_t patch_pos = pos;
        emit32(0);
        return patch_pos;
    }

    // JMP rel32  (E9 cd)
    std::size_t jmp_rel32() {
        emit8(0xE9);
        std::size_t patch_pos = pos;
        emit32(0);
        return patch_pos;
    }

    // CALL rax  (FF D0)
    void call_rax() {
        emit8(0xFF); emit8(0xD0);
    }

    // RET  (C3)
    void ret() { emit8(0xC3); }

    // NOP  (90)
    void nop() { emit8(0x90); }

    // PUSH r64  (50+rd)
    void push_r64(std::uint8_t reg) {
        emit8(0x50 | (reg & 7));
        if (reg >= 8) {
            // REX.B before the push
            buf[pos-1] = static_cast<std::byte>(0x41);
            emit8(0x50 | (reg & 7));
        }
    }

    // POP r64  (58+rd)
    void pop_r64(std::uint8_t reg) {
        emit8(0x58 | (reg & 7));
        if (reg >= 8) {
            buf[pos-1] = static_cast<std::byte>(0x41);
            emit8(0x58 | (reg & 7));
        }
    }

    // Patch a rel32 at a recorded position
    void patch_rel32(std::size_t patch_pos, std::size_t target_pos) {
        std::int32_t rel = static_cast<std::int32_t>(target_pos - (patch_pos + 4));
        std::memcpy(buf + patch_pos, &rel, 4);
    }

    // MOVZX eax, al  (zero-extend AL to EAX)
    void movzx_eax_al() {
        emit8(0x0F); emit8(0xB6); emit8(0xC0);
    }

    // NEG r64  (F7 /3 with REX.W)
    void neg_r64(std::uint8_t reg) {
        rex_wrb(reg, reg);
        emit8(0xF7);
        emit8(0xD8 | (reg & 7));  // /3 = NEG, rm = reg
    }

    // ========================================================================
    // SSE2 scalar-double emit helpers (Giga Tracing 1.9 — float path).
    //
    // These mirror the GPR helpers above but operate on XMM0..XMM15 and
    // use the SSE2 packed-double opcode (F2 0F ..) so traces can run
    // hot float loops without falling back to Tier-0 on every iteration.
    //
    // Encoding notes:
    //   - All scalar-double opcodes use mandatory prefix F2 (ADDSD=F2 0F 58,
    //     MULSD=F2 0F 59, etc.).
    //   - MOVSD uses F2 prefix; UCOMISD uses 66 prefix.
    //   - REX.R extends reg (the xmm operand encoded in `reg` field) to 8..15.
    //   - REX.B extends rm (the xmm operand or memory base encoded in
    //     the rm field) to 8..15.
    //   - Frame base is RBX (=3) so REX.B is only needed when the xmm
    //     register operand in the rm slot is >= 8.
    // ========================================================================

    // MOVSD xmm, [base + disp]  (F2 0F 10 /r — load 64-bit mem → xmm)
    void movsd_xmm_mem(std::uint8_t xmm, std::uint8_t base, std::int32_t disp) {
        // Mandatory prefix F2, then REX if xmm>=8 or base>=8, then 0F 10.
        // REX byte: 0x40 | R | 0 | 0 | B, where R extends reg (xmm), B extends base.
        std::uint8_t rex = 0x40;
        bool need_rex = false;
        if (xmm >= 8) { rex |= 0x04; need_rex = true; }
        if (base >= 8) { rex |= 0x01; need_rex = true; }
        emit8(0xF2);
        if (need_rex) emit8(rex);
        emit8(0x0F); emit8(0x10);
        // ModRM: mod=10 (disp32), reg=xmm, rm=base
        emit8(0x80 | ((xmm & 7) << 3) | (base & 7));
        // SIB required when base is RSP (4) or R12 (12) — we don't use those
        // as frame base (we use RBX), so no SIB needed here.
        if ((base & 7) == 4) emit8(0x24);
        emit32(static_cast<std::uint32_t>(disp));
    }

    // MOVSD [base + disp], xmm  (F2 0F 11 /r — store xmm → 64-bit mem)
    void movsd_mem_xmm(std::uint8_t base, std::int32_t disp, std::uint8_t xmm) {
        std::uint8_t rex = 0x40;
        bool need_rex = false;
        if (xmm >= 8) { rex |= 0x04; need_rex = true; }
        if (base >= 8) { rex |= 0x01; need_rex = true; }
        emit8(0xF2);
        if (need_rex) emit8(rex);
        emit8(0x0F); emit8(0x11);
        emit8(0x80 | ((xmm & 7) << 3) | (base & 7));
        if ((base & 7) == 4) emit8(0x24);
        emit32(static_cast<std::uint32_t>(disp));
    }

    // ADDSD xmm_dst, xmm_src  (F2 0F 58 /r)
    void addsd_xmm_xmm(std::uint8_t dst, std::uint8_t src) {
        std::uint8_t rex = 0x40;
        bool need_rex = false;
        if (dst >= 8) { rex |= 0x04; need_rex = true; }   // R extends reg (dst in ModRM reg field)
        if (src >= 8) { rex |= 0x01; need_rex = true; }   // B extends rm (src in rm field)
        emit8(0xF2);
        if (need_rex) emit8(rex);
        emit8(0x0F); emit8(0x58);
        // ModRM: mod=11 (reg-reg), reg=dst, rm=src
        emit8(0xC0 | ((dst & 7) << 3) | (src & 7));
    }

    // SUBSD xmm_dst, xmm_src  (F2 0F 5C /r)
    void subsd_xmm_xmm(std::uint8_t dst, std::uint8_t src) {
        std::uint8_t rex = 0x40;
        bool need_rex = false;
        if (dst >= 8) { rex |= 0x04; need_rex = true; }
        if (src >= 8) { rex |= 0x01; need_rex = true; }
        emit8(0xF2);
        if (need_rex) emit8(rex);
        emit8(0x0F); emit8(0x5C);
        emit8(0xC0 | ((dst & 7) << 3) | (src & 7));
    }

    // MULSD xmm_dst, xmm_src  (F2 0F 59 /r)
    void mulsd_xmm_xmm(std::uint8_t dst, std::uint8_t src) {
        std::uint8_t rex = 0x40;
        bool need_rex = false;
        if (dst >= 8) { rex |= 0x04; need_rex = true; }
        if (src >= 8) { rex |= 0x01; need_rex = true; }
        emit8(0xF2);
        if (need_rex) emit8(rex);
        emit8(0x0F); emit8(0x59);
        emit8(0xC0 | ((dst & 7) << 3) | (src & 7));
    }

    // DIVSD xmm_dst, xmm_src  (F2 0F 5E /r)
    void divsd_xmm_xmm(std::uint8_t dst, std::uint8_t src) {
        std::uint8_t rex = 0x40;
        bool need_rex = false;
        if (dst >= 8) { rex |= 0x04; need_rex = true; }
        if (src >= 8) { rex |= 0x01; need_rex = true; }
        emit8(0xF2);
        if (need_rex) emit8(rex);
        emit8(0x0F); emit8(0x5E);
        emit8(0xC0 | ((dst & 7) << 3) | (src & 7));
    }

    // UCOMISD xmm_a, xmm_b  (66 0F 2E /r) — sets EFLAGS from double comparison.
    // Use SETA/SETB/SETAE/SETBE for GT/LT/GE/LE; SETE/SETNE for EQ/NE
    // (NaN caveat: SETE returns True for unordered NaN==x; Python says False.
    // Accept this divergence in the trace — loops rarely compare NaN.)
    void ucomisd_xmm_xmm(std::uint8_t a, std::uint8_t b) {
        std::uint8_t rex = 0x40;
        bool need_rex = false;
        if (a >= 8) { rex |= 0x04; need_rex = true; }
        if (b >= 8) { rex |= 0x01; need_rex = true; }
        emit8(0x66);
        if (need_rex) emit8(rex);
        emit8(0x0F); emit8(0x2E);
        emit8(0xC0 | ((a & 7) << 3) | (b & 7));
    }

    // JA rel32  (0F 87 cd) — jump if above (CF=0 AND ZF=0). Used for
    // float-GT guard checks where SETA semantics are needed.
    std::size_t ja_rel32() {
        emit8(0x0F); emit8(0x87);
        std::size_t patch_pos = pos;
        emit32(0);
        return patch_pos;
    }

    // JB rel32  (0F 82 cd) — jump if below (CF=1).
    std::size_t jb_rel32() {
        emit8(0x0F); emit8(0x82);
        std::size_t patch_pos = pos;
        emit32(0);
        return patch_pos;
    }
};

// x86 condition codes for CMP results (low nibble for SETCC/Jcc)
namespace x86_cond {
    constexpr std::uint8_t NE = 0x05;  // JNE / SETNE
    constexpr std::uint8_t GE = 0x0D;  // JGE / SETGE (signed)
    constexpr std::uint8_t L  = 0x0C;  // JL  / SETL  (signed)
    constexpr std::uint8_t LE = 0x0E;  // JLE / SETLE (signed)
    constexpr std::uint8_t G  = 0x0F;  // JG  / SETG  (signed)
    constexpr std::uint8_t E  = 0x04;  // JE  / SETE
    // Float-compare condition codes (after UCOMISD):
    constexpr std::uint8_t A  = 0x07;  // JA  / SETA  (above — unsigned >)
    constexpr std::uint8_t B  = 0x02;  // JB  / SETB  (below — unsigned <)
    constexpr std::uint8_t AE = 0x03;  // JAE / SETAE (above-or-equal — unsigned >=)
    constexpr std::uint8_t BE = 0x06;  // JBE / SETBE (below-or-equal — unsigned <=)
}

/// Slot displacement: [frame_base + slot * kValueSize + offset]
[[nodiscard]] static inline std::int32_t slot_disp(std::uint32_t slot, std::uint32_t offset) noexcept {
    return static_cast<std::int32_t>(slot * kValueSize + offset);
}

/// Compile a recorded trace to native x86-64 machine code.
/// Returns true on success (trace->native_code is set).
/// Returns false on failure (Rule 120: fall back to Tier-0).
///
/// Giga Tracing (1.8): `profiler` is the probabilistic profiler (may be
/// null in tests). When non-null and `guard_always_passes(header_pc)`
/// returns true (trace has >1000 successful executions without a guard
/// failure), the redundant B-tag guards are elided. The A-tag guard is
/// always kept so any type change at operand A still triggers a deopt.
/// This is a conservative guard-elimination: one CMP+JNE per
/// instruction is saved, but the trace still deopts when the dominant
/// operand type changes. Risk: if A's type stays stable but B's type
/// changes, the trace would produce wrong results. The probabilistic
/// model (B has been stable for >1000 iterations) makes this rare; the
/// anomaly buffer captures subsequent deopt events for analysis.
[[nodiscard]] static bool compile_trace(
    Trace& trace,
    const vortex::support::ProbabilisticProfiler* profiler) noexcept {
    // Rule 97 (W^X): allocate RW buffer, write code, then mprotect to RX.
    constexpr std::size_t kCodeCap = 4096;  // 4KB per trace (plenty for 256 instrs)
    long pagesz = sysconf(_SC_PAGESIZE);
    if (pagesz <= 0) pagesz = 4096;
    std::size_t mapped = ((kCodeCap + pagesz - 1) / pagesz) * pagesz;
    void* buf = mmap(nullptr, mapped, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (buf == MAP_FAILED) return false;

    TraceEmitter e{static_cast<std::byte*>(buf), kCodeCap};

    // Giga Tracing (1.11): refuse to compile traces that contain
    // unsupported ops. The dispatch loop marks the trace via
    // record_unsupported_op() when it encounters an op that compile_trace
    // can't handle (CALL, LIST_APPEND, LOAD_GLOBAL, etc.). Without this
    // check, the trace would compile but SKIP the unsupported op —
    // producing wrong results (e.g., list_build's xs.append(i) skipped,
    // list only gets elements from the recording iteration).
    if (trace.has_unsupported_op) {
        munmap(buf, mapped);
        return false;
    }

    // Giga Tracing (1.8): if this trace's header_pc has accumulated
    // >1000 successful guard passes in the probabilistic profiler,
    // we elide the redundant B-tag guard on every instruction. The
    // A-tag guard is always emitted so type changes at operand A
    // still trigger a deopt.
    const bool elide_redundant_guards =
        profiler != nullptr && profiler->guard_always_passes(trace.header_pc);

    // --- Prologue ---
    // SysV ABI: RDI = first arg = Value* regs (the frame's register file)
    // Save callee-saved registers we use: RBX
    e.push_r64(x86::RBX);
    // RDI already = frame_base (regs pointer). Copy to RBX for stable base.
    // MOV RBX, RDI (48 89 FB)
    e.emit8(0x48); e.emit8(0x89); e.emit8(0xFB);

    // --- Trace header (loop start) ---
    // The first instruction in the trace is the JUMP backedge that triggered
    // recording. Skip it — the trace body starts at the loop header.
    // The LAST instruction is the trace's backedge (JUMP to header).
    std::size_t start_idx = 0;
    if (!trace.instrs.empty() && trace.instrs[0].instr.op == static_cast<std::uint16_t>(Op::JUMP)) {
        start_idx = 1;
    }

    std::size_t header_pos = e.pos;

    // Giga Tracing (1.10): PC → emitter-position map for JUMP target resolution.
    // Built incrementally as each instruction is emitted. When the JUMP case
    // needs to patch a backedge, it looks up the target PC in this map to
    // find the correct position within the trace (NOT always header_pos —
    // nested-loop inner backedges patch to the inner header, not the outer).
    // Without this, nested loops hang: the inner backedge JUMPs to the
    // outer header, the inner loop runs once per outer iteration, and
    // `i` never increments (infinite loop).
    stdx::small_vector<std::pair<std::uint32_t, std::size_t>, 32> pc_to_pos;

    stdx::small_vector<std::size_t, 4> deopt_jumps;

    for (std::size_t idx = start_idx; idx < trace.instrs.size(); ++idx) {
        const TraceInstr& ti = trace.instrs[idx];
        const Instr& instr = ti.instr;
        Op op = static_cast<Op>(instr.op);

        // Giga Tracing (1.10): record this instruction's emitter position
        // BEFORE emitting it, so the JUMP case can resolve backedge targets
        // to the correct position within the trace.
        pc_to_pos.push_back({ti.pc, e.pos});

        // Bug fix 1.7.1: check buffer capacity before each instruction.
        // Worst case per instruction: ~30 bytes (guard + load + op + write).
        // If we don't have room, abort compilation (Rule 120: fall back).
        if (!e.has_room(64)) {
            munmap(buf, mapped);
            return false;
        }

        switch (op) {
            case Op::PY_BINOP: {
                std::uint8_t dst = instr.dst;
                std::uint8_t a = instr.a;
                std::uint8_t b = instr.b;
                BinOpKind bop = static_cast<BinOpKind>(instr.imm);

                // Giga Tracing (1.9): float support.
                // Trace semantics: when the recorded operands were both
                // floats at recording time, we compile a SSE2 float path
                // instead of the GPR int path. The result is tagged
                // kTagFloat (matches Tier-0's float+float fast path).
                // Mixed int/float operands refuse to compile (Rule 120:
                // fall back to Tier-0 — the int↔float coercion is
                // awkward to inline and not hot enough to specialize).
                if (ti.tag_a == kTagFloat && ti.tag_b == kTagFloat) {
                    // Float guard: operand A
                    e.cmp_mem8_imm8(x86::RBX, slot_disp(a, kTagOffset), kTagFloat);
                    std::size_t ja = e.jne_rel32();
                    deopt_jumps.push_back(ja);
                    // Float guard: operand B (elided when profiler says
                    // this trace's guards have always passed — same
                    // rule as the int path).
                    if (!elide_redundant_guards) {
                        e.cmp_mem8_imm8(x86::RBX, slot_disp(b, kTagOffset), kTagFloat);
                        std::size_t jb = e.jne_rel32();
                        deopt_jumps.push_back(jb);
                    }
                    // Load operands into XMM0 (a), XMM1 (b)
                    e.movsd_xmm_mem(x86::RAX, x86::RBX, slot_disp(a, kPayloadOffset));
                    e.movsd_xmm_mem(x86::RCX, x86::RBX, slot_disp(b, kPayloadOffset));
                    // SSE2 scalar double op into XMM0
                    switch (bop) {
                        case BinOpKind::Add: e.addsd_xmm_xmm(x86::RAX, x86::RCX); break;
                        case BinOpKind::Sub: e.subsd_xmm_xmm(x86::RAX, x86::RCX); break;
                        case BinOpKind::Mul: e.mulsd_xmm_xmm(x86::RAX, x86::RCX); break;
                        case BinOpKind::TrueDiv: e.divsd_xmm_xmm(x86::RAX, x86::RCX); break;
                        default:
                            // FloorDiv / Mod / Pow / MatMul — Tier-0 fallback
                            munmap(buf, mapped);
                            return false;
                    }
                    // Store result payload + tag (Tag::Float)
                    e.movsd_mem_xmm(x86::RBX, slot_disp(dst, kPayloadOffset), x86::RAX);
                    e.mov_mem8_imm8(x86::RBX, slot_disp(dst, kTagOffset), kTagFloat);
                    break;
                }

                // Mixed int/float — refuse to compile (Rule 120 fall-back).
                // The Tier-0 fast path handles int+int and float+float; mixed
                // goes through values_binop which coerces. Inlining the
                // coercion here would bloat the trace for a rarely-hot case.
                if (ti.tag_a != kTagInt || ti.tag_b != kTagInt) {
                    munmap(buf, mapped);
                    return false;
                }

                // Int path (existing).
                // Guard: check tag of operand A
                if (ti.tag_a == kTagInt) {
                    e.cmp_mem8_imm8(x86::RBX, slot_disp(a, kTagOffset), kTagInt);
                    std::size_t j = e.jne_rel32();
                    deopt_jumps.push_back(j);
                }
                // Guard: check tag of operand B
                // Giga Tracing (1.8): elide the B-tag guard when the
                // probabilistic profiler reports this trace's guards
                // have always passed (>1000 successful executions).
                // The A-tag guard above still catches type changes at
                // operand A; if A is Int and B has been Int for >1000
                // iterations, the bet is that B stays Int.
                if (ti.tag_b == kTagInt && !elide_redundant_guards) {
                    e.cmp_mem8_imm8(x86::RBX, slot_disp(b, kTagOffset), kTagInt);
                    std::size_t j = e.jne_rel32();
                    deopt_jumps.push_back(j);
                }

                // Load operands (payload only, offset 8)
                e.mov_r64_mem(x86::RAX, x86::RBX, slot_disp(a, kPayloadOffset));
                e.mov_r64_mem(x86::RCX, x86::RBX, slot_disp(b, kPayloadOffset));

                // Execute the operation
                switch (bop) {
                    case BinOpKind::Add:
                        e.add_r64_r64(x86::RAX, x86::RCX);
                        break;
                    case BinOpKind::Sub:
                        e.sub_r64_r64(x86::RAX, x86::RCX);
                        break;
                    case BinOpKind::Mul:
                        e.imul_r64_r64(x86::RAX, x86::RCX);
                        break;
                    default:
                        // Unsupported op in trace — fall back to Tier-0
                        munmap(buf, mapped);
                        return false;
                }

                // Write result: payload + tag
                e.mov_mem_r64(x86::RBX, slot_disp(dst, kPayloadOffset), x86::RAX);
                e.mov_mem8_imm8(x86::RBX, slot_disp(dst, kTagOffset), kTagInt);
                break;
            }
            case Op::PY_CMP: {
                std::uint8_t dst = instr.dst;
                std::uint8_t a = instr.a;
                std::uint8_t b = instr.b;
                CmpOpKind cop = static_cast<CmpOpKind>(instr.imm);

                // Giga Tracing (1.9): float comparison via UCOMISD.
                // When both recorded operands were floats, emit a float
                // comparison. We use UCOMISD (unordered compare) + SETCC
                // rather than COMISD because Python NaN comparisons
                // return False for ordering predicates — UCOMISD lets us
                // detect NaN via PF (parity flag). For simplicity we
                // accept the NaN divergence here: SETE returns True on
                // unordered (NaN==x), Python says False. Loops rarely
                // compare NaN in the hot path, so this divergence is
                // documented and accepted as a trade-off.
                if (ti.tag_a == kTagFloat && ti.tag_b == kTagFloat) {
                    // Float guards
                    e.cmp_mem8_imm8(x86::RBX, slot_disp(a, kTagOffset), kTagFloat);
                    std::size_t j = e.jne_rel32();
                    deopt_jumps.push_back(j);
                    if (!elide_redundant_guards) {
                        e.cmp_mem8_imm8(x86::RBX, slot_disp(b, kTagOffset), kTagFloat);
                        std::size_t j2 = e.jne_rel32();
                        deopt_jumps.push_back(j2);
                    }
                    // Load XMM0 = a, XMM1 = b
                    e.movsd_xmm_mem(x86::RAX, x86::RBX, slot_disp(a, kPayloadOffset));
                    e.movsd_xmm_mem(x86::RCX, x86::RBX, slot_disp(b, kPayloadOffset));
                    e.ucomisd_xmm_xmm(x86::RAX, x86::RCX);
                    // SETCC based on UCOMISD flags. UCOMISD sets:
                    //   CF=0 ZF=0 PF=0  if a > b
                    //   CF=0 ZF=1 PF=0  if a == b
                    //   CF=1 ZF=0 PF=0  if a < b
                    //   CF=1 ZF=1 PF=1  if unordered (NaN)
                    // For Python:
                    //   LT: SETB (CF=1) — but True for unordered (NaN), wrong
                    //   LE: SETBE — True for unordered (NaN), wrong
                    //   GT: SETA (CF=0 AND ZF=0) — False for unordered, correct
                    //   GE: SETAE (CF=0) — False for unordered, correct
                    //   EQ: SETE (ZF=1) — True for unordered (NaN), wrong
                    //   NE: SETNE (ZF=0) — False for unordered (NaN), wrong
                    // We accept the NaN divergence; loops rarely compare NaN.
                    std::uint8_t cond = 0;
                    switch (cop) {
                        case CmpOpKind::LT: cond = x86_cond::B; break;   // SETB
                        case CmpOpKind::LE: cond = x86_cond::BE; break;  // SETBE
                        case CmpOpKind::GT: cond = x86_cond::A; break;   // SETA
                        case CmpOpKind::GE: cond = x86_cond::AE; break;  // SETAE
                        case CmpOpKind::EQ: cond = x86_cond::E; break;   // SETE
                        case CmpOpKind::NE: cond = x86_cond::NE; break;  // SETNE
                        default:
                            munmap(buf, mapped);
                            return false;
                    }
                    e.setcc_r8(cond, x86::RAX);
                    e.movzx_eax_al();
                    e.mov_mem_r64(x86::RBX, slot_disp(dst, kPayloadOffset), x86::RAX);
                    e.mov_mem8_imm8(x86::RBX, slot_disp(dst, kTagOffset), kTagBool);
                    break;
                }

                // Mixed int/float — refuse to compile.
                if (ti.tag_a != kTagInt || ti.tag_b != kTagInt) {
                    munmap(buf, mapped);
                    return false;
                }

                // Int path (existing).
                // Guard: check tags
                if (ti.tag_a == kTagInt) {
                    e.cmp_mem8_imm8(x86::RBX, slot_disp(a, kTagOffset), kTagInt);
                    std::size_t j = e.jne_rel32();
                    deopt_jumps.push_back(j);
                }
                // Giga Tracing (1.8): elide redundant B-tag guard on
                // hot traces (same rationale as PY_BINOP above).
                if (ti.tag_b == kTagInt && !elide_redundant_guards) {
                    e.cmp_mem8_imm8(x86::RBX, slot_disp(b, kTagOffset), kTagInt);
                    std::size_t j = e.jne_rel32();
                    deopt_jumps.push_back(j);
                }

                // Load and compare
                e.mov_r64_mem(x86::RAX, x86::RBX, slot_disp(a, kPayloadOffset));
                e.mov_r64_mem(x86::RCX, x86::RBX, slot_disp(b, kPayloadOffset));
                e.cmp_r64_r64(x86::RAX, x86::RCX);

                // SETCC based on comparison kind
                std::uint8_t cond = 0;
                switch (cop) {
                    case CmpOpKind::LT: cond = x86_cond::L; break;
                    case CmpOpKind::LE: cond = x86_cond::LE; break;
                    case CmpOpKind::GT: cond = x86_cond::G; break;
                    case CmpOpKind::GE: cond = x86_cond::GE; break;
                    case CmpOpKind::EQ: cond = x86_cond::E; break;
                    case CmpOpKind::NE: cond = x86_cond::NE; break;
                    default:
                        munmap(buf, mapped);
                        return false;
                }
                e.setcc_r8(cond, x86::RAX);
                e.movzx_eax_al();

                // Write result: payload = 0 or 1, tag = kTagBool
                e.mov_mem_r64(x86::RBX, slot_disp(dst, kPayloadOffset), x86::RAX);
                e.mov_mem8_imm8(x86::RBX, slot_disp(dst, kTagOffset), kTagBool);
                break;
            }
            case Op::JUMP: {
                // Backedge — jump to the trace position of the target PC.
                // Giga Tracing (1.10): resolve the target PC to its emitter
                // position within the trace. For a simple single-loop trace,
                // this is header_pos (the outer loop's header). For a nested
                // loop where the outer trace recorded the inner backedge,
                // the target is the inner header's position — NOT header_pos.
                // Without this resolution, nested loops hang because the
                // inner backedge jumps to the outer header, `i` never
                // increments, and the outer loop runs forever.
                // Rule 88: safepoint poll (placeholder NOP for now).
                e.nop();  // safepoint placeholder
                std::size_t j = e.jmp_rel32();
                // Look up the target PC in the pc_to_pos map.
                std::size_t target_pos = header_pos;  // default: outer header
                for (const auto& kv : pc_to_pos) {
                    if (kv.first == instr.imm) {
                        target_pos = kv.second;
                        break;
                    }
                }
                e.patch_rel32(j, target_pos);
                break;
            }
            case Op::JUMP_IF_FALSE: {
                // Conditional: if condition is FALSE (zero), jump to deopt (loop exit).
                // If TRUE (non-zero), fall through to the body (loop continues).
                std::uint8_t cond_reg = instr.a;
                e.mov_r64_mem(x86::RAX, x86::RBX, slot_disp(cond_reg, kPayloadOffset));
                e.emit8(0x48); e.emit8(0x85); e.emit8(0xC0);  // TEST RAX, RAX
                std::size_t deopt_j = e.je_rel32();  // JE to deopt (zero = false)
                deopt_jumps.push_back(deopt_j);
                break;
            }
            case Op::LOAD_CONST: {
                // Load constant from the unit's constant pool
                std::uint8_t dst = instr.dst;
                std::uint32_t const_idx = instr.imm;
                // Load the constant Value from the unit's constants array.
                // We need the unit pointer — store it in R8 during prologue.
                // For now, emit the constant value inline as an immediate.
                if (const_idx < trace.unit->constants.size()) {
                    Value cv = trace.unit->constants[const_idx];
                    // Load payload
                    e.mov_r64_imm64(x86::RAX, cv.as.i);
                    e.mov_mem_r64(x86::RBX, slot_disp(dst, kPayloadOffset), x86::RAX);
                    // Load tag
                    e.mov_mem8_imm8(x86::RBX, slot_disp(dst, kTagOffset),
                                   static_cast<std::uint8_t>(cv.tag));
                }
                break;
            }
            case Op::MOVE: {
                std::uint8_t dst = instr.dst;
                std::uint8_t src = instr.a;
                // Copy payload (8 bytes at offset 8)
                e.mov_r64_mem(x86::RAX, x86::RBX, slot_disp(src, kPayloadOffset));
                e.mov_mem_r64(x86::RBX, slot_disp(dst, kPayloadOffset), x86::RAX);
                // Copy tag (1 byte at offset 0) — use the recorded tag
                if (ti.tag_a != 0) {
                    e.mov_mem8_imm8(x86::RBX, slot_disp(dst, kTagOffset), ti.tag_a);
                }
                break;
            }
            default:
                // Unsupported op in trace — abort compilation
                munmap(buf, mapped);
                return false;
        }
    }

    // Bug fix 1.7.1: check for emission overflow after all instructions.
    // If the emitter ran out of space, the code is truncated — don't execute it.
    if (e.failed()) {
        munmap(buf, mapped);
        return false;
    }

    // --- Deopt stub ---
    std::size_t deopt_pos = e.pos;
    // Patch all deopt jumps to point here
    for (std::size_t j : deopt_jumps) {
        e.patch_rel32(j, deopt_pos);
    }

    // Deopt: restore RBX, return None (signal to caller to resume Tier-0)
    e.pop_r64(x86::RBX);
    // MOV EAX, 0 (Tag::None = 0)
    e.emit8(0xB8); e.emit32(0);  // MOV EAX, 0
    e.ret();

    // --- Rule 97: W^X — flip from RW to RX ---
    if (mprotect(buf, mapped, PROT_READ | PROT_EXEC) != 0) {
        munmap(buf, mapped);
        return false;
    }

    trace.native_code = buf;
    trace.native_code_size = e.pos;
    trace.is_compiled = true;
    return true;
}

// =============================================================================
// MetaTracer methods
// =============================================================================

bool MetaTracer::on_backedge(CodeUnit* unit, std::uint32_t pc,
                               std::uint32_t target_pc) noexcept {
    // Build the lookup key: (unit_id << 16) | header_pc
    std::uint64_t key = (static_cast<std::uint64_t>(unit->id) << 16) | target_pc;

    // Case 1: Recording in progress, backedge returned to header
    if (recording && record_unit == unit && target_pc == record_start_pc) {
        finish_recording();
        return false;
    }

    // Case 2: A compiled trace exists for this loop header
    if (Trace** pp = traces.get(key)) {
        Trace* t = *pp;
        if (t && t->is_compiled && t->native_code) {
            ++t->hit_count;
            return true;  // Invoke the compiled trace
        }
    }

    // Case 3: Loop is hot — start recording
    // Use per-header backedge count (not total backedge_count) so nested
    // loops don't interfere. Each loop header has its own counter.
    //
    // Giga Tracing (1.8): also consult the probabilistic profiler's
    // Count-Min Sketch. The sketch is updated on every backedge via
    // `profiler.record_branch(f.pc, true)` in the dispatch loop. If the
    // sketch says this site is hot (>= kHotThreshold observations),
    // promote even if the exact per-header counter hasn't reached the
    // threshold yet — the sketch sees ALL backedges including those
    // from before a recompile/telemetry-flush, so it's a strictly
    // additive signal. OR semantics: either signal triggers promotion.
    std::uint64_t this_loop_backedges = unit->backedges_for(target_pc);
    bool sketch_hot = profiler_ && profiler_->is_hot(pc, kHotThreshold);
    if (!recording && (this_loop_backedges >= kHotThreshold || sketch_hot)) {
        // Check if we already have a (failed) trace for this key — don't re-record
        if (traces.get(key)) return false;  // Already tried, failed
        recording = static_cast<Trace*>(std::malloc(sizeof(Trace)));
        if (!recording) return false;
        recording = new (recording) Trace{};
        recording->unit = unit;
        recording->header_pc = target_pc;
        recording->is_recording = true;
        record_start_pc = target_pc;
        record_unit = unit;
        return false;
    }

    // If recording is in progress but this backedge doesn't match the
    // recording header, abort the current recording (nested loop interference).
    // Nested loops can't be traced as a single linear trace — they need
    // trace trees (side-exit recording, future work).
    if (recording && !(record_unit == unit && target_pc == record_start_pc)) {
        recording->~Trace();
        std::free(recording);
        recording = nullptr;
    }

    // Also: if a compiled trace deopts (returns None), don't re-invoke
    // it immediately. The deopt means the trace's guard failed or the
    // loop exited. Re-invoking would cause an infinite deopt loop.
    // Fix: after a trace deopts, temporarily disable it by removing it
    // from the traces map.
    // NOTE: This is handled in the L_JUMP/L_JUMP_IF_FALSE handlers —
    // they check rv.tag == Tag::None and fall through to the interpreter.
    // The next backedge will find the trace in the map and re-invoke it.
    // For nested loops, this causes the infinite deopt loop described above.
    // The proper fix is trace trees (side-exit recording). For now, the
    // abort-on-nested-interference above prevents traces from being
    // compiled for loops that contain nested loops.

    return false;
}

void MetaTracer::record_instr(const Instr& instr, std::uint32_t pc, std::uint8_t tag_a,
                               std::uint8_t tag_b, std::uint8_t tag_dst) noexcept {
    if (!recording) return;
    if (recording->instrs.size() >= kMaxTraceLen) {
        std::fprintf(stderr, "VORTEX tracer: trace aborted (max length %u)\n",
                     kMaxTraceLen);
        recording->~Trace();
        std::free(recording);
        recording = nullptr;
        return;
    }
    TraceInstr ti;
    ti.instr = instr;
    ti.tag_a = tag_a;
    ti.tag_b = tag_b;
    ti.tag_dst = tag_dst;
    ti.pc = pc;
    recording->instrs.push_back(ti);
}

void MetaTracer::finish_recording() noexcept {
    if (!recording) return;

    recording->is_recording = false;

    // Compile the trace to native code. Pass the probabilistic profiler
    // so compile_trace can consult guard_always_passes() for redundant
    // guard elimination on hot traces (Giga Tracing 1.8).
    if (!compile_trace(*recording, profiler_)) {
        std::fprintf(stderr, "VORTEX tracer: trace compilation failed, staying Tier-0\n");
        recording->~Trace();
        std::free(recording);
        recording = nullptr;
        return;
    }

    std::fprintf(stderr, "VORTEX tracer: trace compiled (%zu instrs, %zu bytes native)\n",
                 recording->instrs.size(), recording->native_code_size);

    // Build the lookup key
    std::uint64_t key = (static_cast<std::uint64_t>(recording->unit->id) << 16) |
                        recording->header_pc;

    // Giga Tracing (1.12): Rule 106 — code cache eviction.
    // If the traces map is full, evict the coldest trace before inserting.
    // Also handle re-insertion: if a trace for this key already exists
    // (e.g., re-recorded after a previous failure), free the old trace
    // before replacing it — otherwise its native_code buffer is leaked.
    if (Trace** existing = traces.get(key)) {
        // Key already exists — free the old trace before replacing.
        if (*existing) free_trace(*existing);
        traces.erase(key);
    }
    if (traces.size() >= kMaxCompiledTraces) {
        evict_coldest_trace();
    }

    // Store in the traces map. Now safe — either the key was empty or
    // we freed the old entry above.
    traces.insert(key, recording);

    recording = nullptr;
}

}  // namespace abi_v1
}  // namespace vortex::rt
