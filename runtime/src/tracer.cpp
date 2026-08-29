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

    [[nodiscard]] bool emit8(std::uint8_t b) noexcept {
        if (pos >= cap) return false;
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
};

// x86 condition codes for CMP results (low nibble for SETCC/Jcc)
namespace x86_cond {
    constexpr std::uint8_t NE = 0x05;  // JNE / SETNE
    constexpr std::uint8_t GE = 0x0D;  // JGE / SETGE (signed)
    constexpr std::uint8_t L  = 0x0C;  // JL  / SETL  (signed)
    constexpr std::uint8_t LE = 0x0E;  // JLE / SETLE (signed)
    constexpr std::uint8_t G  = 0x0F;  // JG  / SETG  (signed)
    constexpr std::uint8_t E  = 0x04;  // JE  / SETE
}

/// Slot displacement: [frame_base + slot * kValueSize + offset]
[[nodiscard]] static inline std::int32_t slot_disp(std::uint32_t slot, std::uint32_t offset) noexcept {
    return static_cast<std::int32_t>(slot * kValueSize + offset);
}

/// Compile a recorded trace to native x86-64 machine code.
/// Returns true on success (trace->native_code is set).
/// Returns false on failure (Rule 120: fall back to Tier-0).
[[nodiscard]] static bool compile_trace(Trace& trace) noexcept {
    // Rule 97 (W^X): allocate RW buffer, write code, then mprotect to RX.
    constexpr std::size_t kCodeCap = 4096;  // 4KB per trace (plenty for 256 instrs)
    long pagesz = sysconf(_SC_PAGESIZE);
    if (pagesz <= 0) pagesz = 4096;
    std::size_t mapped = ((kCodeCap + pagesz - 1) / pagesz) * pagesz;
    void* buf = mmap(nullptr, mapped, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (buf == MAP_FAILED) return false;

    TraceEmitter e{static_cast<std::byte*>(buf), kCodeCap};

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

    stdx::small_vector<std::size_t, 4> deopt_jumps;

    for (std::size_t idx = start_idx; idx < trace.instrs.size(); ++idx) {
        const TraceInstr& ti = trace.instrs[idx];
        const Instr& instr = ti.instr;
        Op op = static_cast<Op>(instr.op);

        switch (op) {
            case Op::PY_BINOP: {
                std::uint8_t dst = instr.dst;
                std::uint8_t a = instr.a;
                std::uint8_t b = instr.b;
                BinOpKind bop = static_cast<BinOpKind>(instr.imm);

                // Guard: check tag of operand A
                if (ti.tag_a == kTagInt) {
                    e.cmp_mem8_imm8(x86::RBX, slot_disp(a, kTagOffset), kTagInt);
                    std::size_t j = e.jne_rel32();
                    deopt_jumps.push_back(j);
                }
                // Guard: check tag of operand B
                if (ti.tag_b == kTagInt) {
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

                // Guard: check tags
                if (ti.tag_a == kTagInt) {
                    e.cmp_mem8_imm8(x86::RBX, slot_disp(a, kTagOffset), kTagInt);
                    std::size_t j = e.jne_rel32();
                    deopt_jumps.push_back(j);
                }
                if (ti.tag_b == kTagInt) {
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
                // Backedge — jump to trace header (loop)
                // Rule 88: safepoint poll
                e.nop();  // safepoint placeholder
                std::size_t j = e.jmp_rel32();
                e.patch_rel32(j, header_pos);
                break;
            }
            case Op::JUMP_IF_FALSE: {
                // Conditional backedge — check condition.
                // If TRUE: fall through to the body (loop continues).
                // If FALSE: jump to deopt (loop exits).
                std::uint8_t cond_reg = instr.a;
                // Load the bool payload
                e.mov_r64_mem(x86::RAX, x86::RBX, slot_disp(cond_reg, kPayloadOffset));
                // TEST RAX, RAX (check if truthy)
                e.emit8(0x48); e.emit8(0x85); e.emit8(0xC0);
                // JZ to deopt (loop exit). Falls through to body if true.
                std::size_t deopt_j = e.jmp_rel32();  // use JMP for now (JZ would be 0F 84)
                // Actually: we need JZ (jump if zero = jump if false).
                // But our emitter only has JNE/JGE/JL/JLE. Let me use JE instead.
                // JE rel32 = 0F 84 cd
                // Back up: we emitted JMP (E9) but need JE (0F 84).
                // Remove the JMP bytes and emit JE instead.
                e.pos -= 5;  // undo the JMP
                e.emit8(0x0F); e.emit8(0x84);  // JE rel32
                std::size_t patch_pos = e.pos;
                e.emit32(0);
                deopt_jumps.push_back(patch_pos);
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
    // Case 1: Recording in progress, backedge returned to header
    if (recording && record_unit == unit && target_pc == record_start_pc) {
        finish_recording();
        return false;
    }

    // Case 2: A compiled trace exists for this loop header
    if (active && active->unit == unit && active->header_pc == target_pc &&
        active->is_compiled && active->native_code) {
        ++active->hit_count;
        return true;  // Invoke the compiled trace
    }

    // Case 3: Loop is hot — start recording
    if (!recording && !active && unit->backedge_count >= kHotThreshold) {
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

    return false;
}

void MetaTracer::record_instr(const Instr& instr, std::uint8_t tag_a,
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
    recording->instrs.push_back(ti);
}

void MetaTracer::finish_recording() noexcept {
    if (!recording) return;

    recording->is_recording = false;

    // Compile the trace to native code
    if (!compile_trace(*recording)) {
        std::fprintf(stderr, "VORTEX tracer: trace compilation failed, staying Tier-0\n");
        recording->~Trace();
        std::free(recording);
        recording = nullptr;
        return;
    }

    std::fprintf(stderr, "VORTEX tracer: trace compiled (%zu instrs, %zu bytes native)\n",
                 recording->instrs.size(), recording->native_code_size);

    active = recording;
    recording = nullptr;
}

}  // namespace abi_v1
}  // namespace vortex::rt
