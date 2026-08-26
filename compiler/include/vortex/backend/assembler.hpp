// =============================================================================
// vortex/backend/assembler.hpp — Zero-allocation x86-64 encoder (Rule 7)
//
// A real encoder: REX prefixes, ModRM, SIB, imm8/32, rel32 branches. The
// code buffer is caller-owned (backed by the compilation arena); the
// assembler only bumps a cursor. Buffer exhaustion returns
// std::unexpected(Diagnostic) — never throws (Rule 6).
//
// Emission rules (Rule 23: every field mask named):
//   REX.W = 0x48 (64-bit operand size) | R(4) X(2) B(1) extensions
//   ModRM = mod(2) reg(3) rm(3); SIB = scale(2) index(3) base(3)
// =============================================================================

#pragma once

#include <cstdint>

#include "vortex/support/result.hpp"

namespace vortex::backend {

inline namespace abi_v1 {

class Assembler {
public:
    Assembler(std::byte* buffer, std::size_t capacity) noexcept
        : buffer_(buffer), cursor_(buffer), capacity_(capacity) {}

    [[nodiscard]] std::size_t size() const noexcept {
        return static_cast<std::size_t>(cursor_ - buffer_);
    }
    [[nodiscard]] std::size_t remaining() const noexcept { return capacity_ - size(); }
    [[nodiscard]] std::byte* data() const noexcept { return buffer_; }
    [[nodiscard]] std::byte* cursor() const noexcept { return cursor_; }

    // --- raw ---------------------------------------------------------------------
    [[nodiscard]] bool emit8(std::uint8_t b) noexcept {
        if (remaining() < 1) [[unlikely]] return false;
        *cursor_++ = static_cast<std::byte>(b);
        return true;
    }
    [[nodiscard]] bool emit32(std::uint32_t v) noexcept {
        if (remaining() < 4) [[unlikely]] return false;
        for (int i = 0; i < 4; ++i) {
            *cursor_++ = static_cast<std::byte>((v >> (8 * i)) & 0xFF);
        }
        return true;
    }
    [[nodiscard]] bool emit64(std::uint64_t v) noexcept {
        if (remaining() < 8) [[unlikely]] return false;
        for (int i = 0; i < 8; ++i) {
            *cursor_++ = static_cast<std::byte>((v >> (8 * i)) & 0xFF);
        }
        return true;
    }

    // --- prefixes ------------------------------------------------------------------
    // REX prefixes: W=64-bit, R=reg extension, X=index extension, B=rm/base ext.
    static constexpr std::uint8_t REX = 0x40;
    static constexpr std::uint8_t REX_W = 0x48;
    static constexpr std::uint8_t REX_WB = 0x49;
    static constexpr std::uint8_t REX_WR = 0x4C;
    static constexpr std::uint8_t REX_WRB = 0x4D;

    // ModRM field masks (decode side) and 2-bit mod VALUES (encode side —
    // modrm() takes the 2-bit value and shifts; passing a mask here produced
    // mod=00 everywhere, the reg-reg/SIB encoding bug).
    static constexpr std::uint8_t kModMask = 0xC0;
    static constexpr std::uint8_t kRegMask = 0x38;
    static constexpr std::uint8_t kRmMask = 0x07;
    static constexpr std::uint8_t kModMem = 0;     // [reg] / [reg+disp32 via SIB]
    static constexpr std::uint8_t kModDisp8 = 1;   // [reg+disp8]
    static constexpr std::uint8_t kModDisp32 = 2;  // [reg+disp32]
    static constexpr std::uint8_t kModReg = 3;     // reg, reg

    [[nodiscard]] static std::uint8_t modrm(std::uint8_t mod, std::uint8_t reg,
                                            std::uint8_t rm) noexcept {
        // [[assume]] per Rule 21 — the encoders are closed over valid fields.
        return static_cast<std::uint8_t>(((mod << 6) & kModMask) |
                                         ((reg << 3) & kRegMask) | (rm & kRmMask));
    }

    // --- MOV reg, imm64 (REX.W B8+r io64) --------------------------------------------
    [[nodiscard]] bool mov_r64_imm64(std::uint8_t reg, std::uint64_t imm) noexcept {
        if (reg >= 8) {
            if (!emit8(REX_WB)) return false;
        } else {
            if (!emit8(REX_W)) return false;
        }
        if (!emit8(static_cast<std::uint8_t>(0xB8 + (reg & 7)))) return false;
        return emit64(imm);
    }

    // --- MOV r64, r64 (REX.W 89 /r, reg-form) ---------------------------------------
    // 89 /r: ModRM reg=src, rm=dst, mod=11 (register-register — rm=4 is the
    // RSP REGISTER here, never a SIB trigger; the earlier 8B-with-SIB-path
    // bug produced mov rdi,[r12] instead of mov r12,rdi).
    [[nodiscard]] bool mov_r64_r64(std::uint8_t dst, std::uint8_t src) noexcept {
        std::uint8_t rex = REX_W;
        if (src >= 8) rex |= 0x04;   // R extends the reg field
        if (dst >= 8) rex |= 0x01;   // B extends the rm field
        if (!emit8(rex)) return false;
        if (!emit8(0x89)) return false;
        return emit8(modrm(kModReg, src & 7, dst & 7));
    }

    // --- MOV r64, [r64 + disp] (REX.W 8B /r with base) ---------------------------------
    [[nodiscard]] bool mov_r64_mem(std::uint8_t dst, std::uint8_t base,
                                   std::int32_t disp) noexcept {
        std::uint8_t rex = REX_W;
        if (dst >= 8) rex |= 0x04;   // R extends reg field
        if (base >= 8) rex |= 0x01;  // B extends rm/base
        if (!emit8(rex)) return false;
        if (!emit8(0x8B)) return false;
        if (!modrm_with_base(dst & 7, base & 7, disp)) return false;
        return true;
    }

    // --- MOV [r64 + disp], r64 (REX.W 89 /r) ---------------------------------------------
    [[nodiscard]] bool mov_mem_r64(std::uint8_t base, std::int32_t disp,
                                   std::uint8_t src) noexcept {
        std::uint8_t rex = REX_W;
        if (src >= 8) rex |= 0x04;
        if (base >= 8) rex |= 0x01;
        if (!emit8(rex)) return false;
        if (!emit8(0x89)) return false;
        if (!modrm_with_base(src & 7, base & 7, disp)) return false;
        return true;
    }

    // --- MOV r/m64, imm32 sign-extended (REX.W C7 /0 id) ----------------------------------
    [[nodiscard]] bool mov_mem_imm32(std::uint8_t base, std::int32_t disp,
                                     std::int32_t imm) noexcept {
        std::uint8_t rex = REX_W;
        if (base >= 8) rex |= 0x01;
        if (!emit8(rex)) return false;
        if (!emit8(0xC7)) return false;
        if (!modrm_with_base(0, base & 7, disp)) return false;
        return emit32(static_cast<std::uint32_t>(imm));
    }

    // --- ALU ops: ADD(01), SUB(29), CMP(39), AND(21), OR(09), XOR(31) r/m, r ----------------
    [[nodiscard]] bool alu_r64_r64(std::uint8_t opcode_ext, std::uint8_t dst,
                                   std::uint8_t src) noexcept {
        // opcode_ext is the full 0x01/0x29/0x39 one-byte opcode.
        std::uint8_t rex = REX_W;
        if (src >= 8) rex |= 0x04;
        if (dst >= 8) rex |= 0x01;
        if (!emit8(rex)) return false;
        if (!emit8(opcode_ext)) return false;
        return emit8(modrm(kModReg, src & 7, dst & 7));
    }

    // --- ALU r64, imm32 (REX.W 81 /x id) -----------------------------------------------------
    [[nodiscard]] bool alu_r64_imm32(std::uint8_t modrm_reg, std::uint8_t dst,
                                     std::int32_t imm) noexcept {
        std::uint8_t rex = REX_W;
        if (dst >= 8) rex |= 0x01;
        if (!emit8(rex)) return false;
        if (!emit8(0x81)) return false;
        if (!emit8(modrm(kModReg, modrm_reg & 7, dst & 7))) return false;
        return emit32(static_cast<std::uint32_t>(imm));
    }

    // --- NEG r/m64 (REX.W F7 /3) — one-operand two's-complement negation ---------------
    // The IBE-20 differential test caught that the Neg IR node was silently
    // lowered through the default "copy from home slot" path, producing
    // `result = x` instead of `result = -x` in the JIT. This emitter wraps
    // the F7 /3 form (single-byte opcode with modrm.reg=3 for NEG, vs the
    // /0..5 sub-opcodes F7 dispatches on). REX.W is forced for 64-bit
    // operand size — int payloads are 64-bit by the IR's Value contract.
    [[nodiscard]] bool neg_r64(std::uint8_t reg) noexcept {
        std::uint8_t rex = REX_W;
        if (reg >= 8) rex |= 0x01;   // B extends rm field
        if (!emit8(rex)) return false;
        if (!emit8(0xF7)) return false;
        // modrm(mod=11, reg=3, rm=reg) — /3 is the NEG opcode extension.
        return emit8(modrm(kModReg, 3, reg & 7));
    }

    // --- IMUL r64, r/m64 (REX.W 0F AF /r) ------------------------------------------------------
    [[nodiscard]] bool imul_r64_r64(std::uint8_t dst, std::uint8_t src) noexcept {
        std::uint8_t rex = REX_W;
        if (dst >= 8) rex |= 0x04;
        if (src >= 8) rex |= 0x01;
        if (!emit8(rex)) return false;
        if (!emit8(0x0F)) return false;
        if (!emit8(0xAF)) return false;
        return emit8(modrm(kModReg, dst & 7, src & 7));
    }

    // --- JMP rel32 (E9 cd) — returns the patch site for later fixup -----------------------------
    [[nodiscard]] std::size_t jmp_rel32() noexcept {
        std::size_t site = size();
        if (emit8(0xE9) && emit32(0)) return site;
        return k_invalid_site;
    }
    // --- JMP rel32 (E9 cd) — single-byte opcode + 4-byte rel -------------------------------
    // IBE-6 fix: previous code computed rel = target - (site + 4), but the
    // JMP E9 instruction is 5 bytes total (1 opcode + 4 rel). After
    // execution RIP = site + 5, so rel = target - (site + 5). The off-by-
    // one made every patched JMP land 1 byte past its target. We also
    // document the Jcc variant separately below.
    void patch_rel32(std::size_t site, std::size_t target) noexcept {
        std::int32_t rel = static_cast<std::int32_t>(target - (site + 5));
        std::byte* at = buffer_ + site + 1;
        for (int i = 0; i < 4; ++i) {
            at[i] = static_cast<std::byte>((static_cast<std::uint32_t>(rel) >> (8 * i)) & 0xFF);
        }
    }

    // --- Jcc rel32 (0F 8x cd) — two-byte opcode + 4-byte rel --------------------------------
    [[nodiscard]] std::size_t jcc_rel32(std::uint8_t cond_low) noexcept {
        std::size_t site = size();
        if (emit8(0x0F) && emit8(static_cast<std::uint8_t>(0x80 | (cond_low & 0x0F))) &&
            emit32(0)) {
            return site;
        }
        return k_invalid_site;
    }
    // IBE-7 fix: patch_jcc must write to site+2 (the rel32 starts AFTER
    // the two-byte 0F 8x opcode) and use rel = target - (site + 6) (the
    // Jcc instruction is 6 bytes total; RIP = site + 6 after execution).
    // The previous code delegated to patch_rel32, which writes to site+1
    // (overwriting the 0x8x opcode byte) and uses rel = target - (site+4)
    // — both the write location AND the rel formula were wrong.
    void patch_jcc(std::size_t site, std::size_t target) noexcept {
        std::int32_t rel = static_cast<std::int32_t>(target - (site + 6));
        std::byte* at = buffer_ + site + 2;
        for (int i = 0; i < 4; ++i) {
            at[i] = static_cast<std::byte>((static_cast<std::uint32_t>(rel) >> (8 * i)) & 0xFF);
        }
    }

    // --- CALL rel32 (E8 cd) -------------------------------------------------------------------
    [[nodiscard]] std::size_t call_rel32() noexcept {
        std::size_t site = size();
        if (emit8(0xE8) && emit32(0)) return site;
        return k_invalid_site;
    }
    // IBE-8 fix: CALL E8 cd is 5 bytes total (1 opcode + 4 rel). After
    // execution RIP = buffer_base + site + 5, so rel = target - (buffer_base
    // + site + 5). The previous code used site + 4 — off by one — so the
    // call landed 1 byte past the target.
    void patch_call(std::size_t site, std::uintptr_t target) noexcept {
        std::int32_t rel = static_cast<std::int32_t>(
            static_cast<std::intptr_t>(target) - static_cast<std::intptr_t>(site + 5 + reinterpret_cast<std::uintptr_t>(buffer_)));
        std::byte* at = buffer_ + site + 1;
        for (int i = 0; i < 4; ++i) {
            at[i] = static_cast<std::byte>((static_cast<std::uint32_t>(rel) >> (8 * i)) & 0xFF);
        }
    }

    // --- PUSH/POP (50+rd / 58+rd) ------------------------------------------------------------
    [[nodiscard]] bool push_r64(std::uint8_t reg) noexcept {
        if (reg >= 8) { if (!emit8(0x41)) return false; }
        return emit8(static_cast<std::uint8_t>(0x50 + (reg & 7)));
    }
    [[nodiscard]] bool pop_r64(std::uint8_t reg) noexcept {
        if (reg >= 8) { if (!emit8(0x41)) return false; }
        return emit8(static_cast<std::uint8_t>(0x58 + (reg & 7)));
    }

    // --- RET (C3) --------------------------------------------------------------------------------
    [[nodiscard]] bool ret() noexcept { return emit8(0xC3); }

    // --- NOP (90) — safepoint placeholder ------------------------------------------------------
    [[nodiscard]] bool nop() noexcept { return emit8(0x90); }

    // --- JMP rax (FF E0) — indirect tail-call through a register -----------------------------
    [[nodiscard]] bool jmp_rax_placeholder() noexcept {
        if (!emit8(0xFF)) return false;
        return emit8(0xE0);
    }

    static constexpr std::size_t k_invalid_site = static_cast<std::size_t>(-1);

private:
    // ModRM with a base register and displacement; handles the SIB-required
    // encodings (rm == 100 needs SIB; base == 101 with mod=00 is RIP-rel,
    // so force disp8/disp32).
    [[nodiscard]] bool modrm_with_base(std::uint8_t reg, std::uint8_t base,
                                       std::int32_t disp) noexcept {
        if (disp == 0 && (base & 7) != 5 /* RBP/R13 needs disp */) {
            if ((base & 7) == 4) {   // RSP/R12 needs SIB
                if (!emit8(modrm(kModMem, reg, 4))) return false;
                return emit8(0x24);   // SIB: scale=0, index=RSP(none), base=base
            }
            return emit8(modrm(0x00, reg, base & 7));
        }
        if (disp >= -128 && disp <= 127) {
            if ((base & 7) == 4) {
                if (!emit8(modrm(kModDisp8, reg, 4))) return false;
                if (!emit8(0x24)) return false;
            } else {
                if (!emit8(modrm(kModDisp8, reg, base & 7))) return false;
            }
            return emit8(static_cast<std::uint8_t>(disp & 0xFF));
        }
        if ((base & 7) == 4) {
            if (!emit8(modrm(kModDisp32, reg, 4))) return false;
            if (!emit8(0x24)) return false;
        } else {
            if (!emit8(modrm(kModDisp32, reg, base & 7))) return false;
        }
        return emit32(static_cast<std::uint32_t>(disp));
    }

    std::byte* buffer_{nullptr};
    std::byte* cursor_{nullptr};
    std::size_t capacity_{0};
};

}  // namespace abi_v1
}  // namespace vortex::backend
