// =============================================================================
// vortex/frontend/lexer.hpp — Python-subset lexer
//
// Purpose:
//   Tokenizes the VORTEX Python subset: identifiers, keywords, int/float
//   literals, strings (with escapes; no raw triple-quote docstrings needed
//   by the supported subset), all operators, and Python's significant
//   whitespace (INDENT/DEDENT tokens with paren-depth suppression).
//
// Invariants:
//   - Tokens carry 1-based line/col for Rule 47 diagnostics.
//   - Indentation stack is per-file; tabs are rejected with an actionable
//     diagnostic (consistent with PEP 8 space-only policy).
//
// Subset boundary (documented in docs/frontend-subset.md): no walrus, no
// match statement, no async, no decorators, no f-string nested quotes.
// =============================================================================

#pragma once

#include <cstdint>
#include <string_view>

#include "vortex/frontend/string_pool.hpp"
#include "vortex/stdx/small_vector.hpp"
#include "vortex/support/diagnostic.hpp"
#include "vortex/support/result.hpp"

namespace vortex::fe {

inline namespace abi_v1 {

enum class TokKind : std::uint8_t {
    End,
    Newline,
    Indent,
    Dedent,
    Ident,       // also keywords (see Kw enum)
    IntLit,
    FloatLit,
    StrLit,      // cooked bytes in text
    KwFalse, KwNone, KwTrue, KwAnd, KwAs, KwAssert, KwBreak, KwClass,
    KwContinue, KwDef, KwDel, KwElif, KwElse, KwExcept, KwFinally, KwFor,
    KwFrom, KwGlobal, KwIf, KwImport, KwIn, KwIs, KwLambda, KwNonlocal,
    KwNot, KwOr, KwPass, KwRaise, KwReturn, KwTry, KwWhile, KwYield,
    // operators & delimiters
    Plus, Minus, Star, Slash, DoubleSlash, Percent, At, StarStar,
    Lt, Gt, LtEq, GtEq, EqEq, NotEq, Arrow,
    Bang, Tilde, Amp, Pipe, Caret, Shl, Shr,
    LParen, RParen, LBracket, RBracket, LBrace, RBrace,
    Comma, Colon, Semi, Dot, Assign,
};

enum class Kw : std::uint8_t {
    None_ = 0, False_, True_, And, As, Assert, Break, Class, Continue, Def,
    Del, Elif, Else, Except, Finally, For, From, Global, If, Import, In,
    Is, Lambda, Nonlocal, Not, Or, Pass, Raise, Return, Try, While, Yield,
};

struct Token {
    TokKind kind{TokKind::End};
    std::uint32_t line{1};
    std::uint32_t col{1};
    std::string_view text{};      // identifier / string bytes (cooked for StrLit)
    std::int64_t int_value{0};
    double float_value{0};
};

/// Lexer over one source buffer. The buffer must outlive the Lexer and all
/// tokens (source is owned by the caller — usually the Module loader).
class Lexer {
public:
    explicit Lexer(std::string_view source) noexcept : src_(source) {}

    /// Scan the entire input into `tokens` (End-terminated).
    /// Errors produce Diagnostics with exact locations (Rule 47).
    [[nodiscard]] Result<void> run(stdx::small_vector<Token, 512>& tokens,
                                   StringPool& string_pool) noexcept;

private:
    [[nodiscard]] Result<void> scan_line_prefix(stdx::small_vector<Token, 512>& tokens) noexcept;
    [[nodiscard]] char peek(std::size_t ahead = 0) const noexcept {
        return pos_ + ahead < src_.size() ? src_[pos_ + ahead] : '\0';
    }
    [[nodiscard]] std::uint32_t line() const noexcept { return line_; }
    [[nodiscard]] Token make(TokKind k, std::string_view t) const noexcept {
        return Token{k, line_, col_, t, 0, 0.0};
    }

    std::string_view src_;
    std::size_t pos_{0};
    std::uint32_t line_{1};
    std::uint32_t col_{1};

    // indentation machinery
    static constexpr std::size_t max_indent = 64;
    stdx::small_vector<std::uint32_t, 16> indents_{0};
    std::uint32_t pending_dedents_{0};
    bool at_line_start_{true};
    bool last_was_value_{false};   // for implicit line joining detection
    std::uint32_t paren_depth_{0};
};

/// Map keyword text -> TokKind. Returns End if not a keyword.
[[nodiscard]] TokKind keyword_kind(std::string_view text) noexcept;

}  // namespace abi_v1
}  // namespace vortex::fe
