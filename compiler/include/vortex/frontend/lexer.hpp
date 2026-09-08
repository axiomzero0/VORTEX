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
#include <string>
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
    KwNot, KwOr, KwPass, KwRaise, KwReturn, KwTry, KwWhile, KwWith, KwYield,
    // operators & delimiters
    Plus, Minus, Star, Slash, DoubleSlash, Percent, At, StarStar,
    Lt, Gt, LtEq, GtEq, EqEq, NotEq, Arrow,
    Bang, Tilde, Amp, Pipe, Caret, Shl, Shr,
    LParen, RParen, LBracket, RBracket, LBrace, RBrace,
    Comma, Colon, Semi, Dot, Assign,
    ColonAssign,  // := (walrus operator, PEP 572)
    // Augmented assignment operators (PEP 203)
    PlusEq, MinusEq, StarEq, SlashEq, DoubleSlashEq, PercentEq, StarStarEq,
    AmpEq, PipeEq, CaretEq, ShlEq, ShrEq, AtEq,
    // F-string literal (PEP 498) — carries parts for interpolation
    FStrLit,
};

/// One part of an f-string: either a literal text segment or an
/// expression source to be parsed and str()'d at runtime.
struct FStrPart {
    std::string_view text{};    // literal text OR expression source
    bool is_expr{false};        // true = expression source, false = literal
    char conversion{0};          // 'r', 's', 'a', or 0 for none
};

enum class Kw : std::uint8_t {
    None_ = 0, False_, True_, And, As, Assert, Break, Class, Continue, Def,
    Del, Elif, Else, Except, Finally, For, From, Global, If, Import, In,
    Is, Lambda, Nonlocal, Not, Or, Pass, Raise, Return, Try, While, With, Yield,
};

struct Token {
    TokKind kind{TokKind::End};
    std::uint32_t line{1};
    std::uint32_t col{1};
    std::string_view text{};      // identifier / string bytes (cooked for StrLit)
    std::int64_t int_value{0};
    double float_value{0};
    /// Index into the Lexer's fstr_parts_side_ vector (only for FStrLit
    /// tokens; 0 = none). We store f-string parts in a side-vector rather
    /// than inline in the Token to keep Token small (otherwise the 512-
    /// inline token vector exceeds the 64KB small_vector inline limit).
    std::uint32_t fstr_parts_idx{0};
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

    /// Access f-string parts by index (stored in Token::fstr_parts_idx).
    /// Called by the Parser when building the f-string concatenation AST.
    /// Index 0 is reserved (empty). Returns nullptr if out of range.
    [[nodiscard]] const stdx::small_vector<FStrPart, 4>* fstr_parts(
        std::uint32_t idx) const noexcept {
        if (idx == 0 || idx >= fstr_parts_side_.size()) return nullptr;
        return &fstr_parts_side_[idx];
    }

    /// The f-string parts side-vector. Public so compile_to_ast can
    /// pass its address to the Parser via set_fstr_parts().
    stdx::small_vector<stdx::small_vector<FStrPart, 4>, 16> fstr_parts_side_{};

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

    // LEX-1 fix: stable storage for cooked string-literal bytes. The
    // StringPool (a small_vector<char,4096>) can reallocate when a string
    // literal is large or when many strings push it past 4 KB. Tokens
    // store std::string_view into the pool; if the pool reallocates,
    // previously-built views dangle -> use-after-free. We snapshot each
    // cooked string into an immutable heap allocation (the std::string
    // body never moves once we stop mutating it) and use that stable
    // pointer in the token instead.
    //
    // CRITICAL: the inline capacity must be large enough to hold ALL
    // string literals in a module without reallocation. If it reallocates,
    // the std::string objects are MOVED to a new buffer, and all token
    // string_views (which point into the std::string's SBO buffer) dangle.
    // This was the root cause of the "18-line string corruption" bug:
    // modules with 17+ string literals triggered a reallocation that
    // zeroed the first byte of every earlier string.
    stdx::small_vector<std::string, 256> stabilized_strings_{};

    // Stable storage for f-string parts. Each FStrPart::text string_view
    // points into one of these stabilized std::strings.
    stdx::small_vector<std::string, 32> fstr_stabilized_{};

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
