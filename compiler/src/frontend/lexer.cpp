// =============================================================================
// vortex/frontend/lexer.cpp — Python-subset lexer implementation.
// =============================================================================

#include "vortex/frontend/lexer.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace vortex::fe {
inline namespace abi_v1 {

TokKind keyword_kind(std::string_view text) noexcept {
    struct E { const char* n; TokKind k; };
    static constexpr E table[] = {
        {"False", TokKind::KwFalse}, {"None", TokKind::KwNone}, {"True", TokKind::KwTrue},
        {"and", TokKind::KwAnd}, {"as", TokKind::KwAs}, {"assert", TokKind::KwAssert},
        {"break", TokKind::KwBreak}, {"class", TokKind::KwClass},
        {"continue", TokKind::KwContinue}, {"def", TokKind::KwDef}, {"del", TokKind::KwDel},
        {"elif", TokKind::KwElif}, {"else", TokKind::KwElse}, {"except", TokKind::KwExcept},
        {"finally", TokKind::KwFinally}, {"for", TokKind::KwFor}, {"from", TokKind::KwFrom},
        {"global", TokKind::KwGlobal}, {"if", TokKind::KwIf}, {"import", TokKind::KwImport},
        {"in", TokKind::KwIn}, {"is", TokKind::KwIs}, {"lambda", TokKind::KwLambda},
        {"nonlocal", TokKind::KwNonlocal}, {"not", TokKind::KwNot}, {"or", TokKind::KwOr},
        {"pass", TokKind::KwPass}, {"raise", TokKind::KwRaise}, {"return", TokKind::KwReturn},
        {"try", TokKind::KwTry}, {"while", TokKind::KwWhile}, {"yield", TokKind::KwYield},
    };
    for (const E& e : table) {
        if (text == e.n) return e.k;
    }
    return TokKind::End;
}

namespace {
[[nodiscard]] bool is_ident_start(char c) noexcept {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}
[[nodiscard]] bool is_ident_char(char c) noexcept {
    return is_ident_start(c) || (c >= '0' && c <= '9');
}
[[nodiscard]] bool is_digit(char c) noexcept { return c >= '0' && c <= '9'; }

[[nodiscard]] Diagnostic lex_error(std::uint32_t line, std::uint32_t col,
                                   std::string_view msg, std::string_view actual = {},
                                   std::string_view fix = {}) noexcept {
    Diagnostic d = Diagnostic::error(msg, diag_code::lex_invalid_char);
    d.where.line = line;
    d.where.column = col;
    d.actual = actual.empty() ? msg : actual;
    if (!fix.empty()) d.fix = fix;
    return d;
}
}  // namespace

Result<void> Lexer::run(stdx::small_vector<Token, 512>& tokens,
                        StringPool& string_pool) noexcept {
    bool emitted_value_this_logical_line = false;

    for (;;) {
        if (pos_ >= src_.size()) {
            // EOF: flush final newline + dedents so the parser sees closed
            // blocks (crITICAL: must run even with no pending dedents).
            if (emitted_value_this_logical_line) {
                tokens.push_back(make(TokKind::Newline, ""));
                emitted_value_this_logical_line = false;
            }
            while (indents_.size() > 1) {
                indents_.pop_back();
                tokens.push_back(make(TokKind::Dedent, ""));
            }
            tokens.push_back(make(TokKind::End, ""));
            return {};
        }

        char c = peek();

        // --- line-start indentation handling ---------------------------------
        if (at_line_start_ && paren_depth_ == 0) {
            at_line_start_ = false;
            // measure indent
            std::uint32_t indent_width = 0;
            std::size_t save = pos_;
            std::uint32_t save_col = col_;
            while (pos_ < src_.size() && (peek() == ' ' || peek() == '\t')) {
                if (peek() == '\t') {
                    return fail(lex_error(line_, col_, "tab in indentation",
                                          "\\t",
                                          "Use 4 spaces per indent level (PEP 8)"));
                }
                ++indent_width;
                ++pos_;
                ++col_;
            }
            // blank line or comment-only line: skip entirely
            if (pos_ >= src_.size() || peek() == '\n' || peek() == '#') {
                if (pos_ < src_.size() && peek() == '#') {
                    while (pos_ < src_.size() && peek() != '\n') { ++pos_; ++col_; }
                }
                if (pos_ < src_.size() && peek() == '\n') {
                    ++pos_;
                    ++line_;
                    col_ = 1;
                    at_line_start_ = true;
                }
                (void)save;
                (void)save_col;
                continue;
            }
            // compare with indent stack
            std::uint32_t cur = indents_.back();
            if (indent_width > cur) {
                if (indents_.size() >= max_indent) {
                    return fail(lex_error(line_, col_, "indentation too deep"));
                }
                indents_.push_back(indent_width);
                tokens.push_back(make(TokKind::Indent, ""));
            } else if (indent_width < cur) {
                while (indents_.size() > 1 && indents_.back() > indent_width) {
                    indents_.pop_back();
                    tokens.push_back(make(TokKind::Dedent, ""));
                }
                if (indents_.back() != indent_width) {
                    // Rule 47: format the offending width into a stable buffer
                    // owned by the Diagnostic path (no std::string temporaries).
                    static thread_local char width_buf[16];
                    std::snprintf(width_buf, sizeof(width_buf), "%u", indent_width);
                    return fail(lex_error(line_, col_,
                                          "inconsistent dedent (does not match any outer level)",
                                          width_buf,
                                          "Align dedents with an enclosing indent level"));
                }
            }
            continue;
        }

        // --- whitespace -------------------------------------------------------
        if (c == ' ' || c == '\t' || c == '\r') {
            ++pos_;
            ++col_;
            continue;
        }
        if (c == '\\') {
            // explicit line continuation
            ++pos_;
            if (pos_ < src_.size() && peek() == '\n') {
                ++pos_;
                ++line_;
                col_ = 1;
                continue;
            }
            return fail(lex_error(line_, col_, "stray backslash"));
        }
        if (c == '#') {
            while (pos_ < src_.size() && peek() != '\n') { ++pos_; ++col_; }
            continue;
        }
        if (c == '\n') {
            ++pos_;
            ++line_;
            col_ = 1;
            if (paren_depth_ == 0) {
                if (emitted_value_this_logical_line) {
                    tokens.push_back(make(TokKind::Newline, ""));
                    emitted_value_this_logical_line = false;
                    at_line_start_ = true;
                } else {
                    at_line_start_ = true;   // blank logical line
                }
            }
            continue;
        }

        // --- numbers ------------------------------------------------------------
        if (is_digit(c) || (c == '.' && is_digit(peek(1)))) {
            std::size_t start = pos_;
            bool is_float = false;
            if (c == '0' && (peek(1) == 'x' || peek(1) == 'X')) {
                pos_ += 2;
                std::size_t digits_start = pos_;
                while (pos_ < src_.size() &&
                       ((peek() >= '0' && peek() <= '9') || (peek() >= 'a' && peek() <= 'f') ||
                        (peek() >= 'A' && peek() <= 'F'))) {
                    ++pos_;
                }
                std::string_view hex = src_.substr(digits_start, pos_ - digits_start);
                if (hex.empty()) return fail(lex_error(line_, col_, "bad hex literal"));
                std::int64_t v = 0;
                for (char h : hex) {
                    int d = (h <= '9') ? h - '0' : (h | 32) - 'a' + 10;
                    v = v * 16 + d;
                }
                Token t = make(TokKind::IntLit, src_.substr(start, pos_ - start));
                t.int_value = v;
                tokens.push_back(t);
                col_ += static_cast<std::uint32_t>(pos_ - start);
                emitted_value_this_logical_line = true;
                continue;
            }
            while (pos_ < src_.size() && is_digit(peek())) { ++pos_; }
            if (pos_ < src_.size() && peek() == '.' && is_digit(peek(1))) {
                is_float = true;
                ++pos_;
                while (pos_ < src_.size() && is_digit(peek())) { ++pos_; }
            } else if (pos_ < src_.size() && peek() == '.' &&
                       !is_ident_start(peek(1) == '\0' ? ' ' : peek(1))) {
                // "1." float
                is_float = true;
                ++pos_;
            }
            if (pos_ < src_.size() && (peek() == 'e' || peek() == 'E')) {
                std::size_t save = pos_;
                ++pos_;
                if (pos_ < src_.size() && (peek() == '+' || peek() == '-')) ++pos_;
                if (pos_ < src_.size() && is_digit(peek())) {
                    is_float = true;
                    while (pos_ < src_.size() && is_digit(peek())) { ++pos_; }
                } else {
                    pos_ = save;   // not an exponent (e.g., "1e" ident boundary)
                }
            }
            std::string_view text = src_.substr(start, pos_ - start);
            Token t = make(is_float ? TokKind::FloatLit : TokKind::IntLit, text);
            if (is_float) {
                std::array<char, 64> buf{};
                std::memcpy(buf.data(), text.data(), text.size() < 64 ? text.size() : 63);
                t.float_value = std::strtod(buf.data(), nullptr);
            } else {
                std::array<char, 32> buf{};
                std::memcpy(buf.data(), text.data(), text.size() < 32 ? text.size() : 31);
                t.int_value = std::strtoll(buf.data(), nullptr, 10);
            }
            tokens.push_back(t);
            col_ += static_cast<std::uint32_t>(pos_ - start);
            emitted_value_this_logical_line = true;
            continue;
        }

        // --- identifiers -----------------------------------------------------------
        if (is_ident_start(c)) {
            std::size_t start = pos_;
            while (pos_ < src_.size() && is_ident_char(peek())) { ++pos_; }
            std::string_view text = src_.substr(start, pos_ - start);
            TokKind k = keyword_kind(text);
            Token t = make(k == TokKind::End ? TokKind::Ident : k, text);
            tokens.push_back(t);
            col_ += static_cast<std::uint32_t>(pos_ - start);
            emitted_value_this_logical_line = true;
            continue;
        }

        // --- strings ------------------------------------------------------------
        if (c == '"' || c == '\'') {
            char quote = c;
            ++pos_;
            ++col_;
            std::size_t pool_start = string_pool.size();
            for (;;) {
                if (pos_ >= src_.size()) {
                    return fail(lex_error(line_, col_, "unterminated string literal",
                                          "<eof>",
                                          "Close the string with a matching quote"));
                }
                char ch = peek();
                if (ch == '\n') {
                    return fail(lex_error(line_, col_, "newline in string literal",
                                          "\\n",
                          "Use escaped \\\\n or close the string on one line"));
                }
                if (ch == quote) {
                    ++pos_;
                    ++col_;
                    break;
                }
                if (ch == '\\') {
                    ++pos_;
                    ++col_;
                    char esc = peek();
                    switch (esc) {
                        case 'n': string_pool.push_back('\n'); break;
                        case 't': string_pool.push_back('\t'); break;
                        case 'r': string_pool.push_back('\r'); break;
                        case '\\': string_pool.push_back('\\'); break;
                        case '\'': string_pool.push_back('\''); break;
                        case '"': string_pool.push_back('"'); break;
                        case '0': string_pool.push_back('\0'); break;
                        case '\n': ++line_; break;   // escaped physical newline
                        default:
                            return fail(lex_error(line_, col_, "unknown escape sequence",
                                                  std::string_view(&esc, 1),
                                                  "Supported: \\n \\t \\r \\\\ \\' \\\" \\0"));
                    }
                    ++pos_;
                    ++col_;
                    continue;
                }
                string_pool.push_back(ch);
                ++pos_;
                ++col_;
            }
            std::string_view cooked(string_pool.data() + pool_start,
                                    string_pool.size() - pool_start);
            tokens.push_back(make(TokKind::StrLit, cooked));
            emitted_value_this_logical_line = true;
            continue;
        }

        // --- operators (longest match first) --------------------------------------
        auto try_op = [&](std::string_view op, TokKind k) -> bool {
            if (src_.compare(pos_, op.size(), op) == 0) {
                tokens.push_back(make(k, op));
                pos_ += op.size();
                col_ += static_cast<std::uint32_t>(op.size());
                switch (k) {
                    case TokKind::LParen: case TokKind::LBracket: case TokKind::LBrace:
                        ++paren_depth_; break;
                    case TokKind::RParen:
                    case TokKind::RBracket:
                    case TokKind::RBrace:
                        if (paren_depth_ > 0) {
                            --paren_depth_;
                        }
                        break;
                    default: break;
                }
                emitted_value_this_logical_line = true;
                return true;
            }
            return false;
        };

        if (try_op("**", TokKind::StarStar) || try_op("//", TokKind::DoubleSlash) ||
            try_op("<=", TokKind::LtEq) || try_op(">=", TokKind::GtEq) ||
            try_op("==", TokKind::EqEq) || try_op("!=", TokKind::NotEq) ||
            try_op("<<", TokKind::Shl) || try_op(">>", TokKind::Shr) ||
            try_op("->", TokKind::Arrow) ||
            try_op("+", TokKind::Plus) || try_op("-", TokKind::Minus) ||
            try_op("*", TokKind::Star) || try_op("/", TokKind::Slash) ||
            try_op("%", TokKind::Percent) || try_op("@", TokKind::At) ||
            try_op("<", TokKind::Lt) || try_op(">", TokKind::Gt) ||
            try_op("!", TokKind::Bang) || try_op("~", TokKind::Tilde) ||
            try_op("&", TokKind::Amp) || try_op("|", TokKind::Pipe) ||
            try_op("^", TokKind::Caret) ||
            try_op("(", TokKind::LParen) || try_op(")", TokKind::RParen) ||
            try_op("[", TokKind::LBracket) || try_op("]", TokKind::RBracket) ||
            try_op("{", TokKind::LBrace) || try_op("}", TokKind::RBrace) ||
            try_op(",", TokKind::Comma) || try_op(":", TokKind::Colon) ||
            try_op(";", TokKind::Semi) || try_op(".", TokKind::Dot) ||
            try_op("=", TokKind::Assign)) {
            continue;
        }

        return fail(lex_error(line_, col_, "invalid character",
                              std::string_view(&c, 1),
                              "Character is outside the VORTEX Python subset"));
    }

    tokens.push_back(make(TokKind::End, ""));
    return {};
}

}  // namespace abi_v1
}  // namespace vortex::fe
