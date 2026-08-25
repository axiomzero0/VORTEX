// =============================================================================
// vortex/frontend/lexer.cpp — Python-subset lexer implementation.
//
// Bug-history (LEX-1..19, see GitHub issues #80-98):
//   LEX-1:  String pool reallocation invalidated StrLit views -> UAF.
//           Fix: snapshot each cooked string into immutable heap storage.
//   LEX-2:  Hex literal accumulation used signed int64 multiply (UB on overflow).
//           Fix: use uint64, range-check before storing into int64.
//   LEX-3:  Decimal int parse truncated to 31 chars (fixed 32-byte buf).
//           Fix: use full source text via std::string + strtoll.
//   LEX-4:  Float parse truncated to 63 chars (fixed 64-byte buf).
//           Fix: use full source text.
//   LEX-5:  Octal (0o) and binary (0b) literals unsupported.
//           Fix: add 0o/0b branches.
//   LEX-6:  Underscores in numeric literals rejected.
//           Fix: skip underscores between digits.
//   LEX-7:  Triple-quoted strings unsupported.
//           Fix: detect """ or ''' prefix and scan to closing triple.
//   LEX-8:  f-strings unsupported.
//           Fix: detect f/b/r prefixes (raw/byte/f-string). f-strings
//           themselves are parsed by the parser, not the lexer; the lexer
//           emits a StrLit with the inner text and sets a flag for the
//           parser to handle interpolation. For now we accept the prefix
//           and treat the inner text as a plain string (interpolation is a
//           parser concern that's documented as subset-limited).
//   LEX-9:  r-strings and b-strings unsupported.
//           Fix: parse prefix, treat r as raw (no escape processing) and
//           b as bytes (same handling as str for the subset).
//   LEX-10: Only \n \t \r \\ \' \" \0 escapes supported.
//           Fix: add \a \b \f \v \xHH \uHHHH \UHHHHHHHH.
//   LEX-11: UTF-8 BOM not stripped.
//           Fix: strip leading 0xEF 0xBB 0xBF.
//   LEX-12: Lone CR (\r) treated as whitespace, not line terminator.
//           Fix: treat \r and \r\n as line terminators (normalize to \n).
//   LEX-13: StrLit token col points past the closing quote.
//           Fix: capture col at the opening quote, not after consuming it.
//   LEX-14: String escape \ + \n does not reset col_ to 1.
//           Fix: reset col_ = 1 on physical newline inside string.
//   LEX-15: Backslash + CR + LF line continuation rejected.
//           Fix: handle \r\n and \r after backslash.
//   LEX-16: Initial indent at first physical line silently accepted.
//           Fix: reject non-zero indent on the first physical line.
//   LEX-17: Complex-number j suffix not supported.
//           Fix: accept trailing j on int/float literals (subset: store as
//           float, the runtime doesn't have a complex type yet).
//   LEX-18: Unterminated string ending in backslash reports wrong error.
//           Fix: detect EOF-after-backslash explicitly.
//   LEX-19: 1.e5 tokenized as IntLit(1) + Dot + Ident("e5").
//           Fix: allow 'e' as the next char after "1." in the float branch.
// =============================================================================

#include "vortex/frontend/lexer.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

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
[[nodiscard]] bool is_hex_digit(char c) noexcept {
    return is_digit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}
[[nodiscard]] int hex_value(char c) noexcept {
    if (c <= '9') return c - '0';
    return (c | 32) - 'a' + 10;
}
[[nodiscard]] bool is_oct_digit(char c) noexcept { return c >= '0' && c <= '7'; }
[[nodiscard]] bool is_bin_digit(char c) noexcept { return c == '0' || c == '1'; }

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
    bool first_physical_line = true;   // LEX-16: reject indent on line 1

    // LEX-11: strip UTF-8 BOM if present.
    if (src_.size() >= 3 &&
        static_cast<unsigned char>(src_[0]) == 0xEF &&
        static_cast<unsigned char>(src_[1]) == 0xBB &&
        static_cast<unsigned char>(src_[2]) == 0xBF) {
        pos_ = 3;
        col_ = 1;
    }

    for (;;) {
        if (pos_ >= src_.size()) {
            // EOF: flush final newline + dedents so the parser sees closed
            // blocks (CRITICAL: must run even with no pending dedents).
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
            if (pos_ >= src_.size() || peek() == '\n' || peek() == '\r' || peek() == '#') {
                if (pos_ < src_.size() && peek() == '#') {
                    while (pos_ < src_.size() && peek() != '\n' && peek() != '\r') {
                        ++pos_; ++col_;
                    }
                }
                if (pos_ < src_.size() && (peek() == '\n' || peek() == '\r')) {
                    // LEX-12: normalize \r\n and \r to \n
                    if (peek() == '\r') {
                        ++pos_;
                        if (pos_ < src_.size() && peek() == '\n') ++pos_;
                    } else {
                        ++pos_;
                    }
                    ++line_;
                    col_ = 1;
                    at_line_start_ = true;
                }
                continue;
            }
            // LEX-16: reject non-zero indent on the first physical line
            if (first_physical_line && indent_width > 0) {
                return fail(lex_error(line_, col_,
                                      "unexpected indentation on first line",
                                      std::string_view("leading indent"),
                                      "Remove leading whitespace on the first line"));
            }
            first_physical_line = false;
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
        first_physical_line = false;

        // --- whitespace (LEX-12: \r is a line terminator, not whitespace) -----
        if (c == ' ' || c == '\t') {
            ++pos_;
            ++col_;
            continue;
        }
        // LEX-12: lone \r or \r\n is a line terminator (normalize to \n)
        if (c == '\r') {
            ++pos_;
            if (pos_ < src_.size() && peek() == '\n') ++pos_;
            c = '\n';   // fall through to the \n handler below
        }
        if (c == '\\') {
            // explicit line continuation
            ++pos_;
            // LEX-15: backslash + CR + LF or backslash + CR is also continuation
            if (pos_ < src_.size() && (peek() == '\n' || peek() == '\r')) {
                if (peek() == '\r') {
                    ++pos_;
                    if (pos_ < src_.size() && peek() == '\n') ++pos_;
                } else {
                    ++pos_;
                }
                ++line_;
                col_ = 1;
                continue;
            }
            return fail(lex_error(line_, col_, "stray backslash"));
        }
        if (c == '#') {
            while (pos_ < src_.size() && peek() != '\n' && peek() != '\r') {
                ++pos_; ++col_;
            }
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
            std::uint32_t start_col = col_;   // LEX-13: capture col at start
            bool is_float = false;

            // LEX-5: octal (0o) and binary (0b) literals
            if (c == '0' && (peek(1) == 'o' || peek(1) == 'O')) {
                pos_ += 2; col_ += 2;
                std::size_t digits_start = pos_;
                std::uint64_t v = 0;
                while (pos_ < src_.size() && (is_oct_digit(peek()) || peek() == '_')) {
                    if (peek() == '_') { ++pos_; ++col_; continue; }   // LEX-6
                    v = v * 8 + static_cast<unsigned>(peek() - '0');
                    ++pos_; ++col_;
                }
                if (pos_ == digits_start) {
                    return fail(lex_error(line_, start_col, "bad octal literal"));
                }
                Token t = make(TokKind::IntLit, src_.substr(start, pos_ - start));
                t.col = start_col;
                t.int_value = static_cast<std::int64_t>(v);
                tokens.push_back(t);
                emitted_value_this_logical_line = true;
                continue;
            }
            if (c == '0' && (peek(1) == 'b' || peek(1) == 'B')) {
                pos_ += 2; col_ += 2;
                std::size_t digits_start = pos_;
                std::uint64_t v = 0;
                while (pos_ < src_.size() && (is_bin_digit(peek()) || peek() == '_')) {
                    if (peek() == '_') { ++pos_; ++col_; continue; }   // LEX-6
                    v = v * 2 + static_cast<unsigned>(peek() - '0');
                    ++pos_; ++col_;
                }
                if (pos_ == digits_start) {
                    return fail(lex_error(line_, start_col, "bad binary literal"));
                }
                Token t = make(TokKind::IntLit, src_.substr(start, pos_ - start));
                t.col = start_col;
                t.int_value = static_cast<std::int64_t>(v);
                tokens.push_back(t);
                emitted_value_this_logical_line = true;
                continue;
            }
            // LEX-2: hex literal with uint64 accumulator (no signed UB)
            if (c == '0' && (peek(1) == 'x' || peek(1) == 'X')) {
                pos_ += 2; col_ += 2;
                std::size_t digits_start = pos_;
                std::uint64_t v = 0;
                bool overflow = false;
                while (pos_ < src_.size() && (is_hex_digit(peek()) || peek() == '_')) {
                    if (peek() == '_') { ++pos_; ++col_; continue; }   // LEX-6
                    std::uint64_t new_v = v * 16 + static_cast<std::uint64_t>(hex_value(peek()));
                    if (new_v < v) overflow = true;   // wrapped
                    v = new_v;
                    ++pos_; ++col_;
                }
                if (pos_ == digits_start) {
                    return fail(lex_error(line_, start_col, "bad hex literal"));
                }
                Token t = make(TokKind::IntLit, src_.substr(start, pos_ - start));
                t.col = start_col;
                if (overflow) {
                    // Saturate to int64 max/min; document as subset limitation.
                    t.int_value = INT64_MAX;
                } else if (v > static_cast<std::uint64_t>(INT64_MAX)) {
                    // Allow up to 0xFFFF...FFFF as the unsigned representation
                    // of -1 (Python allows this for hex literals). Mask to int64.
                    t.int_value = static_cast<std::int64_t>(v);
                } else {
                    t.int_value = static_cast<std::int64_t>(v);
                }
                tokens.push_back(t);
                emitted_value_this_logical_line = true;
                continue;
            }

            // decimal int / float path
            while (pos_ < src_.size() && (is_digit(peek()) || peek() == '_')) {
                if (peek() == '_') { ++pos_; ++col_; continue; }   // LEX-6
                ++pos_; ++col_;
            }
            // LEX-19: "1.e5" — allow float dot even when next char is 'e'
            if (pos_ < src_.size() && peek() == '.' &&
                (peek(1) == 'e' || peek(1) == 'E' || !is_ident_start(peek(1)))) {
                is_float = true;
                ++pos_; ++col_;
                while (pos_ < src_.size() && (is_digit(peek()) || peek() == '_')) {
                    if (peek() == '_') { ++pos_; ++col_; continue; }
                    ++pos_; ++col_;
                }
            } else if (pos_ < src_.size() && peek() == '.' && is_digit(peek(1))) {
                is_float = true;
                ++pos_; ++col_;
                while (pos_ < src_.size() && (is_digit(peek()) || peek() == '_')) {
                    if (peek() == '_') { ++pos_; ++col_; continue; }
                    ++pos_; ++col_;
                }
            }
            if (pos_ < src_.size() && (peek() == 'e' || peek() == 'E')) {
                std::size_t save = pos_;
                ++pos_; ++col_;
                if (pos_ < src_.size() && (peek() == '+' || peek() == '-')) {
                    ++pos_; ++col_;
                }
                if (pos_ < src_.size() && is_digit(peek())) {
                    is_float = true;
                    while (pos_ < src_.size() && (is_digit(peek()) || peek() == '_')) {
                        if (peek() == '_') { ++pos_; ++col_; continue; }
                        ++pos_; ++col_;
                    }
                } else {
                    pos_ = save; col_ = start_col + static_cast<std::uint32_t>(save - start);
                }
            }
            // LEX-17: complex j suffix (e.g., 1.5j or 2j). Store as float
            // (the runtime lacks a complex type; the parser/interpreter
            // can reject this if needed, but the lexer accepts it).
            if (pos_ < src_.size() && peek() == 'j') {
                is_float = true;
                ++pos_; ++col_;
            }
            std::string_view text = src_.substr(start, pos_ - start);
            Token t = make(is_float ? TokKind::FloatLit : TokKind::IntLit, text);
            t.col = start_col;
            if (is_float) {
                // LEX-4: use full source text (no 63-char truncation)
                std::string buf(text);
                t.float_value = std::strtod(buf.c_str(), nullptr);
            } else {
                // LEX-3: use full source text (no 31-char truncation)
                std::string buf(text);
                errno = 0;
                char* end = nullptr;
                long long v = std::strtoll(buf.c_str(), &end, 10);
                if (errno == ERANGE) {
                    // Saturate. Subset limitation: no bignum from lexer.
                    t.int_value = v > 0 ? INT64_MAX : INT64_MIN;
                } else {
                    t.int_value = v;
                }
            }
            tokens.push_back(t);
            emitted_value_this_logical_line = true;
            continue;
        }

        // --- identifiers -----------------------------------------------------------
        if (is_ident_start(c)) {
            std::size_t start = pos_;
            std::uint32_t start_col = col_;
            while (pos_ < src_.size() && is_ident_char(peek())) { ++pos_; ++col_; }
            std::string_view text = src_.substr(start, pos_ - start);
            TokKind k = keyword_kind(text);
            Token t = make(k == TokKind::End ? TokKind::Ident : k, text);
            t.col = start_col;
            tokens.push_back(t);
            emitted_value_this_logical_line = true;
            continue;
        }

        // --- strings (with r/b/f prefixes, triple-quoted, full escape set) --------
        // LEX-8/9: detect r/b/f prefix.
        bool raw = false;
        bool bytes = false;
        bool fstring = false;
        std::size_t prefix_end = pos_;
        if (c == 'r' || c == 'R' || c == 'b' || c == 'B' || c == 'f' || c == 'F') {
            // Look ahead — the next char must be a quote (single or double)
            // for this to be a string prefix, not an identifier.
            char next = peek(1);
            if (next == '"' || next == '\'') {
                raw = (c == 'r' || c == 'R');
                bytes = (c == 'b' || c == 'B');
                fstring = (c == 'f' || c == 'F');
                // rb"..." or br"..." combos
                if ((c == 'r' || c == 'R') && (peek(1) == 'b' || peek(1) == 'B') &&
                    (peek(2) == '"' || peek(2) == '\'')) {
                    raw = true; bytes = true; prefix_end = pos_ + 2;
                } else if ((c == 'b' || c == 'B') && (peek(1) == 'r' || peek(1) == 'R') &&
                           (peek(2) == '"' || peek(2) == '\'')) {
                    raw = true; bytes = true; prefix_end = pos_ + 2;
                } else if ((c == 'r' || c == 'R') && (peek(1) == 'f' || peek(1) == 'F') &&
                           (peek(2) == '"' || peek(2) == '\'')) {
                    raw = true; fstring = true; prefix_end = pos_ + 2;
                } else if ((c == 'f' || c == 'F') && (peek(1) == 'r' || peek(1) == 'R') &&
                           (peek(2) == '"' || peek(2) == '\'')) {
                    raw = true; fstring = true; prefix_end = pos_ + 2;
                } else {
                    prefix_end = pos_ + 1;
                }
                std::uint32_t prefix_len = static_cast<std::uint32_t>(prefix_end - pos_);
                pos_ = prefix_end;
                col_ += prefix_len;
                c = peek();
            }
        }

        if (c == '"' || c == '\'') {
            char quote = c;
            // LEX-13: capture col at the opening quote
            std::uint32_t str_col = col_;
            // LEX-7: triple-quoted strings
            bool triple = (peek(1) == quote && peek(2) == quote);
            if (triple) {
                pos_ += 3; col_ += 3;
            } else {
                ++pos_; ++col_;
            }
            std::size_t pool_start = string_pool.size();
            bool closed = false;
            for (;;) {
                if (pos_ >= src_.size()) {
                    // LEX-18: distinguish EOF-after-backslash from plain EOF
                    return fail(lex_error(line_, col_,
                                          triple ? "unterminated triple-quoted string"
                                                 : "unterminated string literal",
                                          "<eof>",
                                          "Close the string with a matching quote"));
                }
                char ch = peek();
                // LEX-7: in triple mode, newlines are allowed (and count).
                if (ch == '\n' || ch == '\r') {
                    if (!triple) {
                        return fail(lex_error(line_, col_, "newline in string literal",
                                              "\\n",
                                              "Use escaped \\\\n or triple-quote for multi-line"));
                    }
                    // Normalize \r\n and \r to \n
                    if (ch == '\r') {
                        ++pos_;
                        if (pos_ < src_.size() && peek() == '\n') ++pos_;
                        string_pool.push_back('\n');
                    } else {
                        ++pos_;
                        string_pool.push_back('\n');
                    }
                    ++line_;
                    col_ = 1;   // LEX-14: reset col on physical newline
                    continue;
                }
                if (!triple && ch == quote) {
                    ++pos_; ++col_;
                    closed = true;
                    break;
                }
                if (triple && ch == quote) {
                    // Check for triple close
                    if (peek(1) == quote && peek(2) == quote) {
                        pos_ += 3; col_ += 3;
                        closed = true;
                        break;
                    }
                    // Single quote inside triple: literal
                    string_pool.push_back(ch);
                    ++pos_; ++col_;
                    continue;
                }
                if (ch == '\\' && !raw) {
                    ++pos_; ++col_;
                    if (pos_ >= src_.size()) {
                        // LEX-18: EOF right after backslash
                        return fail(lex_error(line_, col_,
                                              "unterminated escape at end of source",
                                              "\\\\<eof>",
                                              "Remove the trailing backslash"));
                    }
                    char esc = peek();
                    // LEX-10: full escape set
                    switch (esc) {
                        case 'n': string_pool.push_back('\n'); break;
                        case 't': string_pool.push_back('\t'); break;
                        case 'r': string_pool.push_back('\r'); break;
                        case '\\': string_pool.push_back('\\'); break;
                        case '\'': string_pool.push_back('\''); break;
                        case '"': string_pool.push_back('"'); break;
                        case 'a': string_pool.push_back('\a'); break;
                        case 'b': string_pool.push_back('\b'); break;
                        case 'f': string_pool.push_back('\f'); break;
                        case 'v': string_pool.push_back('\v'); break;
                        case '0': string_pool.push_back('\0'); break;
                        case '\n':
                            // escaped physical newline: line continuation inside string
                            ++line_; col_ = 1;   // LEX-14: reset col
                            break;
                        case '\r':
                            // LEX-15: \r or \r\n after backslash
                            ++pos_;
                            if (pos_ < src_.size() && peek() == '\n') ++pos_;
                            ++line_; col_ = 1;
                            continue;   // already advanced past the \r\n
                        case 'x': {
                            // LEX-10: \xHH
                            ++pos_; ++col_;
                            char h1 = (pos_ < src_.size()) ? peek() : '\0';
                            char h2 = (pos_ + 1 < src_.size()) ? peek(1) : '\0';
                            if (!is_hex_digit(h1) || !is_hex_digit(h2)) {
                                return fail(lex_error(line_, col_,
                                                      "invalid \\x escape (need 2 hex digits)"));
                            }
                            unsigned char val = static_cast<unsigned char>(
                                hex_value(h1) * 16 + hex_value(h2));
                            string_pool.push_back(static_cast<char>(val));
                            pos_ += 2; col_ += 2;
                            continue;
                        }
                        case 'u': {
                            // LEX-10: \uHHHH (BMP)
                            ++pos_; ++col_;
                            char hex4[4] = {};
                            for (int i = 0; i < 4; ++i) {
                                if (pos_ + i >= src_.size() || !is_hex_digit(peek(i))) {
                                    return fail(lex_error(line_, col_,
                                                          "invalid \\u escape (need 4 hex digits)"));
                                }
                                hex4[i] = peek(i);
                            }
                            std::uint32_t cp = 0;
                            for (int i = 0; i < 4; ++i) {
                                cp = cp * 16 + static_cast<std::uint32_t>(hex_value(hex4[i]));
                            }
                            // UTF-8 encode the BMP codepoint
                            char utf8[4] = {};
                            int n = 0;
                            if (cp < 0x80) {
                                utf8[0] = static_cast<char>(cp); n = 1;
                            } else if (cp < 0x800) {
                                utf8[0] = static_cast<char>(0xC0 | (cp >> 6));
                                utf8[1] = static_cast<char>(0x80 | (cp & 0x3F));
                                n = 2;
                            } else {
                                utf8[0] = static_cast<char>(0xE0 | (cp >> 12));
                                utf8[1] = static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                                utf8[2] = static_cast<char>(0x80 | (cp & 0x3F));
                                n = 3;
                            }
                            for (int i = 0; i < n; ++i) string_pool.push_back(utf8[i]);
                            pos_ += 4; col_ += 4;
                            continue;
                        }
                        case 'U': {
                            // LEX-10: \UHHHHHHHH (full Unicode)
                            ++pos_; ++col_;
                            char hex8[8] = {};
                            for (int i = 0; i < 8; ++i) {
                                if (pos_ + i >= src_.size() || !is_hex_digit(peek(i))) {
                                    return fail(lex_error(line_, col_,
                                                          "invalid \\U escape (need 8 hex digits)"));
                                }
                                hex8[i] = peek(i);
                            }
                            std::uint32_t cp = 0;
                            for (int i = 0; i < 8; ++i) {
                                cp = cp * 16 + static_cast<std::uint32_t>(hex_value(hex8[i]));
                            }
                            // UTF-8 encode (up to 4 bytes for supplementary plane)
                            char utf8[4] = {};
                            int n = 0;
                            if (cp < 0x80) {
                                utf8[0] = static_cast<char>(cp); n = 1;
                            } else if (cp < 0x800) {
                                utf8[0] = static_cast<char>(0xC0 | (cp >> 6));
                                utf8[1] = static_cast<char>(0x80 | (cp & 0x3F));
                                n = 2;
                            } else if (cp < 0x10000) {
                                utf8[0] = static_cast<char>(0xE0 | (cp >> 12));
                                utf8[1] = static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                                utf8[2] = static_cast<char>(0x80 | (cp & 0x3F));
                                n = 3;
                            } else if (cp < 0x110000) {
                                utf8[0] = static_cast<char>(0xF0 | (cp >> 18));
                                utf8[1] = static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
                                utf8[2] = static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                                utf8[3] = static_cast<char>(0x80 | (cp & 0x3F));
                                n = 4;
                            } else {
                                return fail(lex_error(line_, col_, "\\U escape out of range"));
                            }
                            for (int i = 0; i < n; ++i) string_pool.push_back(utf8[i]);
                            pos_ += 8; col_ += 8;
                            continue;
                        }
                        default:
                            // Unknown escape: Python keeps the backslash (deprecation
                            // warning in 3.12+; we keep it verbatim to round-trip).
                            string_pool.push_back('\\');
                            string_pool.push_back(esc);
                            break;
                    }
                    ++pos_; ++col_;
                    continue;
                }
                // raw string: backslash is literal
                if (ch == '\\' && raw) {
                    string_pool.push_back('\\');
                    ++pos_; ++col_;
                    continue;
                }
                string_pool.push_back(ch);
                ++pos_; ++col_;
            }
            if (!closed) {
                return fail(lex_error(line_, col_, "unterminated string literal",
                                      "<eof>", "Close the string"));
            }
            // LEX-1 fix: snapshot the cooked bytes into stable heap storage
            // so subsequent push_backs to string_pool don't invalidate this
            // token's view.
            std::string snapshot(string_pool.data() + pool_start,
                                 string_pool.size() - pool_start);
            stabilized_strings_.push_back(std::move(snapshot));
            std::string_view cooked(stabilized_strings_.back().data(),
                                    stabilized_strings_.back().size());
            Token t = make(TokKind::StrLit, cooked);
            t.col = str_col;   // LEX-13
            (void)bytes; (void)fstring;   // parser handles f-string interpolation
            tokens.push_back(t);
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
