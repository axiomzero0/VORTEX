// =============================================================================
// vortex/frontend/parser.hpp — Python-subset recursive-descent parser
//
// Purpose:
//   Token stream (from Lexer) -> Module AST. Standard Python grammar subset
//   with precedence climbing for expressions:
//     lambda < ifexp < or < and < not < comparison < | < ^ < & < shift
//     < + - < * @ / // % < unary -+~ < ** < trailer (call/attr/subscript)
//
// Diagnostics (Rule 47): every error names the offending token, its line,
// what was expected, and a suggested fix.
// =============================================================================

#pragma once

#include "vortex/frontend/ast.hpp"
#include "vortex/frontend/lexer.hpp"

namespace vortex::fe {

inline namespace abi_v1 {

class Parser {
public:
    Parser(stdx::small_vector<Token, 512>& tokens, Module& module) noexcept
        : tokens_(tokens), module_(module) {}

    [[nodiscard]] Result<void> parse_module() noexcept;

private:
    // --- token helpers ---------------------------------------------------
    [[nodiscard]] const Token& peek(std::uint32_t ahead = 0) const noexcept {
        std::size_t i = pos_ + ahead;
        if (i >= tokens_.size()) i = tokens_.size() - 1;
        return tokens_[i];
    }
    const Token& advance() noexcept {
        const Token& t = tokens_[pos_];
        if (pos_ + 1 < tokens_.size()) ++pos_;
        return t;
    }
    [[nodiscard]] bool check(TokKind k) const noexcept { return peek().kind == k; }
    [[nodiscard]] bool accept(TokKind k) noexcept {
        if (check(k)) {
            advance();
            return true;
        }
        return false;
    }
    [[nodiscard]] Result<Token> expect(TokKind k, const char* what) noexcept;

    [[nodiscard]] Diagnostic err_expected(const char* expected) noexcept;

    // --- statements ---------------------------------------------------------
    [[nodiscard]] Result<Stmt*> parse_statement() noexcept;
    [[nodiscard]] Result<StmtList> parse_block() noexcept;  // after ':'
    [[nodiscard]] Result<Stmt*> parse_function_def() noexcept;
    [[nodiscard]] Result<Stmt*> parse_class_def() noexcept;
    [[nodiscard]] Result<Stmt*> parse_if() noexcept;
    [[nodiscard]] Result<Stmt*> parse_while() noexcept;
    [[nodiscard]] Result<Stmt*> parse_for() noexcept;
    [[nodiscard]] Result<Stmt*> parse_try() noexcept;
    [[nodiscard]] Result<Stmt*> parse_simple_stmt() noexcept;

    // --- expressions ----------------------------------------------------------
    [[nodiscard]] Result<Expr*> parse_expr() noexcept;              // lambda / ternary allowed
    [[nodiscard]] Result<Expr*> parse_testlist(bool allow_tuple) noexcept;
    [[nodiscard]] Result<Expr*> parse_or() noexcept;
    [[nodiscard]] Result<Expr*> parse_and() noexcept;
    [[nodiscard]] Result<Expr*> parse_not() noexcept;
    [[nodiscard]] Result<Expr*> parse_comparison() noexcept;        // chains
    [[nodiscard]] Result<Expr*> parse_bitor() noexcept;
    [[nodiscard]] Result<Expr*> parse_bitxor() noexcept;
    [[nodiscard]] Result<Expr*> parse_bitand() noexcept;
    [[nodiscard]] Result<Expr*> parse_shift() noexcept;
    [[nodiscard]] Result<Expr*> parse_arith() noexcept;
    [[nodiscard]] Result<Expr*> parse_term() noexcept;
    [[nodiscard]] Result<Expr*> parse_factor() noexcept;
    [[nodiscard]] Result<Expr*> parse_power() noexcept;
    [[nodiscard]] Result<Expr*> parse_atom_with_trailers() noexcept;
    [[nodiscard]] Result<Expr*> parse_atom() noexcept;
    [[nodiscard]] Result<Expr*> parse_call_args(Expr* callee) noexcept;
    [[nodiscard]] Result<Expr*> parse_listcomp_or_list(Expr* first) noexcept;
    [[nodiscard]] Result<Expr*> parse_dict_or_set() noexcept;
    [[nodiscard]] Result<Expr*> parse_tuple_tail(Expr* first) noexcept;
    [[nodiscard]] Result<Expr*> parse_comprehension_tail(Expr* elt, bool genexp) noexcept;
    [[nodiscard]] bool stmt_boundaries() const noexcept;

    [[nodiscard]] Expr* new_expr(ExprKind k, std::uint32_t line) noexcept;
    [[nodiscard]] Stmt* new_stmt(StmtKind k, std::uint32_t line) noexcept;
    [[nodiscard]] SymbolId intern_tok(const Token& t) noexcept;

    stdx::small_vector<Token, 512>& tokens_;
    Module& module_;
    std::size_t pos_{0};
    std::uint32_t errors_{0};
    std::uint32_t suppress_in_operator_{0};   // >0 while parsing a for-header target
};

/// Convenience: lex + parse in one call.
[[nodiscard]] Result<Module*> compile_to_ast(BumpArena& module_arena,
                                             std::string_view source) noexcept;

}  // namespace abi_v1
}  // namespace vortex::fe
