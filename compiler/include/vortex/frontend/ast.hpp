// =============================================================================
// vortex/frontend/ast.hpp — Python-subset AST
//
// Purpose:
//   Parsed program representation consumed by Pass 1 (frontend lowering).
//   Nodes are arena-allocated (Rule 7) and reference symbols via SymbolId
//   (Rule 16). String literal bytes live in the Module's string pool.
//
// Subset (docs/frontend-subset.md):
//   stmts : def / class / return / assign / augassign / if-elif-else / while /
//           for-in / break / continue / pass / global / nonlocal / assert /
//           raise / try-except-else-finally / del / expr / import (own modules)
//   exprs : boolop / not / compare-chain / binop / unary / lambda / ternary /
//           call / attr / subscript / list / tuple / dict / listcomp /
//           genexp-lite (for sum/gen consumption) / yield / literals
// =============================================================================

#pragma once

#include <cstdint>

#include "vortex/stdx/small_vector.hpp"
#include "vortex/support/arena.hpp"
#include "vortex/support/symbol_table.hpp"

namespace vortex::fe {

inline namespace abi_v1 {

using vortex::SymbolId;

enum class ExprKind : std::uint8_t {
    Name, IntLit, FloatLit, StrLit, BoolLit, NoneLit,
    BinOp, UnaryOp, BoolOp, Compare, Call, Attribute, Subscript,
    ListLit, TupleLit, DictLit, ListComp, Lambda, IfExp, Yield,
};

enum class StmtKind : std::uint8_t {
    FunctionDef, ClassDef, Return, Assign, AugAssign, If, While, For,
    Break, Continue, Pass, Global, Nonlocal, Assert, Raise, Try, Expr,
    Del, Import,
};

struct Expr;
struct Stmt;

/// One uniform statement-list type across the AST (avoids cross-capacity
/// container conversions; Rule 57 — no structural duplication).
using StmtList = stdx::small_vector<Stmt*, 16>;
/// Module string pool type shared with the Lexer.
using StringPool = stdx::small_vector<char, 4096>;

struct Comprehension {
    Expr* target{};      // Name (or Tuple of Names)
    Expr* iter{};
    Expr* cond{};        // may be null
    bool is_genexp{false};
};

constexpr SymbolId arg_invalid = 0xFFFF'FFFF;   // positional
constexpr SymbolId arg_star = 0xFFFF'FFFE;      // *expr expansion
constexpr SymbolId arg_kwargs = 0xFFFF'FFFD;    // **expr expansion

struct Argument {
    Expr* value{};
    SymbolId keyword{arg_invalid};
};

struct DictEntry {
    Expr* key{};
    Expr* value{};
};

struct ExceptClause {
    SymbolId type_name{0xFFFF'FFFF};   // class name to match (resolved at runtime); invalid = bare
    SymbolId bind_name{0xFFFF'FFFF};   // `as name` binding; invalid = none
    StmtList body{};
};

struct Expr {
    ExprKind kind;
    std::uint32_t line{0};
    // payload union-ish (per kind)
    SymbolId name{0xFFFF'FFFF};       // Name
    std::int64_t int_value{0};        // IntLit
    double float_value{0};            // FloatLit
    std::uint32_t str_offset{0};      // StrLit -> module string pool
    std::uint32_t str_length{0};
    bool bool_value{false};           // BoolLit
    std::uint16_t op{0};              // BinOp kind / Compare kind / Unary kind / BoolOp
    stdx::small_vector<Expr*, 4> args{};   // BinOp operands / Call args / BoolOp values /
                                           // List/Tuple elements / Compare chain
    stdx::small_vector<Argument, 4> call_args{};  // Call (positional + keyword)
    Expr* sub{nullptr};               // Unary operand / subscript base / attribute base /
                                      // IfExp cond / listcomp elt / yield value
    SymbolId attr{0xFFFF'FFFF};       // Attribute name
    Expr* index{nullptr};             // Subscript index
    Expr* lower{nullptr};             // IfExp then / dict value
    Expr* upper{nullptr};             // IfExp else
    Comprehension comp{};             // ListComp / genexp
    // Lambda: parameter names (subset: no default args in lambdas).
    stdx::small_vector<SymbolId, 4> lambda_params{};
    // Compare chain: op codes stored as (CmpOpKind+1); pairs operands in args.
    stdx::small_vector<std::uint16_t, 2> cmp_ops{};
    // DictLit: args is [k0,v0,k1,v1,...] pairs.
};

struct Stmt {
    StmtKind kind;
    std::uint32_t line{0};
    SymbolId name{0xFFFF'FFFF};       // def/class name / assign target / global names
    stdx::small_vector<SymbolId, 4> names{};   // global/nonlocal/import lists
    stdx::small_vector<Expr*, 4> targets{};    // assign targets (chained a = b = e)
    stdx::small_vector<Expr*, 4> decorators_unused{};
    Expr* value{nullptr};             // assign rhs / return value / expr stmt / raise exc
    Expr* iter{nullptr};              // for iterable
    Expr* cond{nullptr};              // if/while condition / assert test
    Expr* test{nullptr};              // assert message
    std::uint16_t aug_op{0};          // AugAssign binop kind
    // def
    stdx::small_vector<SymbolId, 8> params{};
    stdx::small_vector<Expr*, 8> defaults{};
    bool has_varargs{false};
    bool has_kwargs{false};
    StmtList body{};
    StmtList orelse{};
    Expr* for_target{};             // For loop target
    // class
    SymbolId base_name{0xFFFF'FFFF};  // single inheritance base
    // try
    stdx::small_vector<ExceptClause, 4> handlers{};
    StmtList finalbody{};
};

struct Module {
    stdx::small_vector<Stmt*, 128> body{};
    StringPool string_pool{};   // cooked string bytes
    BumpArena ast_arena;                              // owns all Expr/Stmt
    SymbolId module_name{0xFFFF'FFFF};

    template <typename T, typename... Args>
    [[nodiscard]] T* make(Args&&... args) noexcept {
        return ast_arena.create<T>(std::forward<Args>(args)...);
    }
    [[nodiscard]] std::string_view string(std::uint32_t offset, std::uint32_t len) const noexcept {
        return std::string_view(string_pool.data() + offset, len);
    }
    /// Intern a cooked string into the pool; returns offset/length pair.
    std::pair<std::uint32_t, std::uint32_t> pool_string(std::string_view text) noexcept {
        std::uint32_t off = static_cast<std::uint32_t>(string_pool.size());
        for (char c : text) string_pool.push_back(c);
        return {off, static_cast<std::uint32_t>(text.size())};
    }
};

}  // namespace abi_v1
}  // namespace vortex::fe
