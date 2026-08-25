// =============================================================================
// vortex/frontend/parser.cpp — Python-subset parser implementation.
// =============================================================================

#include "vortex/frontend/parser.hpp"

#include "vortex/ir/node.hpp"

namespace vortex::fe {
inline namespace abi_v1 {

namespace {
using vortex::ir::BinOpKind;
using vortex::ir::CmpOpKind;

constexpr std::uint16_t bool_and = 1, bool_or = 2;
constexpr std::uint16_t un_neg = 1, un_invert = 2, un_not = 3;
}  // namespace

Expr* Parser::new_expr(ExprKind k, std::uint32_t line) noexcept {
    Expr* e = module_.make<Expr>();
    e->kind = k;
    e->line = line;
    return e;
}

Stmt* Parser::new_stmt(StmtKind k, std::uint32_t line) noexcept {
    Stmt* s = module_.make<Stmt>();
    s->kind = k;
    s->line = line;
    return s;
}

SymbolId Parser::intern_tok(const Token& t) noexcept {
    return global_symbols().intern(t.text);
}

Diagnostic Parser::err_expected(const char* expected) noexcept {
    const Token& t = peek();
    Diagnostic d = Diagnostic::error("parse: unexpected token", diag_code::parse_unexpected_token);
    d.where.line = t.line;
    d.where.column = t.col;
    d.expected = expected;
    d.actual = t.kind == TokKind::End ? "<end of file>" : t.text;
    d.fix = "Rewrite to match the VORTEX Python subset grammar (docs/frontend-subset.md)";
    return d;
}

Result<Token> Parser::expect(TokKind k, const char* what) noexcept {
    if (!check(k)) {
        return fail(err_expected(what));
    }
    return advance();
}

Result<void> Parser::parse_module() noexcept {
    while (!check(TokKind::End)) {
        if (accept(TokKind::Newline) || accept(TokKind::Indent) || accept(TokKind::Dedent)) {
            continue;
        }
        Result<Stmt*> s = parse_statement();
        if (!s) return std::unexpected(s.error());
        module_.body.push_back(*s);
    }
    return {};
}

Result<StmtList> Parser::parse_block() noexcept {
    auto colon = expect(TokKind::Colon, "':'");
    if (!colon) return std::unexpected(colon.error());
    StmtList out;

    if (accept(TokKind::Newline)) {
        if (!accept(TokKind::Indent)) {
            return fail(err_expected("an indented block"));
        }
        while (!check(TokKind::Dedent) && !check(TokKind::End)) {
            if (accept(TokKind::Newline)) continue;
            Result<Stmt*> s = parse_statement();
            if (!s) return std::unexpected(s.error());
            out.push_back(*s);
        }
        if (!accept(TokKind::Dedent)) {
            return fail(err_expected("DEDENT closing the block"));
        }
    } else {
        for (;;) {
            Result<Stmt*> s = parse_simple_stmt();
            if (!s) return std::unexpected(s.error());
            out.push_back(*s);
            if (accept(TokKind::Semi)) {
                if (check(TokKind::Newline)) break;
                continue;
            }
            auto nl = expect(TokKind::Newline, "newline after simple statement");
            if (!nl) return std::unexpected(nl.error());
            break;
        }
    }
    return out;
}

Result<Stmt*> Parser::parse_statement() noexcept {
    const Token& t = peek();
    switch (t.kind) {
        case TokKind::KwDef: return parse_function_def();
        case TokKind::KwClass: return parse_class_def();
        case TokKind::KwIf: return parse_if();
        case TokKind::KwWhile: return parse_while();
        case TokKind::KwFor: return parse_for();
        case TokKind::KwTry: return parse_try();
        case TokKind::At: {
            // PAR-1 fix: decorator must be followed by a newline, then def/class.
            // The previous code REJECTED @dec\n (newline after decorator)
            // and ACCEPTED @dec def (no newline) — backwards. Python grammar
            // requires the decorator on its own line, then def/class on the
            // next line. Each decorator ends with a newline.
            while (accept(TokKind::At)) {
                Result<Expr*> dec = parse_atom_with_trailers();
                if (!dec) return std::unexpected(dec.error());
                // Each decorator must end with a newline.
                auto nl = expect(TokKind::Newline, "newline after decorator");
                if (!nl) return std::unexpected(nl.error());
            }
            if (check(TokKind::KwDef)) return parse_function_def();
            if (check(TokKind::KwClass)) return parse_class_def();
            return fail_msg("parse: decorator must be followed by def or class",
                            diag_code::parse_unexpected_token);
        }
        default: break;
    }

    Result<Stmt*> first = parse_simple_stmt();
    if (!first) return first;
    if (check(TokKind::Semi)) {
        StmtList stmts;
        stmts.push_back(*first);
        while (accept(TokKind::Semi)) {
            if (check(TokKind::Newline) || check(TokKind::End)) break;
            Result<Stmt*> s = parse_simple_stmt();
            if (!s) return std::unexpected(s.error());
            stmts.push_back(*s);
        }
        auto nl = expect(TokKind::Newline, "newline at end of statement line");
        if (!nl) return std::unexpected(nl.error());
        if (stmts.size() == 1) return first;
        Stmt* seq = new_stmt(StmtKind::If, (*first)->line);
        Expr* truth = new_expr(ExprKind::BoolLit, t.line);
        truth->bool_value = true;
        seq->cond = truth;
        seq->body = std::move(stmts);
        return seq;
    }
    auto nl = expect(TokKind::Newline, "newline at end of statement");
    if (!nl) return std::unexpected(nl.error());
    return first;
}

Result<Stmt*> Parser::parse_function_def() noexcept {
    auto kw = expect(TokKind::KwDef, "'def'");
    if (!kw) return std::unexpected(kw.error());
    auto name = expect(TokKind::Ident, "function name");
    if (!name) return std::unexpected(name.error());
    Stmt* fn = new_stmt(StmtKind::FunctionDef, kw.value().line);
    fn->name = intern_tok(name.value());

    auto lp = expect(TokKind::LParen, "'(' after function name");
    if (!lp) return std::unexpected(lp.error());
    bool first = true;
    bool seen_star_args = false;   // PAR-6: track *args for kw-only detection
    bool seen_kwargs = false;     // PAR-5: **kwargs must be last
    while (!check(TokKind::RParen)) {
        if (!first) {
            auto comma = expect(TokKind::Comma, "',' between parameters");
            if (!comma) return std::unexpected(comma.error());
        }
        first = false;
        if (check(TokKind::Star)) {
            advance();
            // PAR-7 fix: bare '*' without a name is rejected (Python uses
            // it to mark subsequent params as keyword-only; subset rejects
            // it instead of silently setting has_varargs with no name).
            if (!check(TokKind::Ident) && check(TokKind::Comma)) {
                return fail_msg("parse: bare '*' without name is outside the VORTEX subset",
                                diag_code::parse_unexpected_token);
            }
            if (fn->has_kwargs || seen_kwargs) {
                return fail_msg("parse: *args after **kwargs", diag_code::parse_unexpected_token);
            }
            if (check(TokKind::Ident)) fn->params.push_back(intern_tok(advance()));
            fn->has_varargs = true;
            seen_star_args = true;
            continue;
        }
        if (check(TokKind::StarStar)) {
            advance();
            // PAR-8 fix: bare '**' without a name is rejected.
            if (!check(TokKind::Ident) && (check(TokKind::Comma) || check(TokKind::RParen))) {
                return fail_msg("parse: bare '**' without name is outside the VORTEX subset",
                                diag_code::parse_unexpected_token);
            }
            if (seen_kwargs) {
                return fail_msg("parse: duplicate **kwargs", diag_code::parse_unexpected_token);
            }
            if (check(TokKind::Ident)) fn->params.push_back(intern_tok(advance()));
            fn->has_kwargs = true;
            seen_kwargs = true;
            continue;
        }
        // PAR-5 fix: positional parameter after **kwargs is rejected.
        if (seen_kwargs) {
            return fail_msg("parse: parameter after **kwargs is not allowed",
                            diag_code::parse_unexpected_token);
        }
        auto p = expect(TokKind::Ident, "parameter name");
        if (!p) return std::unexpected(p.error());
        fn->params.push_back(intern_tok(p.value()));
        if (accept(TokKind::Assign)) {
            Result<Expr*> d = parse_expr();
            if (!d) return std::unexpected(d.error());
            fn->defaults.push_back(*d);
        } else if (!fn->defaults.empty() && !fn->has_varargs && !fn->has_kwargs) {
            return fail_msg("parse: non-default parameter after default parameter",
                            diag_code::parse_unexpected_token);
        }
        // (PAR-6 subset note: params after *args are keyword-only in Python.
        // We accept them as positional-with-keyword-default; the runtime's
        // bind_parameters honors has_varargs/has_kwargs for the call-site
        // contract. Full kw-only enforcement is a future improvement.)
    }
    auto rp = expect(TokKind::RParen, "')' closing parameter list");
    if (!rp) return std::unexpected(rp.error());

    if (accept(TokKind::Arrow)) {
        Result<Expr*> ann = parse_expr();   // annotation parsed, advisory only
        if (!ann) return std::unexpected(ann.error());
    }

    Result<StmtList> body = parse_block();
    if (!body) return std::unexpected(body.error());
    fn->body = std::move(*body);
    return fn;
}

Result<Stmt*> Parser::parse_class_def() noexcept {
    auto kw = expect(TokKind::KwClass, "'class'");
    if (!kw) return std::unexpected(kw.error());
    auto name = expect(TokKind::Ident, "class name");
    if (!name) return std::unexpected(name.error());
    Stmt* cls = new_stmt(StmtKind::ClassDef, kw.value().line);
    cls->name = intern_tok(name.value());

    if (accept(TokKind::LParen)) {
        if (check(TokKind::Ident)) cls->base_name = intern_tok(advance());
        auto rp = expect(TokKind::RParen, "')' after base class");
        if (!rp) return std::unexpected(rp.error());
    }
    Result<StmtList> body = parse_block();
    if (!body) return std::unexpected(body.error());
    cls->body = std::move(*body);
    return cls;
}

Result<Stmt*> Parser::parse_if() noexcept {
    auto kw = expect(TokKind::KwIf, "'if'");
    if (!kw) return std::unexpected(kw.error());
    Stmt* head = new_stmt(StmtKind::If, kw.value().line);
    Result<Expr*> cond = parse_expr();
    if (!cond) return std::unexpected(cond.error());
    head->cond = *cond;
    Result<StmtList> body = parse_block();
    if (!body) return std::unexpected(body.error());
    head->body = std::move(*body);

    // elif/else chain: nested in orelse (single element each level).
    Stmt* tail = head;
    for (;;) {
        if (check(TokKind::KwElif)) {
            advance();
            Stmt* elif = new_stmt(StmtKind::If, peek().line);
            Result<Expr*> c = parse_expr();
            if (!c) return std::unexpected(c.error());
            elif->cond = *c;
            Result<StmtList> b = parse_block();
            if (!b) return std::unexpected(b.error());
            elif->body = std::move(*b);
            tail->orelse.push_back(elif);
            tail = elif;
            continue;
        }
        if (check(TokKind::KwElse)) {
            advance();
            Result<StmtList> b = parse_block();
            if (!b) return std::unexpected(b.error());
            tail->orelse = std::move(*b);
        }
        break;
    }
    return head;
}

Result<Stmt*> Parser::parse_while() noexcept {
    auto kw = expect(TokKind::KwWhile, "'while'");
    if (!kw) return std::unexpected(kw.error());
    Stmt* st = new_stmt(StmtKind::While, kw.value().line);
    Result<Expr*> cond = parse_expr();
    if (!cond) return std::unexpected(cond.error());
    st->cond = *cond;
    Result<StmtList> body = parse_block();
    if (!body) return std::unexpected(body.error());
    st->body = std::move(*body);
    if (check(TokKind::KwElse)) {
        advance();
        Result<StmtList> b = parse_block();
        if (!b) return std::unexpected(b.error());
        st->orelse = std::move(*b);
    }
    return st;
}

Result<Stmt*> Parser::parse_for() noexcept {
    auto kw = expect(TokKind::KwFor, "'for'");
    if (!kw) return std::unexpected(kw.error());
    Stmt* st = new_stmt(StmtKind::For, kw.value().line);
    // For-header targets must not treat `in` as a comparison operator.
    ++suppress_in_operator_;
    Result<Expr*> target = parse_testlist(true);
    --suppress_in_operator_;
    if (!target) return std::unexpected(target.error());
    auto in = expect(TokKind::KwIn, "'in' after for-target");
    if (!in) return std::unexpected(in.error());
    Result<Expr*> iter = parse_testlist(false);
    if (!iter) return std::unexpected(iter.error());
    st->for_target = *target;
    st->iter = *iter;
    Result<StmtList> body = parse_block();
    if (!body) return std::unexpected(body.error());
    st->body = std::move(*body);
    if (check(TokKind::KwElse)) {
        advance();
        Result<StmtList> b = parse_block();
        if (!b) return std::unexpected(b.error());
        st->orelse = std::move(*b);
    }
    return st;
}

Result<Stmt*> Parser::parse_try() noexcept {
    auto kw = expect(TokKind::KwTry, "'try'");
    if (!kw) return std::unexpected(kw.error());
    Stmt* st = new_stmt(StmtKind::Try, kw.value().line);
    Result<StmtList> body = parse_block();
    if (!body) return std::unexpected(body.error());
    st->body = std::move(*body);

    while (check(TokKind::KwExcept)) {
        advance();
        ExceptClause clause;
        if (check(TokKind::Ident)) {
            clause.type_name = intern_tok(advance());
            if (accept(TokKind::KwAs)) {
                auto n = expect(TokKind::Ident, "exception binding name");
                if (!n) return std::unexpected(n.error());
                clause.bind_name = intern_tok(n.value());
            }
        }
        Result<StmtList> hb = parse_block();
        if (!hb) return std::unexpected(hb.error());
        clause.body = std::move(*hb);
        st->handlers.push_back(clause);
    }
    if (check(TokKind::KwElse)) {
        advance();
        Result<StmtList> b = parse_block();
        if (!b) return std::unexpected(b.error());
        st->orelse = std::move(*b);
    }
    if (check(TokKind::KwFinally)) {
        advance();
        Result<StmtList> b = parse_block();
        if (!b) return std::unexpected(b.error());
        st->finalbody = std::move(*b);
    }
    if (st->handlers.empty() && st->finalbody.empty()) {
        return fail_msg("parse: try requires at least one except or finally",
                        diag_code::parse_unexpected_token);
    }
    return st;
}

Result<Stmt*> Parser::parse_simple_stmt() noexcept {
    const Token& t = peek();
    switch (t.kind) {
        case TokKind::KwPass: advance(); return new_stmt(StmtKind::Pass, t.line);
        case TokKind::KwBreak: advance(); return new_stmt(StmtKind::Break, t.line);
        case TokKind::KwContinue: advance(); return new_stmt(StmtKind::Continue, t.line);
        case TokKind::KwReturn: {
            advance();
            Stmt* s = new_stmt(StmtKind::Return, t.line);
            if (!stmt_boundaries()) {
                Result<Expr*> v = parse_testlist(true);
                if (!v) return std::unexpected(v.error());
                s->value = *v;
            }
            return s;
        }
        case TokKind::KwRaise: {
            advance();
            Stmt* s = new_stmt(StmtKind::Raise, t.line);
            if (!stmt_boundaries()) {
                Result<Expr*> v = parse_expr();
                if (!v) return std::unexpected(v.error());
                s->value = *v;
            }
            return s;
        }
        case TokKind::KwGlobal:
        case TokKind::KwNonlocal: {
            bool is_global = t.kind == TokKind::KwGlobal;
            advance();
            Stmt* s = new_stmt(is_global ? StmtKind::Global : StmtKind::Nonlocal, t.line);
            for (;;) {
                auto n = expect(TokKind::Ident, "name");
                if (!n) return std::unexpected(n.error());
                s->names.push_back(intern_tok(n.value()));
                if (!accept(TokKind::Comma)) break;
            }
            return s;
        }
        case TokKind::KwDel: {
            advance();
            Stmt* s = new_stmt(StmtKind::Del, t.line);
            for (;;) {
                Result<Expr*> v = parse_atom_with_trailers();
                if (!v) return std::unexpected(v.error());
                s->targets.push_back(*v);
                if (!accept(TokKind::Comma)) break;
            }
            return s;
        }
        case TokKind::KwImport:
        case TokKind::KwFrom: {
            bool from = t.kind == TokKind::KwFrom;
            advance();
            Stmt* s = new_stmt(StmtKind::Import, t.line);
            auto mod = expect(TokKind::Ident, "module name");
            if (!mod) return std::unexpected(mod.error());
            s->name = intern_tok(mod.value());
            if (from) {
                auto imp = expect(TokKind::KwImport, "'import'");
                if (!imp) return std::unexpected(imp.error());
                for (;;) {
                    auto n = expect(TokKind::Ident, "imported name");
                    if (!n) return std::unexpected(n.error());
                    s->names.push_back(intern_tok(n.value()));
                    if (!accept(TokKind::Comma)) break;
                }
            }
            return s;
        }
        case TokKind::KwAssert: {
            advance();
            Stmt* s = new_stmt(StmtKind::Assert, t.line);
            Result<Expr*> test = parse_expr();
            if (!test) return std::unexpected(test.error());
            s->cond = *test;
            if (accept(TokKind::Comma)) {
                Result<Expr*> msg = parse_expr();
                if (!msg) return std::unexpected(msg.error());
                s->test = *msg;
            }
            return s;
        }
        default: break;
    }

    Result<Expr*> lhs = parse_testlist(true);
    if (!lhs) return std::unexpected(lhs.error());

    if (check(TokKind::Assign)) {
        Stmt* s = new_stmt(StmtKind::Assign, t.line);
        s->targets.push_back(*lhs);
        for (;;) {
            advance();   // consume '='
            if (check(TokKind::KwYield)) {
                Result<Expr*> v = parse_expr();
                if (!v) return std::unexpected(v.error());
                s->value = *v;
                break;
            }
            Result<Expr*> rhs = parse_testlist(true);
            if (!rhs) return std::unexpected(rhs.error());
            if (check(TokKind::Assign)) {
                s->targets.push_back(*rhs);
                continue;
            }
            s->value = *rhs;
            break;
        }
        return s;
    }

    std::uint16_t aug = 0xFFFF;
    switch (peek().kind) {
        case TokKind::Plus: aug = static_cast<std::uint16_t>(BinOpKind::Add); break;
        case TokKind::Minus: aug = static_cast<std::uint16_t>(BinOpKind::Sub); break;
        case TokKind::Star: aug = static_cast<std::uint16_t>(BinOpKind::Mul); break;
        case TokKind::Slash: aug = static_cast<std::uint16_t>(BinOpKind::TrueDiv); break;
        case TokKind::DoubleSlash: aug = static_cast<std::uint16_t>(BinOpKind::FloorDiv); break;
        case TokKind::Percent: aug = static_cast<std::uint16_t>(BinOpKind::Mod); break;
        case TokKind::StarStar: aug = static_cast<std::uint16_t>(BinOpKind::Pow); break;
        case TokKind::Shl: aug = static_cast<std::uint16_t>(BinOpKind::LShift); break;
        case TokKind::Shr: aug = static_cast<std::uint16_t>(BinOpKind::RShift); break;
        case TokKind::Amp: aug = static_cast<std::uint16_t>(BinOpKind::BitAnd); break;
        case TokKind::Pipe: aug = static_cast<std::uint16_t>(BinOpKind::BitOr); break;
        case TokKind::Caret: aug = static_cast<std::uint16_t>(BinOpKind::BitXor); break;
        default: aug = 0xFFFF; break;
    }
    if (aug != 0xFFFF) {
        advance();
        auto eq = expect(TokKind::Assign, "'=' after augmented operator");
        if (!eq) return std::unexpected(eq.error());
        Stmt* s = new_stmt(StmtKind::AugAssign, t.line);
        s->targets.push_back(*lhs);
        s->aug_op = aug;
        Result<Expr*> rhs = parse_testlist(true);
        if (!rhs) return std::unexpected(rhs.error());
        s->value = *rhs;
        return s;
    }

    Stmt* s = new_stmt(StmtKind::Expr, t.line);
    s->value = *lhs;
    return s;
}

bool Parser::stmt_boundaries() const noexcept {
    return check(TokKind::Newline) || check(TokKind::Semi) || check(TokKind::End);
}

// -----------------------------------------------------------------------------
// expressions
// -----------------------------------------------------------------------------
Result<Expr*> Parser::parse_expr() noexcept {
    if (check(TokKind::KwLambda)) {
        advance();
        Expr* lam = new_expr(ExprKind::Lambda, peek().line);
        for (;;) {
            if (check(TokKind::Colon)) break;
            auto n = expect(TokKind::Ident, "lambda parameter");
            if (!n) return std::unexpected(n.error());
            lam->lambda_params.push_back(intern_tok(n.value()));
            if (!accept(TokKind::Comma)) break;
        }
        auto colon = expect(TokKind::Colon, "':' in lambda");
        if (!colon) return std::unexpected(colon.error());
        Result<Expr*> body = parse_expr();
        if (!body) return std::unexpected(body.error());
        lam->sub = *body;
        return lam;
    }
    if (check(TokKind::KwYield)) {
        advance();
        Expr* y = new_expr(ExprKind::Yield, peek().line);
        if (!stmt_boundaries() && !check(TokKind::RParen) && !check(TokKind::Comma) &&
            !check(TokKind::RBracket) && !check(TokKind::RBrace)) {
            Result<Expr*> v = parse_testlist(true);
            if (!v) return std::unexpected(v.error());
            y->sub = *v;
        }
        return y;
    }
    Result<Expr*> e = parse_or();
    if (!e) return e;
    if (check(TokKind::KwIf)) {
        advance();
        Expr* tern = new_expr(ExprKind::IfExp, (*e)->line);
        Result<Expr*> c = parse_or();
        if (!c) return std::unexpected(c.error());
        auto el = expect(TokKind::KwElse, "'else' in conditional expression");
        if (!el) return std::unexpected(el.error());
        Result<Expr*> b = parse_expr();
        if (!b) return std::unexpected(b.error());
        tern->sub = *e;      // true branch
        tern->lower = *c;    // condition
        tern->upper = *b;    // false branch
        return tern;
    }
    return e;
}

Result<Expr*> Parser::parse_testlist(bool allow_tuple) noexcept {
    Result<Expr*> first = parse_expr();
    if (!first) return first;
    if (allow_tuple && check(TokKind::Comma)) {
        return parse_tuple_tail(*first);
    }
    return first;
}

Result<Expr*> Parser::parse_tuple_tail(Expr* first) noexcept {
    Expr* tup = new_expr(ExprKind::TupleLit, first->line);
    tup->args.push_back(first);
    while (accept(TokKind::Comma)) {
        if (stmt_boundaries() || check(TokKind::RParen) || check(TokKind::EqEq) ||
            check(TokKind::Colon) || check(TokKind::RBracket) || check(TokKind::RBrace) ||
            check(TokKind::KwIn)) {
            break;
        }
        Result<Expr*> e = parse_expr();
        if (!e) return std::unexpected(e.error());
        tup->args.push_back(*e);
    }
    return tup;
}

Result<Expr*> Parser::parse_or() noexcept {
    Result<Expr*> e = parse_and();
    if (!e) return e;
    while (check(TokKind::KwOr)) {
        advance();
        Result<Expr*> rhs = parse_and();
        if (!rhs) return std::unexpected(rhs.error());
        Expr* op = new_expr(ExprKind::BoolOp, (*e)->line);
        op->op = bool_or;
        op->args.push_back(*e);
        op->args.push_back(*rhs);
        e = op;
    }
    return e;
}

Result<Expr*> Parser::parse_and() noexcept {
    Result<Expr*> e = parse_not();
    if (!e) return e;
    while (check(TokKind::KwAnd)) {
        advance();
        Result<Expr*> rhs = parse_not();
        if (!rhs) return std::unexpected(rhs.error());
        Expr* op = new_expr(ExprKind::BoolOp, (*e)->line);
        op->op = bool_and;
        op->args.push_back(*e);
        op->args.push_back(*rhs);
        e = op;
    }
    return e;
}

Result<Expr*> Parser::parse_not() noexcept {
    if (check(TokKind::KwNot)) {
        advance();
        Result<Expr*> e = parse_not();
        if (!e) return std::unexpected(e.error());
        Expr* op = new_expr(ExprKind::UnaryOp, (*e)->line);
        op->op = un_not;
        op->sub = *e;
        return op;
    }
    return parse_comparison();
}

Result<Expr*> Parser::parse_comparison() noexcept {
    Result<Expr*> e = parse_bitor();
    if (!e) return e;
    bool any = false;
    Expr* chain = new_expr(ExprKind::Compare, (*e)->line);
    chain->args.push_back(*e);
    for (;;) {
        std::uint16_t op = 0xFFFF;
        switch (peek().kind) {
            case TokKind::Lt: op = static_cast<std::uint16_t>(CmpOpKind::LT); break;
            case TokKind::LtEq: op = static_cast<std::uint16_t>(CmpOpKind::LE); break;
            case TokKind::Gt: op = static_cast<std::uint16_t>(CmpOpKind::GT); break;
            case TokKind::GtEq: op = static_cast<std::uint16_t>(CmpOpKind::GE); break;
            case TokKind::EqEq: op = static_cast<std::uint16_t>(CmpOpKind::EQ); break;
            case TokKind::NotEq: op = static_cast<std::uint16_t>(CmpOpKind::NE); break;
            case TokKind::KwIs:
                op = static_cast<std::uint16_t>(CmpOpKind::Is);
                if (peek(1).kind == TokKind::KwNot) {
                    advance();
                    op = static_cast<std::uint16_t>(CmpOpKind::IsNot);
                }
                break;
            case TokKind::KwIn:
                if (suppress_in_operator_ > 0) break;
                op = static_cast<std::uint16_t>(CmpOpKind::In);
                break;
            case TokKind::KwNot:
                if (peek(1).kind == TokKind::KwIn) {
                    advance();
                    op = static_cast<std::uint16_t>(CmpOpKind::NotIn);
                }
                break;
            default: op = 0xFFFF; break;
        }
        if (op == 0xFFFF) break;
        advance();
        any = true;
        Result<Expr*> rhs = parse_bitor();
        if (!rhs) return std::unexpected(rhs.error());
        chain->args.push_back(*rhs);
        chain->cmp_ops.push_back(op);
    }
    if (!any) return e;
    return chain;
}

Result<Expr*> Parser::parse_bitor() noexcept {
    Result<Expr*> e = parse_bitxor();
    if (!e) return e;
    while (check(TokKind::Pipe)) {
        advance();
        Result<Expr*> rhs = parse_bitxor();
        if (!rhs) return std::unexpected(rhs.error());
        Expr* op = new_expr(ExprKind::BinOp, (*e)->line);
        op->op = static_cast<std::uint16_t>(BinOpKind::BitOr);
        op->args.push_back(*e);
        op->args.push_back(*rhs);
        e = op;
    }
    return e;
}

Result<Expr*> Parser::parse_bitxor() noexcept {
    Result<Expr*> e = parse_bitand();
    if (!e) return e;
    while (check(TokKind::Caret)) {
        advance();
        Result<Expr*> rhs = parse_bitand();
        if (!rhs) return std::unexpected(rhs.error());
        Expr* op = new_expr(ExprKind::BinOp, (*e)->line);
        op->op = static_cast<std::uint16_t>(BinOpKind::BitXor);
        op->args.push_back(*e);
        op->args.push_back(*rhs);
        e = op;
    }
    return e;
}

Result<Expr*> Parser::parse_bitand() noexcept {
    Result<Expr*> e = parse_shift();
    if (!e) return e;
    while (check(TokKind::Amp)) {
        advance();
        Result<Expr*> rhs = parse_shift();
        if (!rhs) return std::unexpected(rhs.error());
        Expr* op = new_expr(ExprKind::BinOp, (*e)->line);
        op->op = static_cast<std::uint16_t>(BinOpKind::BitAnd);
        op->args.push_back(*e);
        op->args.push_back(*rhs);
        e = op;
    }
    return e;
}

Result<Expr*> Parser::parse_shift() noexcept {
    Result<Expr*> e = parse_arith();
    if (!e) return e;
    for (;;) {
        BinOpKind k{};
        if (check(TokKind::Shl)) {
            k = BinOpKind::LShift;
        } else if (check(TokKind::Shr)) {
            k = BinOpKind::RShift;
        } else {
            break;
        }
        advance();
        Result<Expr*> rhs = parse_arith();
        if (!rhs) return std::unexpected(rhs.error());
        Expr* op = new_expr(ExprKind::BinOp, (*e)->line);
        op->op = static_cast<std::uint16_t>(k);
        op->args.push_back(*e);
        op->args.push_back(*rhs);
        e = op;
    }
    return e;
}

Result<Expr*> Parser::parse_arith() noexcept {
    Result<Expr*> e = parse_term();
    if (!e) return e;
    for (;;) {
        BinOpKind k{};
        if (check(TokKind::Plus)) {
            k = BinOpKind::Add;
        } else if (check(TokKind::Minus)) {
            k = BinOpKind::Sub;
        } else {
            break;
        }
        advance();
        Result<Expr*> rhs = parse_term();
        if (!rhs) return std::unexpected(rhs.error());
        Expr* op = new_expr(ExprKind::BinOp, (*e)->line);
        op->op = static_cast<std::uint16_t>(k);
        op->args.push_back(*e);
        op->args.push_back(*rhs);
        e = op;
    }
    return e;
}

Result<Expr*> Parser::parse_term() noexcept {
    Result<Expr*> e = parse_factor();
    if (!e) return e;
    for (;;) {
        BinOpKind k{};
        if (check(TokKind::Star)) {
            k = BinOpKind::Mul;
        } else if (check(TokKind::Slash)) {
            k = BinOpKind::TrueDiv;
        } else if (check(TokKind::DoubleSlash)) {
            k = BinOpKind::FloorDiv;
        } else if (check(TokKind::Percent)) {
            k = BinOpKind::Mod;
        } else if (check(TokKind::At)) {
            k = BinOpKind::MatMul;
        } else {
            break;
        }
        advance();
        Result<Expr*> rhs = parse_factor();
        if (!rhs) return std::unexpected(rhs.error());
        Expr* op = new_expr(ExprKind::BinOp, (*e)->line);
        op->op = static_cast<std::uint16_t>(k);
        op->args.push_back(*e);
        op->args.push_back(*rhs);
        e = op;
    }
    return e;
}

Result<Expr*> Parser::parse_factor() noexcept {
    if (check(TokKind::Minus)) {
        advance();
        Result<Expr*> e = parse_factor();
        if (!e) return std::unexpected(e.error());
        Expr* op = new_expr(ExprKind::UnaryOp, (*e)->line);
        op->op = un_neg;
        op->sub = *e;
        return op;
    }
    if (check(TokKind::Plus)) {
        advance();
        return parse_factor();
    }
    if (check(TokKind::Tilde)) {
        advance();
        Result<Expr*> e = parse_factor();
        if (!e) return std::unexpected(e.error());
        Expr* op = new_expr(ExprKind::UnaryOp, (*e)->line);
        op->op = un_invert;
        op->sub = *e;
        return op;
    }
    return parse_power();
}

Result<Expr*> Parser::parse_power() noexcept {
    Result<Expr*> base = parse_atom_with_trailers();
    if (!base) return base;
    if (check(TokKind::StarStar)) {
        advance();
        Result<Expr*> exp = parse_factor();   // right-assoc
        if (!exp) return std::unexpected(exp.error());
        Expr* op = new_expr(ExprKind::BinOp, (*base)->line);
        op->op = static_cast<std::uint16_t>(BinOpKind::Pow);
        op->args.push_back(*base);
        op->args.push_back(*exp);
        return op;
    }
    return base;
}

Result<Expr*> Parser::parse_atom_with_trailers() noexcept {
    Result<Expr*> atom = parse_atom();
    if (!atom) return atom;
    Expr* e = *atom;
    for (;;) {
        if (check(TokKind::Dot)) {
            advance();
            auto name = expect(TokKind::Ident, "attribute name after '.'");
            if (!name) return std::unexpected(name.error());
            Expr* attr = new_expr(ExprKind::Attribute, e->line);
            attr->sub = e;
            attr->attr = intern_tok(name.value());
            e = attr;
        } else if (check(TokKind::LParen)) {
            Result<Expr*> call = parse_call_args(e);
            if (!call) return std::unexpected(call.error());
            e = *call;
        } else if (check(TokKind::LBracket)) {
            advance();
            // Slice support: [a], [a:b], [a:b:c], [:b], [a:], [:], [::2]
            if (check(TokKind::Colon)) {
                // omitted lower bound
                Expr* slice = new_expr(ExprKind::SliceLit, peek().line);
                slice->sub = new_expr(ExprKind::NoneLit, peek().line);
                advance();   // ':'
                if (!check(TokKind::RBracket) && !check(TokKind::Colon)) {
                    Result<Expr*> upper = parse_testlist(true);
                    if (!upper) return std::unexpected(upper.error());
                    slice->index = *upper;
                }
                if (accept(TokKind::Colon)) {
                    if (!check(TokKind::RBracket)) {
                        Result<Expr*> step = parse_testlist(true);
                        if (!step) return std::unexpected(step.error());
                        slice->lower = *step;
                    }
                }
                auto rb = expect(TokKind::RBracket, "']' closing subscript");
                if (!rb) return std::unexpected(rb.error());
                Expr* sub = new_expr(ExprKind::Subscript, e->line);
                sub->sub = e;
                sub->index = slice;
                e = sub;
                continue;
            }
            Result<Expr*> idx = parse_testlist(true);
            if (!idx) return std::unexpected(idx.error());
            if (check(TokKind::Colon)) {
                Expr* slice = new_expr(ExprKind::SliceLit, (*idx)->line);
                slice->sub = *idx;          // lower (may be NoneLit marker)
                advance();                  // ':'
                if (!check(TokKind::RBracket) && !check(TokKind::Colon)) {
                    Result<Expr*> upper = parse_testlist(true);
                    if (!upper) return std::unexpected(upper.error());
                    slice->index = *upper;
                }
                if (accept(TokKind::Colon)) {
                    if (!check(TokKind::RBracket)) {
                        Result<Expr*> step = parse_testlist(true);
                        if (!step) return std::unexpected(step.error());
                        slice->lower = *step;
                    }
                }
                auto rb = expect(TokKind::RBracket, "']' closing subscript");
                if (!rb) return std::unexpected(rb.error());
                Expr* sub = new_expr(ExprKind::Subscript, e->line);
                sub->sub = e;
                sub->index = slice;
                e = sub;
            } else {
                auto rb = expect(TokKind::RBracket, "']' closing subscript");
                if (!rb) return std::unexpected(rb.error());
                Expr* sub = new_expr(ExprKind::Subscript, e->line);
                sub->sub = e;
                sub->index = *idx;
                e = sub;
            }
        } else {
            break;
        }
    }
    return e;
}

Result<Expr*> Parser::parse_call_args(Expr* callee) noexcept {
    auto lp = expect(TokKind::LParen, "'('");
    if (!lp) return std::unexpected(lp.error());
    Expr* call = new_expr(ExprKind::Call, callee->line);
    call->sub = callee;
    bool first = true;
    bool saw_kw = false;
    while (!check(TokKind::RParen)) {
        if (!first) {
            auto comma = expect(TokKind::Comma, "',' between arguments");
            if (!comma) return std::unexpected(comma.error());
        }
        first = false;
        if (check(TokKind::RParen)) break;
        if (check(TokKind::Star)) {
            advance();
            Result<Expr*> v = parse_expr();
            if (!v) return std::unexpected(v.error());
            Argument a;
            a.value = *v;
            a.keyword = arg_star;
            call->call_args.push_back(a);
            continue;
        }
        if (check(TokKind::StarStar)) {
            advance();
            Result<Expr*> v = parse_expr();
            if (!v) return std::unexpected(v.error());
            Argument a;
            a.value = *v;
            a.keyword = arg_kwargs;
            call->call_args.push_back(a);
            continue;
        }
        if (check(TokKind::Ident) && peek(1).kind == TokKind::Assign) {
            SymbolId key = intern_tok(peek());
            advance();
            advance();
            Result<Expr*> v = parse_expr();
            if (!v) return std::unexpected(v.error());
            Argument a;
            a.value = *v;
            a.keyword = key;
            call->call_args.push_back(a);
            saw_kw = true;
            continue;
        }
        if (saw_kw) {
            return fail_msg("parse: positional argument after keyword argument",
                            diag_code::parse_unexpected_token);
        }
        Result<Expr*> v = parse_expr();
        if (!v) return std::unexpected(v.error());
        if (check(TokKind::KwFor)) {
            // Generator expression as sole unparenthesized argument:
            // sum(x for x in ys) — Python grammar special case.
            advance();
            v = parse_comprehension_tail(*v, true);
            if (!v) return std::unexpected(v.error());
        }
        Argument a;
        a.value = *v;
        call->call_args.push_back(a);
    }
    auto rp = expect(TokKind::RParen, "')' closing call");
    if (!rp) return std::unexpected(rp.error());
    return call;
}

Result<Expr*> Parser::parse_comprehension_tail(Expr* elt, bool genexp) noexcept {
    // Multi-clause comprehensions: [x for a in A for b in B if c ...] —
    // nested loops sharing one element expression. 'for' already consumed.
    stdx::small_vector<Comprehension, 4> clauses;
    for (;;) {
        Comprehension clause;
        ++suppress_in_operator_;
        Result<Expr*> target = parse_testlist(true);
        --suppress_in_operator_;
        if (!target) return std::unexpected(target.error());
        clause.target = *target;
        auto in = expect(TokKind::KwIn, "'in' inside comprehension");
        if (!in) return std::unexpected(in.error());
        Result<Expr*> iter = parse_or();
        if (!iter) return std::unexpected(iter.error());
        clause.iter = *iter;
        // PAR-10 fix: multiple if-clauses per for are accepted
        // (Python: [x for x in xs if c1 if c2] == [x for x in xs if c1 and c2]).
        // We chain them as nested IfExp conds by re-using the same `cond` slot
        // with a synthetic BoolOp(And). Simpler: just take the LAST if as
        // the cond and chain subsequent ifs as nested BoolOp(And) into it.
        if (check(TokKind::KwIf)) {
            advance();
            // PAR-9 fix: use parse_expr (allows lambda in if-cond) instead
            // of parse_or (which stops at lambda keyword).
            Result<Expr*> c = parse_expr();
            if (!c) return std::unexpected(c.error());
            clause.cond = *c;
            while (check(TokKind::KwIf)) {
                advance();
                Result<Expr*> c2 = parse_expr();
                if (!c2) return std::unexpected(c2.error());
                Expr* and_op = new_expr(ExprKind::BoolOp, (*c)->line);
                and_op->op = bool_and;
                and_op->args.push_back(clause.cond);
                and_op->args.push_back(*c2);
                clause.cond = and_op;
            }
        }
        clauses.push_back(clause);
        if (check(TokKind::KwFor)) {
            advance();
            continue;
        }
        break;
    }
    Expr* comp = new_expr(ExprKind::ListComp, elt->line);
    comp->comp = clauses[clauses.size() - 1];
    comp->args.push_back(elt);
    for (std::size_t i = clauses.size() - 1; i-- > 0;) {
        Expr* outer = new_expr(ExprKind::ListComp, elt->line);
        outer->comp = clauses[i];
        outer->args.push_back(comp);
        comp = outer;
    }
    for (Expr* e = comp; e && e->kind == ExprKind::ListComp;) {
        e->comp.is_genexp = genexp;
        e = e->args.empty() ? nullptr : e->args[0];
    }
    if (!genexp) {
        auto rb = expect(TokKind::RBracket, "']' closing comprehension");
        if (!rb) return std::unexpected(rb.error());
    }
    return comp;
}

Result<Expr*> Parser::parse_listcomp_or_list(Expr* first) noexcept {
    if (check(TokKind::KwFor)) {
        advance();
        return parse_comprehension_tail(first, false);
    }
    Expr* list = new_expr(ExprKind::ListLit, first->line);
    list->args.push_back(first);
    while (accept(TokKind::Comma)) {
        if (check(TokKind::RBracket)) break;
        Result<Expr*> e = parse_expr();
        if (!e) return std::unexpected(e.error());
        list->args.push_back(*e);
    }
    auto rb = expect(TokKind::RBracket, "']' closing list");
    if (!rb) return std::unexpected(rb.error());
    return list;
}

Result<Expr*> Parser::parse_dict_or_set() noexcept {
    if (check(TokKind::RBrace)) {
        advance();
        return new_expr(ExprKind::DictLit, peek().line);
    }
    Result<Expr*> first_key = parse_expr();
    if (!first_key) return first_key;
    if (accept(TokKind::Colon)) {
        Expr* dict = new_expr(ExprKind::DictLit, (*first_key)->line);
        Result<Expr*> v = parse_expr();
        if (!v) return std::unexpected(v.error());
        dict->args.push_back(*first_key);
        dict->args.push_back(*v);
        while (accept(TokKind::Comma)) {
            if (check(TokKind::RBrace)) break;
            Result<Expr*> k = parse_expr();
            if (!k) return std::unexpected(k.error());
            auto colon = expect(TokKind::Colon, "':' in dict entry");
            if (!colon) return std::unexpected(colon.error());
            Result<Expr*> val = parse_expr();
            if (!val) return std::unexpected(val.error());
            dict->args.push_back(*k);
            dict->args.push_back(*val);
        }
        auto rb = expect(TokKind::RBrace, "'}' closing dict");
        if (!rb) return std::unexpected(rb.error());
        return dict;
    }
    return fail_msg("parse: set literals are outside the VORTEX subset (use dict or list)",
                    diag_code::parse_unexpected_token);
}

Result<Expr*> Parser::parse_atom() noexcept {
    const Token& t = peek();
    switch (t.kind) {
        case TokKind::IntLit: {
            advance();
            Expr* e = new_expr(ExprKind::IntLit, t.line);
            e->int_value = t.int_value;
            return e;
        }
        case TokKind::FloatLit: {
            advance();
            Expr* e = new_expr(ExprKind::FloatLit, t.line);
            e->float_value = t.float_value;
            return e;
        }
        case TokKind::StrLit: {
            advance();
            Expr* e = new_expr(ExprKind::StrLit, t.line);
            auto [off, len] = module_.pool_string(t.text);
            e->str_offset = off;
            e->str_length = len;
            return e;
        }
        case TokKind::KwTrue:
        case TokKind::KwFalse: {
            bool v = t.kind == TokKind::KwTrue;
            advance();
            Expr* e = new_expr(ExprKind::BoolLit, t.line);
            e->bool_value = v;
            return e;
        }
        case TokKind::KwNone: {
            advance();
            return new_expr(ExprKind::NoneLit, t.line);
        }
        case TokKind::Ident: {
            advance();
            Expr* e = new_expr(ExprKind::Name, t.line);
            e->name = intern_tok(t);
            return e;
        }
        case TokKind::LParen: {
            advance();
            if (check(TokKind::RParen)) {
                advance();
                return new_expr(ExprKind::TupleLit, t.line);   // empty tuple
            }
            Result<Expr*> first = parse_expr();
            if (!first) return std::unexpected(first.error());
            if (check(TokKind::KwFor)) {
                advance();
                Result<Expr*> g = parse_comprehension_tail(*first, true);
                if (!g) return std::unexpected(g.error());
                // PAR-3 fix: a generator expression in parens must consume
                // the closing ')'. parse_comprehension_tail returns without
                // consuming the paren for genexp-in-parens.
                auto rp = expect(TokKind::RParen, "')' closing generator expression");
                if (!rp) return std::unexpected(rp.error());
                return g;
            }
            if (check(TokKind::Comma)) {
                Result<Expr*> tup = parse_tuple_tail(*first);
                if (!tup) return std::unexpected(tup.error());
                auto rp = expect(TokKind::RParen, "')' closing tuple");
                if (!rp) return std::unexpected(rp.error());
                return tup;
            }
            auto rp = expect(TokKind::RParen, "')'");
            if (!rp) return std::unexpected(rp.error());
            return first;   // parenthesized expression (grouping only)
        }
        case TokKind::LBracket: {
            advance();
            if (check(TokKind::RBracket)) {
                advance();
                return new_expr(ExprKind::ListLit, t.line);
            }
            Result<Expr*> first = parse_expr();
            if (!first) return std::unexpected(first.error());
            return parse_listcomp_or_list(*first);
        }
        case TokKind::LBrace: {
            advance();
            return parse_dict_or_set();
        }
        default:
            return fail(err_expected("an expression"));
    }
}

Result<Module*> compile_to_ast(BumpArena& module_arena, std::string_view source) noexcept {
    Module* m = module_arena.create<Module>();
    Lexer lexer(source);
    stdx::small_vector<Token, 512> tokens;
    Result<void> lexed = lexer.run(tokens, m->string_pool);
    if (!lexed) return std::unexpected(lexed.error());
    Parser parser(tokens, *m);
    Result<void> parsed = parser.parse_module();
    if (!parsed) return std::unexpected(parsed.error());
    return m;
}

}  // namespace abi_v1
}  // namespace vortex::fe
