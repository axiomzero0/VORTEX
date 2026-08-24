// =============================================================================
// tests/unit/frontend_test.cpp — lexer / parser / Pass-1 lowering tests.
// =============================================================================

#include "harness.hpp"

#include "vortex/frontend/lexer.hpp"
#include "vortex/frontend/lowering.hpp"
#include "vortex/frontend/parser.hpp"
#include "vortex/ir/printer.hpp"
#include "vortex/ir/verifier.hpp"

#include <cstdio>

using namespace vortex;
using namespace vortex::fe;
using namespace vortex::ir;
using vortex::ir::Node;
using vortex::ir::NodeId;

namespace {

Module* must_parse(BumpArena& arena, std::string_view src) noexcept {
    auto r = compile_to_ast(arena, src);
    if (!r) {
        r.error().report(stderr);
        ++::vortex_test::failures();
        return nullptr;
    }
    return *r;
}

LoweredUnit lower_def(BumpArena& arena, std::string_view src) noexcept {
    Module* m = must_parse(arena, src);
    LoweredUnit empty{};
    if (!m || m->body.empty() || m->body[0]->kind != StmtKind::FunctionDef) {
        std::fprintf(stderr, "  test source must define one function\n");
        ++::vortex_test::failures();
        return empty;
    }
    LowerContext ctx;
    stdx::small_vector<SymbolId, 8> no_captures;
    auto r = lower_unit(*m, ctx, m->body[0], m->body[0]->name, no_captures);
    if (!r) {
        r.error().report(stderr);
        ++::vortex_test::failures();
        return empty;
    }
    return *r;
}

}  // namespace

TEST(lexer_tokens_and_indent) {
    BumpArena arena;
    Module* m = arena.create<Module>();
    Lexer lex("def f(x):\n    return x + 1\n");
    stdx::small_vector<Token, 512> toks;
    Result<void> r = lex.run(toks, m->string_pool);
    CHECK(r.has_value());
    CHECK_EQ(toks[0].kind, TokKind::KwDef);
    CHECK_EQ(toks[1].kind, TokKind::Ident);
    CHECK(toks[1].text == "f");
    CHECK_EQ(toks[2].kind, TokKind::LParen);
    CHECK_EQ(toks[3].kind, TokKind::Ident);
    CHECK_EQ(toks[5].kind, TokKind::Colon);
    CHECK_EQ(toks[6].kind, TokKind::Newline);
    CHECK_EQ(toks[7].kind, TokKind::Indent);
    CHECK_EQ(toks[8].kind, TokKind::KwReturn);
    CHECK_EQ(toks[10].kind, TokKind::Plus);
    CHECK_EQ(toks[11].kind, TokKind::IntLit);
    CHECK_EQ(toks[11].int_value, 1);
    CHECK_EQ(toks[12].kind, TokKind::Newline);
    bool saw_dedent = false;
    for (Token& t : toks) {
        if (t.kind == TokKind::Dedent) saw_dedent = true;
    }
    CHECK(saw_dedent);
}

TEST(lexer_strings_and_numbers) {
    BumpArena arena;
    Module* m = arena.create<Module>();
    Lexer lex("s = \"a\\nb\"\nf = 3.25\nh = 0xFF\n");
    stdx::small_vector<Token, 512> toks;
    Result<void> r = lex.run(toks, m->string_pool);
    CHECK(r.has_value());
    bool found_str = false;
    bool found_hex = false;
    for (Token& t : toks) {
        if (t.kind == TokKind::StrLit) {
            found_str = true;
            CHECK_EQ(t.text.size(), 3u);
            CHECK(t.text[1] == '\n');
        }
        if (t.kind == TokKind::FloatLit) CHECK(t.float_value == 3.25);
        if (t.kind == TokKind::IntLit && t.int_value == 255) found_hex = true;
    }
    CHECK(found_str);
    CHECK(found_hex);
}

TEST(lexer_rejects_tab_indent) {
    BumpArena arena;
    Module* m = arena.create<Module>();
    Lexer lex("if x:\n\treturn 1\n");
    stdx::small_vector<Token, 512> toks;
    Result<void> r = lex.run(toks, m->string_pool);
    CHECK(!r.has_value());
    CHECK_EQ(r.error().code, diag_code::lex_invalid_char);
}

TEST(parser_arith_precedence) {
    BumpArena arena;
    Module* m = must_parse(arena, "def f(a, b, c):\n    return a + b * c - c / b\n");
    if (!m) return;
    Stmt* fn = m->body[0];
    CHECK_EQ(fn->kind, StmtKind::FunctionDef);
    CHECK_EQ(fn->params.size(), 3u);
    Stmt* ret = fn->body[0];
    CHECK_EQ(ret->kind, StmtKind::Return);
    Expr* e = ret->value;
    CHECK_EQ(e->kind, ExprKind::BinOp);
    CHECK_EQ(static_cast<BinOpKind>(e->op), BinOpKind::Sub);
    CHECK_EQ(e->args[0]->kind, ExprKind::BinOp);
    CHECK_EQ(static_cast<BinOpKind>(e->args[0]->op), BinOpKind::Add);
    CHECK_EQ(e->args[1]->kind, ExprKind::BinOp);
    CHECK_EQ(static_cast<BinOpKind>(e->args[1]->op), BinOpKind::TrueDiv);
    CHECK_EQ(static_cast<BinOpKind>(e->args[0]->args[1]->op), BinOpKind::Mul);
}

TEST(parser_control_flow_shapes) {
    BumpArena arena;
    Module* m = must_parse(
        arena,
        "def f(n):\n"
        "    if n > 0:\n"
        "        return 1\n"
        "    elif n < 0:\n"
        "        return -1\n"
        "    else:\n"
        "        return 0\n"
        "    while n:\n"
        "        n = n - 1\n"
        "        if n == 2:\n"
        "            break\n"
        "        continue\n"
        "    for i in range(n):\n"
        "        pass\n"
        "    return 0\n");
    if (!m) return;
    Stmt* fn = m->body[0];
    CHECK_EQ(fn->body.size(), 4u);
    Stmt* iff = fn->body[0];
    CHECK_EQ(iff->kind, StmtKind::If);
    CHECK_EQ(iff->body.size(), 1u);
    CHECK_EQ(iff->orelse.size(), 1u);   // elif nested in orelse
    Stmt* wh = fn->body[1];
    CHECK_EQ(wh->kind, StmtKind::While);
    CHECK_EQ(wh->body[0]->kind, StmtKind::Assign);
    CHECK_EQ(wh->body[1]->kind, StmtKind::If);
    CHECK_EQ(wh->body[1]->body[0]->kind, StmtKind::Break);
    CHECK_EQ(wh->body[2]->kind, StmtKind::Continue);
    Stmt* fr = fn->body[2];
    CHECK_EQ(fr->kind, StmtKind::For);
    CHECK_EQ(fr->body[0]->kind, StmtKind::Pass);
}

TEST(parser_try_except_finally) {
    BumpArena arena;
    Module* m = must_parse(
        arena,
        "def f(x):\n"
        "    try:\n"
        "        return x / 1\n"
        "    except ValueError as e:\n"
        "        return e\n"
        "    except Exception:\n"
        "        return 2\n"
        "    finally:\n"
        "        pass\n");
    if (!m) return;
    Stmt* fn = m->body[0];
    Stmt* tr = fn->body[0];
    CHECK_EQ(tr->kind, StmtKind::Try);
    CHECK_EQ(tr->handlers.size(), 2u);
    CHECK(tr->handlers[0].type_name != 0xFFFF'FFFF);
    CHECK(tr->handlers[0].bind_name != 0xFFFF'FFFF);
    CHECK(tr->handlers[1].type_name != 0xFFFF'FFFF);   // 'except Exception:' names its type
    CHECK(tr->handlers[1].bind_name == 0xFFFF'FFFF);
    CHECK_EQ(tr->finalbody.size(), 1u);
}

TEST(parser_listcomp_and_calls) {
    BumpArena arena;
    Module* m = must_parse(
        arena,
        "def f(xs, k):\n"
        "    ys = [x * k for x in xs if x > 0]\n"
        "    g(ys, key=1, *rest, **kw)\n"
        "    return sum(x for x in ys)\n");
    if (!m) return;
    Stmt* fn = m->body[0];
    Stmt* assign = fn->body[0];
    CHECK_EQ(assign->kind, StmtKind::Assign);
    Expr* comp = assign->value;
    CHECK_EQ(comp->kind, ExprKind::ListComp);
    CHECK_EQ(comp->comp.target->kind, ExprKind::Name);
    CHECK(comp->comp.cond != nullptr);
    Stmt* call = fn->body[1];
    CHECK_EQ(call->kind, StmtKind::Expr);
    CHECK_EQ(call->value->kind, ExprKind::Call);
    CHECK_EQ(call->value->call_args.size(), 4u);
    int kw = 0, star = 0, kwstar = 0, pos = 0;
    for (Argument& a : call->value->call_args) {
        if (a.keyword == arg_invalid) ++pos;
        else if (a.keyword == arg_star) ++star;
        else if (a.keyword == arg_kwargs) ++kwstar;
        else ++kw;
    }
    CHECK_EQ(pos, 1);
    CHECK_EQ(kw, 1);
    CHECK_EQ(star, 1);
    CHECK_EQ(kwstar, 1);
    Expr* sumcall = fn->body[2]->value;
    CHECK_EQ(sumcall->kind, ExprKind::Call);
    Expr* gen = sumcall->call_args[0].value;
    CHECK_EQ(gen->kind, ExprKind::ListComp);
    CHECK(gen->comp.is_genexp);
}

TEST(lowering_smoke_fib) {
    BumpArena arena;
    LoweredUnit unit = lower_def(
        arena,
        "def fib(n):\n"
        "    if n < 2:\n"
        "        return n\n"
        "    return fib(n - 1) + fib(n - 2)\n");
    Graph& g = unit.graph;
    CHECK_EQ(g.n_parameters, 1u);
    CHECK(verify_or_report(g, "lower-fib"));
    bool saw_compare = false, saw_if = false, saw_call = false, saw_ret = false;
    g.for_each_live([&](NodeId id) {
        Node& n = g.node(id);
        if (n.kind == NodeKind::PyCompare) saw_compare = true;
        if (n.kind == NodeKind::If) saw_if = true;
        if (n.kind == NodeKind::CallPy) saw_call = true;
        if (n.kind == NodeKind::Return) saw_ret = true;
    });
    CHECK(saw_compare && saw_if && saw_call && saw_ret);
    stdx::small_vector<char, 4096> text;
    CHECK(graph_to_string(g, text));
}

TEST(lowering_loop_phis) {
    BumpArena arena;
    LoweredUnit unit = lower_def(
        arena,
        "def total(n):\n"
        "    s = 0\n"
        "    i = 0\n"
        "    while i < n:\n"
        "        s = s + i\n"
        "        i = i + 1\n"
        "    return s\n");
    Graph& g = unit.graph;
    CHECK(verify_or_report(g, "lower-total"));
    bool saw_loop = false, saw_effect_phi = false;
    std::uint32_t phi_count = 0;
    g.for_each_live([&](NodeId id) {
        Node& n = g.node(id);
        if (n.kind == NodeKind::Loop) saw_loop = true;
        if (n.kind == NodeKind::EffectPhi) saw_effect_phi = true;
        if (n.kind == NodeKind::Phi) ++phi_count;
    });
    CHECK(saw_loop);
    CHECK(saw_effect_phi);
    CHECK(phi_count >= 2u);
}

TEST(lowering_closures_cells) {
    BumpArena arena;
    Module* m = must_parse(
        arena,
        "def outer(n):\n"
        "    def inner(x):\n"
        "        return x + n\n"
        "    return inner(1) + inner(2)\n");
    if (!m) return;
    LowerContext ctx;
    stdx::small_vector<SymbolId, 8> no_captures;
    auto unit = lower_unit(*m, ctx, m->body[0], m->body[0]->name, no_captures);
    CHECK(unit.has_value());
    if (unit) CHECK_EQ(unit->children.size(), 1u);
}

TEST(lowering_class_def) {
    BumpArena arena;
    Module* m = must_parse(
        arena,
        "class Point:\n"
        "    def __init__(self, x, y):\n"
        "        self.x = x\n"
        "        self.y = y\n"
        "\n"
        "def use():\n"
        "    p = Point(1, 2)\n"
        "    return p.x + p.y\n");
    if (!m) return;
    LowerContext ctx;
    stdx::small_vector<SymbolId, 8> no_captures;
    auto unit = lower_unit(*m, ctx, nullptr, 0, no_captures);
    CHECK(unit.has_value());
    if (unit) {
        CHECK_EQ(unit->children.size(), 2u);   // Point body + use (__init__ nests under Point)
        CHECK(verify_or_report(unit->graph, "module"));
    }
}

TEST(lowering_try_and_generators) {
    BumpArena arena;
    LoweredUnit gen = lower_def(
        arena,
        "def g(n):\n"
        "    for i in range(n):\n"
        "        yield i * 2\n");
    CHECK(gen.is_generator);
    CHECK(verify_or_report(gen.graph, "gen"));
}
