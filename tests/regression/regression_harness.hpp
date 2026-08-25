// =============================================================================
// tests/regression/regression_harness.hpp — shared regression-test utilities.
//
// These helpers exist for one reason: to catch the kind of regression that
// slipped through last time (IBE-18 — a "fix" to one MOVri path clobbered the
// Return terminator's output via a shared home slot, breaking JIT results
// silently because no test ran the full Tier-0-vs-JIT differential on a
// corpus). Each helper below is a *guardrail* that the unit tests do not
// provide:
//
//   - capture_stdout    : run a source program with stdout redirected.
//   - lower_function    : lower a single function from source to its IR Graph.
//   - run_full_pipeline  : run Pass 3..51 over a Graph in a fresh context.
//   - fingerprint_ir    : a deterministic textual signature of a Graph's shape
//                          (node count by kind + edge count + max NodeId),
//                          used to detect accidental pipeline reordering.
//   - assert_idempotent : run a pass twice; the second run MUST report no
//                          change (Rule 10).
//   - install_verifier   : install g_verify_after_each_pass to flag every
//                          post-pass IR that fails the structural verifier
//                          (Rule 40), and return the failure log so the
//                          caller can CHECK it stays empty.
//
// Everything is header-only and uses only the project's own support headers,
// so adding a new regression file is a one-line include.
// =============================================================================

#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

#include "vortex/frontend/lowering.hpp"
#include "vortex/frontend/parser.hpp"
#include "vortex/ir/graph.hpp"
#include "vortex/ir/verifier.hpp"
#include "vortex/passes/pass_pipeline.hpp"
#include "vortex/passes/passes_fwd.hpp"
#include "vortex/rt/driver.hpp"
#include "vortex/rt/interp.hpp"
#include "vortex/support/arena.hpp"
#include "vortex/support/flags.hpp"
#include "vortex/support/symbol_table.hpp"

#include "harness.hpp"   // bring in TEST / CHECK / CHECK_EQ

namespace vortex_test {

using namespace vortex;
using namespace vortex::ir;
namespace fe = vortex::fe;
namespace passes = vortex::passes;
namespace rt = vortex::rt;

// ---------------------------------------------------------------------------
// capture_stdout — run `src` through the full stack and return its stdout.
// Returns the captured text; sets *ok to whether compilation succeeded.
// Mirrors tests/unit/lang_test.cpp's helper but lives here so every
// regression file can use it without redefining it.
// ---------------------------------------------------------------------------
[[nodiscard]] inline std::string capture_stdout(const char* src, bool* ok,
                                                 const rt::CompileOptions& opts = {}) {
    rt::Vm vm;
    rt::set_vm_for_builtins(&vm);
    rt::install_builtins(vm.program);
    std::fflush(stdout);
    int saved = dup(fileno(stdout));
    FILE* cap = tmpfile();
    if (!cap) [[unlikely]] {
        dup2(saved, fileno(stdout));
        close(saved);
        *ok = false;
        return "ERROR: tmpfile() failed";
    }
    dup2(fileno(cap), fileno(stdout));
    Result<Value> r = rt::run_source(vm, src, opts);
    std::fflush(stdout);
    dup2(saved, fileno(stdout));
    close(saved);
    long size = ftell(cap);
    if (size < 0) [[unlikely]] {
        if (!r) {
            std::string msg(r.error().message.data(), r.error().message.size());
            fclose(cap);
            *ok = false;
            return "ERROR: " + msg;
        }
        fclose(cap);
        *ok = r.has_value();
        return "ERROR: ftell failed";
    }
    rewind(cap);
    std::string got(static_cast<std::size_t>(size), ' ');
    if (size > 0) fread(got.data(), 1, static_cast<std::size_t>(size), cap);
    fclose(cap);
    *ok = r.has_value();
    if (!*ok && !r) {
        std::string msg(r.error().message.data(), r.error().message.size());
        return "ERROR: " + msg;
    }
    return got;
}

// ---------------------------------------------------------------------------
// lower_function — extract the FIRST nested function from `src` as a Graph.
//
// Used by per-pass regression tests that want a real loop / branch / call
// graph without hand-rolling NodeIds. The source must define at least one
// top-level def; the helper returns its lowered Sea-of-Nodes IR.
// ---------------------------------------------------------------------------
[[nodiscard]] inline Graph lower_function(std::string_view src, bool* ok) {
    *ok = false;
    Graph empty;
    vortex::BumpArena arena;
    Result<fe::Module*> ast = fe::compile_to_ast(arena, src);
    if (!ast) return empty;
    fe::LowerContext lctx;
    stdx::small_vector<SymbolId, 8> no_caps;
    SymbolId mod = global_symbols().intern("regr_mod");
    Result<fe::LoweredUnit> top =
        fe::lower_unit(**ast, lctx, nullptr, mod, no_caps, false, 0xFFFFFFFF);
    if (!top || (*top).children.empty()) return empty;
    fe::PendingFunction& f = (*top).children[0];
    Result<fe::LoweredUnit> fu = fe::lower_unit(**ast, lctx, f.def, f.name,
                                                f.captures, false, f.code_unit_hint);
    if (!fu) return empty;
    *ok = true;
    return (*fu).graph;
}

// ---------------------------------------------------------------------------
// run_full_pipeline — apply P03..P51 over `g` with a Tier2 context.
// Returns true if no diagnostic was raised. The pipeline's own telemetry
// is consulted for budget breaches (Rule 26).
// ---------------------------------------------------------------------------
[[nodiscard]] inline bool run_full_pipeline(Graph& g, passes::TierMode tier,
                                             const Flags<passes::OptOption>& opts = {}) {
    passes::PassContext ctx;
    ctx.tier = tier;
    ctx.options = opts;
    passes::OptPipeline pipeline;
    Result<void> r = passes::run_pipeline(g, ctx, pipeline);
    return r.has_value();
}

// ---------------------------------------------------------------------------
// fingerprint_ir — a deterministic textual signature of a Graph's shape.
//
// We hash the *kind histogram* and total edge count rather than NodeIds:
// NodeIds are not stable across recompilations of the frontend (each pass
// can renumber), but the shape (how many Loops, how many Add, total edges)
// is invariant under correct optimization. A change in the fingerprint
// between commits flags a pass that changed behavior — which then needs
// deliberate review.
//
// Output: "<live_nodes>:<edges>:<loops>:<returns>:<calls>:<allocs>"
// ---------------------------------------------------------------------------
[[nodiscard]] inline std::string fingerprint_ir(const Graph& g) noexcept {
    std::uint32_t live = 0, edges = 0;
    std::uint32_t loops = 0, returns = 0, calls = 0, allocs = 0;
    std::uint32_t ifs = 0, regions = 0, params = 0, consts = 0;
    std::uint32_t adds = 0, loads = 0, stores = 0;
    g.for_each_live([&](NodeId id) {
        const Node& n = g.node(id);
        ++live;
        edges += static_cast<std::uint32_t>(n.ins.size());
        switch (n.kind) {
            case NodeKind::Loop:        ++loops; break;
            case NodeKind::Return:      ++returns; break;
            case NodeKind::CallPy:      ++calls; break;
            case NodeKind::NewList:     ++allocs; break;
            case NodeKind::NewDict:     ++allocs; break;
            case NodeKind::NewTuple:    ++allocs; break;
            case NodeKind::If:          ++ifs; break;
            case NodeKind::Region:      ++regions; break;
            case NodeKind::Parameter:   ++params; break;
            case NodeKind::ConstInt:    ++consts; break;
            case NodeKind::ConstPy:     ++consts; break;
            case NodeKind::Add:         ++adds; break;
            case NodeKind::LoadIndex:   ++loads; break;
            case NodeKind::StoreIndex:  ++stores; break;
            default: break;
        }
    });
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "live=%u edges=%u loops=%u rets=%u calls=%u allocs=%u "
                  "ifs=%u regions=%u params=%u consts=%u adds=%u loads=%u stores=%u",
                  live, edges, loops, returns, calls, allocs,
                  ifs, regions, params, consts, adds, loads, stores);
    return std::string(buf);
}

// ---------------------------------------------------------------------------
// Verifier hook (Rule 40) — install a global verifier that records every
// post-pass failure instead of aborting, so the regression can report the
// list at the end. Returns the captured-failures vector by reference so
// the caller can CHECK it's empty.
// ---------------------------------------------------------------------------
struct VerifierFailures {
    std::vector<std::string> msgs{};
};

// Each test file owns its own VerifierFailures; this single shared pointer
// is set per-test so the C-linkage verifier hook can find it.
inline VerifierFailures* g_active_failures = nullptr;

inline bool verifier_record_failures(const Graph& g, const char* pass_name) noexcept {
    if (!g_active_failures) return true;   // no observer: assume OK
    stdx::small_vector<Diagnostic, 4> diags = verify_graph(g);
    bool ok = true;
    for (const Diagnostic& d : diags) {
        if (!d.is_error()) continue;
        ok = false;
        char buf[256];
        std::snprintf(buf, sizeof(buf), "%s: %.*s", pass_name,
                       static_cast<int>(d.message.size()), d.message.data());
        g_active_failures->msgs.push_back(std::string(buf));
    }
    return ok;
}

struct VerifierScope {
    VerifierFailures fails;
    VerifierFailures* prev;
    bool (*prev_hook)(const Graph&, const char*) = nullptr;
    VerifierScope() {
        prev = g_active_failures;
        g_active_failures = &fails;
        prev_hook = passes::g_verify_after_each_pass;
        passes::g_verify_after_each_pass = verifier_record_failures;
    }
    ~VerifierScope() {
        passes::g_verify_after_each_pass = prev_hook;
        g_active_failures = prev;
    }
    [[nodiscard]] bool empty() const noexcept { return fails.msgs.empty(); }
    [[nodiscard]] std::size_t size() const noexcept { return fails.msgs.size(); }
};

// ---------------------------------------------------------------------------
// assert_idempotent — Rule 10: a second run of the pass must be a no-op.
// `pass` is any pass type with `.run(Graph&, const PassContext&)`.
// Returns true if the second run reports `changed == false`.
// ---------------------------------------------------------------------------
template <typename P>
[[nodiscard]] inline bool second_run_is_noop(P& pass, Graph& g,
                                              const passes::PassContext& ctx) {
    Result<passes::PassResult> r2 = pass.run(g, ctx);
    if (!r2) return false;
    return !r2->changed;
}

}  // namespace vortex_test
