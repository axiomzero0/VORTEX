// =============================================================================
// tests/regression/opt_in_toggle_regr.cpp — opt-out flag differential.
//
// Rule 23: default-on optimizations (polyhedral today, more in the future)
// must not change observable program results when toggled. They MAY change
// IR shape (that's the point — they transform loops), but the program's
// observable output must be byte-identical.
//
// The bug this guards: a future default-on pass that fires when it shouldn't
// (e.g., the gate becomes inverted) and produces wrong results — the bug
// would only manifest in the DEFAULT path (which is most programs), and
// would slip past the unit tests because the unit tests don't run the
// full corpus.
//
// We run the ENTIRE lang_cases corpus twice per case — once with the
// opt-out flag set (polyhedral OFF), once without (polyhedral ON, the
// default) — and assert the stdout is identical.
// =============================================================================

#include "regression_harness.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include "../lang/lang_cases.hpp"

using namespace vortex;
namespace rt = vortex::rt;

namespace {

// Run `fn` in a forked child process so that a hard abort (SIGSEGV/SIGABRT
// from a runtime FATAL) in the polyhedral-on path doesn't kill the test
// runner. The parent waits for the child, classifies the outcome, and
// returns:
//   - "ok" + the captured stdout if the child exited 0
//   - "crash:<sig>" if the child was killed by a signal
//   - "exit:<code>" if the child exited with non-zero
[[nodiscard]] std::string run_in_child(const char* src, const rt::CompileOptions& opts,
                                        bool* ok) {
    *ok = false;
    // Pipe to capture the child's stdout.
    int pipefd[2];
    if (pipe(pipefd) != 0) return "ERROR: pipe failed";
    pid_t pid = fork();
    if (pid < 0) {
        close(pipefd[0]); close(pipefd[1]);
        return "ERROR: fork failed";
    }
    if (pid == 0) {
        // Child: redirect stdout to the pipe's write end, run, exit.
        close(pipefd[0]);
        dup2(pipefd[1], fileno(stdout));
        close(pipefd[1]);
        rt::Vm vm;
        rt::set_vm_for_builtins(&vm);
        rt::install_builtins(vm.program);
        Result<Value> r = rt::run_source(vm, src, opts);
        std::fflush(stdout);
        _exit(r.has_value() ? 0 : 1);
    }
    // Parent: read everything the child wrote, wait for it.
    close(pipefd[1]);
    std::string out;
    char buf[4096];
    while (true) {
        ssize_t n = read(pipefd[0], buf, sizeof(buf));
        if (n <= 0) break;
        out.append(buf, static_cast<std::size_t>(n));
    }
    close(pipefd[0]);
    int status = 0;
    waitpid(pid, &status, 0);

    if (WIFEXITED(status)) {
        int code = WEXITSTATUS(status);
        if (code == 0) { *ok = true; return out; }
        *ok = false;
        char buf2[32];
        std::snprintf(buf2, sizeof(buf2), "exit:%d", code);
        return std::string(buf2);
    }
    if (WIFSIGNALED(status)) {
        *ok = false;
        char buf2[32];
        std::snprintf(buf2, sizeof(buf2), "crash:%d", WTERMSIG(status));
        return std::string(buf2);
    }
    *ok = false;
    return "unknown";
}

}  // namespace

// =============================================================================
// Toggle polyhedral default-on vs opt-out across the full corpus. Output
// must match.
//
// The polyhedral-default-on path runs in a forked child so that if a future
// change introduces a regression (e.g., a crash on a scalar nested loop),
// the regression surfaces as a per-case drift report rather than killing
// the test runner. The child crashing IS the regression — opting IN to
// the default path produced no valid output, which is a behavioral change.
// =============================================================================
TEST(regr_opt_out_toggle_polyhedral_preserves_corpus_output) {
    int drift = 0;
    int ran = 0;
    int crashes = 0;
    for (std::size_t i = 0; i < kLangCaseCount; ++i) {
        const LangCase& c = kLangCases[i];

        // Default: polyhedral ON. Opt-out: polyhedral OFF.
        rt::CompileOptions default_on;
        rt::CompileOptions opt_out;
        opt_out.disable_polyhedral = true;

        bool ok_off = false, ok_on = false;
        // Run the DEFAULT-ON path in a child because it's the more
        // aggressive one — a crash here is the regression, not the runner
        // dying.
        std::string out_on  = run_in_child(c.src, default_on, &ok_on);
        // Run the opt-out path in-process (it's the conservative subset).
        std::string out_off = vortex_test::capture_stdout(c.src, &ok_off, opt_out);

        ++ran;
        if (!ok_off) {
            std::fprintf(stderr, "  [opt-out] %s: polyhedral-OFF compilation failed\n",
                         c.name);
            ++drift;
            continue;
        }
        if (!ok_on) {
            // The polyhedral-ON path crashed (SIGABRT from a runtime FATAL).
            // That IS the regression: the default path changed observable
            // behavior (the program produced no valid output at all).
            // Report and continue to the next case so the suite keeps going.
            std::fprintf(stderr,
                         "  [opt-out] %s: polyhedral-ON path crashed (%s) — "
                         "the default path must not crash\n",
                         c.name, out_on.c_str());
            ++crashes;
            ++drift;
            continue;
        }
        if (out_off != out_on) {
            std::fprintf(stderr,
                         "  [opt-out] %s: polyhedral toggle changed output\n"
                         "    off: %s\n"
                         "    on:  %s\n",
                         c.name, out_off.c_str(), out_on.c_str());
            ++drift;
            continue;
        }
        if (out_off != c.expect) {
            // The corpus case itself is failing — that's a separate
            // regression (caught by lang_full_stack too). Don't double-
            // count here.
            std::fprintf(stderr,
                         "  [opt-out] %s: case expectation already failing "
                         "(see lang_full_stack)\n", c.name);
        }
    }
    CHECK(ran > 0);
    // The default-on path must produce byte-identical output to the
    // opt-out path on every corpus case — that's the contract. Any drift
    // is a correctness regression.
    if (drift > 0) {
        std::fprintf(stderr,
                     "  [opt-out] %d drift(s) detected (%d crash(es)). "
                     "Polyhedral default-on must preserve corpus output.\n",
                     drift, crashes);
    }
    CHECK_EQ(drift, 0);
}

// =============================================================================
// Pin the default state: polyhedral must be ON in a default-constructed
// CompileOptions (disable_polyhedral must be false). A future change that
// flips the default would silently disable the optimization on every
// program — exactly what Rule 28 forbids (no optimization without measurable
// win; polyhedral preserves cache locality on nested loops).
// =============================================================================
TEST(regr_opt_out_default_compile_options_polyhedral_on) {
    rt::CompileOptions defaults;
    CHECK(!defaults.disable_polyhedral);
}

// =============================================================================
// Pin the PassContext default: OptOption::DisablePolyhedral must be UNSET
// in a default-constructed PassContext (polyhedral is ON by default).
// =============================================================================
TEST(regr_opt_out_default_pass_context_polyhedral_on) {
    passes::PassContext defaults;
    CHECK(!defaults.options.has(passes::OptOption::DisablePolyhedral));
}

// =============================================================================
// Pin the TierFilter gate: polyhedral must be INCLUDED in every tier
// (except Tier 1, where the budget gate excludes it) when the opt-out flag
// is unset, and EXCLUDED in every tier when the opt-out flag is set.
// =============================================================================
TEST(regr_opt_out_tier_filter_gates_polyhedral_symmetrically) {
    for (passes::TierMode tier : {passes::TierMode::Tier1,
                                  passes::TierMode::Tier2,
                                  passes::TierMode::Tier3}) {
        // Default: polyhedral ON (no opt-out flag set).
        passes::PassContext without;
        without.tier = tier;
        passes::TierFilter f_without{tier};
        if (tier == passes::TierMode::Tier1) {
            // Budget gate excludes polyhedral from Tier 1 even when
            // default-on — that's correct: Tier 1 is the budget-constrained
            // baseline JIT; polyhedral's fixpoint-heavy analysis is too
            // expensive for the baseline budget.
            CHECK(!f_without.include("33_polyhedral", without));
        } else {
            CHECK(f_without.include("33_polyhedral", without));
        }

        // Opt-out: polyhedral OFF in every tier.
        passes::PassContext with;
        with.tier = tier;
        with.options.set(passes::OptOption::DisablePolyhedral);
        passes::TierFilter f_with{tier};
        CHECK(!f_with.include("33_polyhedral", with));
    }
}
