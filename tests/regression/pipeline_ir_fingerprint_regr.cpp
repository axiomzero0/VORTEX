// =============================================================================
// tests/regression/pipeline_ir_fingerprint_regr.cpp — pipeline stability guard.
//
// Catches accidental reordering or behavior changes in the pass pipeline by
// snapshotting the IR fingerprint (node count by kind + edge count) after
// the full pipeline on each lang_case, and asserting it matches the
// known-stable values in `golden_fingerprints.txt`.
//
// A fingerprint change is NOT automatically a bug — it's a flag for review.
// When a pass is deliberately changed (e.g., a new transform fires), the
// maintainer updates the golden file via `vortex_regression_tests --update-golden`
// after verifying the change is correct. The test guards against UNINTENDED
// drift: a commit that changes a fingerprint without updating the golden file
// fails CI until either the pass is fixed or the golden file is updated.
//
// The fingerprint is structural (no NodeIds), so it's stable across frontend
// re-numberings — what we're pinning is "this many Loops, this many Calls,
// this many edges" for each program. Any drift means SOMETHING in the
// pipeline transformed differently than before.
// =============================================================================

#include "regression_harness.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

#include "../lang/lang_cases.hpp"

using namespace vortex;
using namespace vortex::ir;
namespace passes = vortex::passes;

namespace {

// Path to the golden file. Resolved relative to the binary's CWD; the ctest
// invocation `cd /home/z/my-project/VORTEX && ctest` runs the binary in
// the repo root, so this works. Override with $VORTEX_GOLDEN_FP for
// custom CI layouts.
[[nodiscard]] std::string golden_path() {
    if (const char* p = std::getenv("VORTEX_GOLDEN_FP")) return std::string(p);
    return "tests/regression/golden_fingerprints.txt";
}

// Load golden fingerprints. Format: "<case_name> = <fingerprint>".
// Returns an empty map if the file is missing (first-run / capture mode).
[[nodiscard]] std::unordered_map<std::string, std::string> load_golden() {
    std::unordered_map<std::string, std::string> m;
    std::ifstream in(golden_path());
    if (!in.is_open()) return m;
    std::string line;
    while (std::getline(in, line)) {
        // Skip blank lines and comments.
        if (line.empty() || line[0] == '#') continue;
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string name = line.substr(0, eq);
        std::string fp = line.substr(eq + 1);
        // Trim spaces.
        while (!name.empty() && std::isspace(static_cast<unsigned char>(name.back()))) name.pop_back();
        while (!name.empty() && std::isspace(static_cast<unsigned char>(name.front()))) name.erase(name.begin());
        while (!fp.empty() && std::isspace(static_cast<unsigned char>(fp.back()))) fp.pop_back();
        while (!fp.empty() && std::isspace(static_cast<unsigned char>(fp.front()))) fp.erase(fp.begin());
        m[name] = fp;
    }
    return m;
}

// Write the entire captured set back as the new golden file. Used by the
// `--update-golden` CLI flag below.
void write_golden(const std::unordered_map<std::string, std::string>& m) {
    std::ofstream out(golden_path(), std::ios::trunc);
    if (!out.is_open()) {
        std::fprintf(stderr, "  [fingerprint] FAILED to write golden file: %s\n",
                     golden_path().c_str());
        return;
    }
    out << "# VORTEX pipeline IR fingerprint baseline.\n";
    out << "# Format: <case_name> = <fingerprint>\n";
    out << "# Drift = CI failure. Update via: vortex_regression_tests --update-golden\n";
    out << "# after verifying the pass change that caused the drift is correct.\n\n";
    for (const auto& [name, fp] : m) {
        out << name << " = " << fp << '\n';
    }
}

// CLI flag check: --update-golden regenerates the golden file from the
// current pipeline output (used after a deliberate pass change).
[[nodiscard]] bool update_golden_requested() {
    const char* arg = std::getenv("VORTEX_UPDATE_GOLDEN");
    return arg != nullptr && arg[0] == '1';
}

}  // namespace

// =============================================================================
// Capture the fingerprint of each function-shaped lang_case after running
// the full pipeline. Compare against the golden file.
//
// Modes:
//   - default: missing golden file → log every fingerprint, no failure;
//              mismatched golden entry → failure.
//   - VORTEX_UPDATE_GOLDEN=1: rewrite the golden file from the current
//              pipeline output. Always passes; the file IS the new baseline.
// =============================================================================
TEST(regr_pipeline_ir_fingerprint_stability) {
    auto golden = load_golden();
    const bool update_mode = update_golden_requested();

    int drift = 0;
    int ran = 0;
    int new_entries = 0;
    std::unordered_map<std::string, std::string> captured;

    for (std::size_t i = 0; i < kLangCaseCount; ++i) {
        const LangCase& c = kLangCases[i];
        bool ok = false;
        Graph g = vortex_test::lower_function(c.src, &ok);
        if (!ok) continue;   // module-level only — skip.

        bool ran_ok = vortex_test::run_full_pipeline(g, passes::TierMode::Tier2);
        if (!ran_ok) {
            std::fprintf(stderr, "  [fingerprint] %s: pipeline raised a diagnostic\n",
                         c.name);
            ++drift;
            continue;
        }
        ++ran;
        std::string fp = vortex_test::fingerprint_ir(g);
        captured[c.name] = fp;

        if (update_mode) {
            // Just collect; we'll write below.
            continue;
        }

        auto it = golden.find(c.name);
        if (it == golden.end()) {
            // No baseline yet. Print it so the maintainer can capture.
            std::fprintf(stderr, "  [fingerprint] %s (new): %s\n", c.name, fp.c_str());
            ++new_entries;
            continue;
        }
        if (it->second != fp) {
            std::fprintf(stderr,
                         "  [fingerprint] %s DRIFT:\n    expected: %s\n    got:      %s\n",
                         c.name, it->second.c_str(), fp.c_str());
            ++drift;
        }
    }

    if (update_mode) {
        write_golden(captured);
        std::printf("  golden fingerprints updated (%zu entries, %d ran)\n",
                    captured.size(), ran);
        return;
    }

    CHECK(ran > 0);
    // New entries don't fail the test — they're awaiting stabilization.
    // Only DRIFT (golden present but mismatched) fails. New entries are
    // added to the golden file by re-running with VORTEX_UPDATE_GOLDEN=1.
    CHECK_EQ(drift, 0);
    if (new_entries > 0) {
        std::printf("  %d new fingerprints awaiting stabilization "
                    "(re-run with VORTEX_UPDATE_GOLDEN=1 to commit them)\n",
                    new_entries);
    }
}
