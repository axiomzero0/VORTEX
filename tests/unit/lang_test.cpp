// =============================================================================
// tests/unit/lang_test.cpp — end-to-end language tests over the full stack
// (frontend -> passes -> scheduler -> Tier-0 VM), with stdout capture.
// =============================================================================

#include "harness.hpp"

#include <cstdio>
#include <string>
#include <unistd.h>

#include "vortex/rt/driver.hpp"
#include "vortex/rt/interp.hpp"
#include "../lang/lang_cases.hpp"

using namespace vortex;
using namespace vortex::rt;

namespace {

std::string capture_stdout(const char* src, bool* ok) {
    Vm vm;
    set_vm_for_builtins(&vm);
    install_builtins(vm.program);
    std::fflush(stdout);
    int saved = dup(fileno(stdout));
    FILE* cap = tmpfile();
    dup2(fileno(cap), fileno(stdout));
    Result<Value> r = run_source(vm, src);
    std::fflush(stdout);
    dup2(saved, fileno(stdout));
    close(saved);
    long size = ftell(cap);
    rewind(cap);
    std::string got(size, ' ');
    if (size > 0) fread(got.data(), 1, static_cast<size_t>(size), cap);
    fclose(cap);
    *ok = r.has_value();
    if (!*ok) {
        std::string msg(r.error().message.data(), r.error().message.size());
        return "ERROR: " + msg;
    }
    return got;
}

}  // namespace

TEST(lang_full_stack) {
    int pass_count = 0;
    int fail_count = 0;
    for (std::size_t i = 0; i < kLangCaseCount; ++i) {
        const LangCase& c = kLangCases[i];
        std::fprintf(stderr, "[lang] running %s\n", c.name);
        bool ok = false;
        std::string got = capture_stdout(c.src, &ok);
        if (!ok || got != c.expect) {
            ++fail_count;
            std::printf("[FAIL] %s%s\n  expect: %s\n  got: %s\n", c.name,
                        ok ? "" : " (error)", c.expect, got.c_str());
            ++::vortex_test::failures();
        } else {
            ++pass_count;
        }
    }
    std::printf("  lang: %d passed, %d failed\n", pass_count, fail_count);
    CHECK(fail_count == 0);
}
