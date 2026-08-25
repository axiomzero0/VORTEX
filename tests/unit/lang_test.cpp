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
    // TTC-1 fix: tmpfile can fail (resource limits). Guard every C-API
    // call whose result is consumed below; otherwise we feed a nullptr
    // FILE* into dup2/fread/fclose and crash with a null-deref or read
    // garbage into the expected string.
    if (!cap) [[unlikely]] {
        dup2(saved, fileno(stdout));
        close(saved);
        *ok = false;
        return "ERROR: tmpfile() failed in capture_stdout";
    }
    dup2(fileno(cap), fileno(stdout));
    Result<Value> r = run_source(vm, src);
    std::fflush(stdout);
    dup2(saved, fileno(stdout));
    close(saved);
    long size = ftell(cap);
    // TTC-1: ftell returns -1L on error (and 0 on an empty stream that
    // may legitimately have nothing to read). A negative size used to
    // construct std::string(size, ' ') -> std::string::__init with
    // size_t(-1) -> std::length_error crash. Guard the negative case.
    if (size < 0) [[unlikely]] {
        fclose(cap);
        *ok = r.has_value();
        if (!*ok) {
            std::string msg(r.error().message.data(), r.error().message.size());
            return "ERROR: " + msg;
        }
        return "ERROR: ftell failed in capture_stdout";
    }
    rewind(cap);
    std::string got(static_cast<std::size_t>(size), ' ');
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
