// =============================================================================
// benchmarks/bench_main.cpp — VORTEX benchmark driver.
//
// Compiles each Python 3.16-subset workload from bench_cases.hpp once,
// then re-runs the same source `target_iterations` times (each iteration
// = one full Tier-0 + JIT-when-hot execution). Measures wall-clock time
// per iteration and emits a CSV row to stdout.
//
// Usage: vortex_bench [--filter <name>] [--iterations N]
//   --filter <name>   run only the case with that name
//   --iterations N    override every case's target_iterations with N
//
// Output (CSV to stdout):
//   name,iterations,median_ns,mean_ns,p99_ns,correctness
//
// Correctness is "ok" if the printed stdout matches the expected stdout
// for that case; otherwise "FAIL: <diff>". The correctness pin catches
// any regression the perf changes might introduce — the LSRA->XMM
// change MUST NOT alter the printed stdout.
//
// The benchmark is a single executable that links against
// libvortex_compiler + libvortex_runtime (the same libs the unit tests
// use). No external benchmark harness (pyperformance, etc.) — the
// intent is to measure VORTEX's own Tier-0 + JIT path, not CPython.
// =============================================================================

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vortex/rt/driver.hpp"
#include "vortex/rt/interp.hpp"

#include "cases/bench_cases.hpp"

using namespace vortex;
using namespace vortex::rt;

namespace {

struct TimingResult {
    double median_ns;
    double mean_ns;
    double p99_ns;
};

// Capture stdout while running a source. Mirrors tests/unit/lang_test.cpp's
// capture_stdout. Returns the captured string; sets *ok to false on error.
std::string capture_stdout(Vm& vm, const char* src, bool* ok) {
    set_vm_for_builtins(&vm);
    install_builtins(vm.program);
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
    Result<Value> r = run_source(vm, src);
    std::fflush(stdout);
    dup2(saved, fileno(stdout));
    close(saved);
    long size = ftell(cap);
    if (size < 0) [[unlikely]] {
        fclose(cap);
        *ok = r.has_value();
        if (!*ok) {
            std::string msg(r.error().message.data(), r.error().message.size());
            return "ERROR: " + msg;
        }
        return "ERROR: ftell failed";
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

// Run one benchmark case `iterations` times, returning the timing summary
// and the correctness check. Each iteration uses a fresh Vm (so we don't
// carry state across runs — the JIT path's tiering counters would
// otherwise bias the second iteration's runtime).
TimingResult run_case(const BenchCase& c, std::size_t iterations,
                       bool* correct, std::string* got_out) {
    std::vector<double> samples_ns;
    samples_ns.reserve(iterations);
    bool first_ok = false;
    std::string first_out;

    for (std::size_t i = 0; i < iterations; ++i) {
        Vm vm;
        bool ok = false;
        auto t0 = std::chrono::high_resolution_clock::now();
        std::string got = capture_stdout(vm, c.src, &ok);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        samples_ns.push_back(ns);
        if (i == 0) {
            first_ok = ok;
            first_out = got;
        } else {
            // Subsequent runs must match the first run's output AND
            // correctness — pin determinism (Rule 34).
            if (ok != first_ok || got != first_out) {
                std::fprintf(stderr,
                    "[bench] WARN: %s iter %zu diverged from iter 0\n",
                    c.name, i);
            }
        }
    }

    std::sort(samples_ns.begin(), samples_ns.end());
    TimingResult out;
    out.median_ns = samples_ns[samples_ns.size() / 2];
    double sum = 0;
    for (double s : samples_ns) sum += s;
    out.mean_ns = sum / samples_ns.size();
    // p99: the value at index floor(0.99 * N). For N <= 100, this is
    // effectively the max; for larger N it's the true 99th percentile.
    std::size_t p99_idx = static_cast<std::size_t>(
        0.99 * (samples_ns.size() - 1));
    out.p99_ns = samples_ns[p99_idx];

    *correct = first_ok && (first_out == c.expect);
    *got_out = first_out;
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    const char* filter = nullptr;
    std::size_t override_iters = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--filter") == 0 && i + 1 < argc) {
            filter = argv[++i];
        } else if (std::strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            override_iters = static_cast<std::size_t>(std::atol(argv[++i]));
        } else {
            std::fprintf(stderr, "usage: %s [--filter <name>] [--iterations N]\n",
                         argv[0]);
            return 2;
        }
    }

    std::printf("name,iterations,median_ns,mean_ns,p99_ns,correctness\n");
    for (std::size_t i = 0; i < kBenchCaseCount; ++i) {
        const BenchCase& c = kBenchCases[i];
        if (filter && std::strcmp(filter, c.name) != 0) continue;
        std::size_t iters = override_iters > 0 ? override_iters
                                                : c.target_iterations;

        bool correct = false;
        std::string got;
        TimingResult t = run_case(c, iters, &correct, &got);

        std::printf("%s,%zu,%.0f,%.0f,%.0f,%s\n",
                    c.name, iters, t.median_ns, t.mean_ns, t.p99_ns,
                    correct ? "ok" : "FAIL");
        if (!correct) {
            std::fprintf(stderr,
                "[bench] %s: correctness FAIL\n  expect: %s\n  got: %s\n",
                c.name, c.expect, got.c_str());
        }
        std::fflush(stdout);
    }
    return 0;
}
