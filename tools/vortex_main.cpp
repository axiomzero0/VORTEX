// =============================================================================
// vortex/tools/vortex_main.cpp — the `vortex` CLI.
//
// Usage: vortex <file.py>          Run a Python source file
//        vortex -c "source"        Run a source string
//        vortex -e "<expr>"        Evaluate an expression, print its repr
//
// The CLI is the thin entry point: it reads the source, hands it to the
// runtime driver (which runs the full Tier-0 + JIT pipeline), and prints
// any diagnostics to stderr. Exit code is 0 on success, 1 on compile
// error, 2 on usage error.
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

#include "vortex/rt/driver.hpp"
#include "vortex/rt/interp.hpp"
#include "vortex/support/symbol_table.hpp"

namespace {

[[nodiscard]] std::string read_file(const char* path) noexcept {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

[[nodiscard]] int run_source(std::string_view src, const char* origin) noexcept {
    vortex::rt::Vm vm;
    vortex::rt::set_vm_for_builtins(&vm);
    vortex::rt::install_builtins(vm.program);

    vortex::SymbolId mod_name = vortex::global_symbols().intern("__main__");
    vortex::rt::CompileOutcome outcome =
        vortex::rt::compile_program(vm, src, mod_name);
    if (!outcome.ok) {
        const vortex::Diagnostic& d = outcome.diagnostic;
        std::fprintf(stderr, "vortex: %s: %.*s\n", origin,
                     static_cast<int>(d.message.size()), d.message.data());
        if (!d.fix.empty()) {
            std::fprintf(stderr, "  hint: %.*s\n",
                         static_cast<int>(d.fix.size()), d.fix.data());
        }
        return 1;
    }

    vortex::rt::CodeUnit* toplevel =
        outcome.module_unit_id < vm.program.units.size()
            ? vm.program.units[outcome.module_unit_id]
            : nullptr;
    if (!toplevel) {
        std::fprintf(stderr, "vortex: internal error: no module toplevel unit\n");
        return 1;
    }

    vortex::Result<vortex::Value> r = vm.run_module(toplevel);
    if (!r) {
        const vortex::Diagnostic& d = r.error();
        std::fprintf(stderr, "vortex: runtime error: %.*s\n",
                     static_cast<int>(d.message.size()), d.message.data());
        return 1;
    }

    // Rule 26 / Rule 119: Dump telemetry + profiler on exit.
    vm.telemetry.write_report(stderr);
    vm.profiler.write_report(stderr);

    return 0;
}

void usage() noexcept {
    std::fputs(
        "VORTEX — optimizing compiler for Python 3.16\n"
        "\n"
        "Usage:\n"
        "  vortex <file.py>          Run a Python source file\n"
        "  vortex -c \"<source>\"      Run a source string\n"
        "  vortex -e \"<expr>\"        Evaluate an expression, print result\n"
        "  vortex --help             Show this help\n",
        stderr);
}

}  // namespace

int main(int argc, char** argv) noexcept {
    if (argc < 2) {
        usage();
        return 2;
    }

    // Parse flags.
    if (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0) {
        usage();
        return 0;
    }
    if (std::strcmp(argv[1], "-c") == 0) {
        if (argc < 3) { usage(); return 2; }
        return run_source(argv[2], "-c");
    }

    // Default: treat argv[1] as a file path.
    std::string src = read_file(argv[1]);
    if (src.empty()) {
        std::fprintf(stderr, "vortex: cannot read file: %s\n", argv[1]);
        return 2;
    }
    return run_source(src, argv[1]);
}
