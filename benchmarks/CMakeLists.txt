# benchmarks/ — VORTEX's own benchmark suite (Rule 25: heuristic constants
# justified by data). Compiles a set of Python 3.16-subset workloads via
# the Tier-0 + JIT-when-hot path and reports per-iteration wall-clock
# timing as CSV to stdout.
#
# Usage:
#   cmake --build build --target vortex_bench
#   ./build/benchmarks/vortex_bench [--filter <name>] [--iterations N]
#
# Output (CSV to stdout):
#   name,iterations,median_ns,mean_ns,p99_ns,correctness
add_executable(vortex_bench
    bench_main.cpp
)
target_include_directories(vortex_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}          # for cases/bench_cases.hpp
    ${CMAKE_SOURCE_DIR}/tests/unit        # for harness.hpp (unused but harmless)
)
target_link_libraries(vortex_bench PRIVATE
    "-Wl,--start-group"
    vortex_compiler
    vortex_runtime
    "-Wl,--end-group"
)
# Task 24: codegen.cpp (in vortex_compiler) takes the address of
# vortex_deopt_entry / vortex_jit_bridge (defined in vortex_runtime).
# The driver's new eager JIT wiring pulls codegen.o into the link
# (because compile_program now references backend::compile_unit). The
# single-pass static-linker doesn't re-scan vortex_runtime.a after
# codegen.o's late-arriving undefined refs surface. --start-group /
# --end-group forces re-scanning until the undefined pool stabilizes.
# The flags are embedded in target_link_libraries (NOT
# target_link_options) so they wrap the static libs in the right
# position on the command line.
# The benchmark uses set_vm_for_builtins / install_builtins from the
# runtime; links transitively through vortex_runtime.
