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
    vortex_runtime
    vortex_compiler
)
# The benchmark uses set_vm_for_builtins / install_builtins from the
# runtime; links transitively through vortex_runtime.
