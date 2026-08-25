# =============================================================================
# VORTEX — CMake superbuild
#
# Laws satisfied here:
#   Rule  6 — compiler & runtime objects are built -fno-exceptions
#   Rule  8 — -fno-rtti everywhere in compiler/ runtime/ tools/
#   Rule 46 — modular targets so single-file changes rebuild in <2s (Rule 50)
#
# Every target is a static library with an explicit, narrow interface so the
# dependency graph stays shallow and incremental rebuilds stay fast.
# =============================================================================
cmake_minimum_required(VERSION 3.24)

project(VORTEX
    VERSION 0.1.0
    DESCRIPTION "VORTEX — tiered optimizing compiler infrastructure for Python 3.16"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# --- VORTEX build switches (Rule 23: named, documented configuration) --------
option(VORTEX_ENABLE_ASSERTS   "Enable VORTEX internal invariant checks"     ON)
option(VORTEX_ENABLE_T0_TRACES "Trace Tier-0 execution (debug builds only)"  OFF)
option(VORTEX_BUILD_TESTS      "Build the test tree"                         ON)
option(VORTEX_BUILD_BENCH      "Build the benchmark tree"                    ON)
option(VORTEX_WERROR           "Treat warnings as errors"                    OFF)

# Cache line override (bytes): by default the target layer probes the real
# value (CPUID CLFLUSH / CTR_EL0 / compiler interference constant). Set this
# ONLY for targets where probing is impossible or known-wrong (e.g. some
# emulators) — a wrong override is worse than no override.
set(VORTEX_CACHE_LINE_BYTES "" CACHE STRING "Force cache line size in bytes (empty = probe)")
if(VORTEX_CACHE_LINE_BYTES)
    add_compile_definitions(VORTEX_CACHE_LINE_BYTES=${VORTEX_CACHE_LINE_BYTES})
endif()

include(cmake/CompilerWarnings.cmake)

# --- Library targets ----------------------------------------------------------
add_subdirectory(common)     # header-only shared vocabulary (Value, ids)
add_subdirectory(compiler)   # libvortex_compiler: IR, passes, backend
add_subdirectory(runtime)    # libvortex_runtime : objects, Tier-0 VM, deopt
add_subdirectory(tools)      # vortex CLI

if(VORTEX_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
if(VORTEX_BUILD_BENCH)
    add_subdirectory(benchmarks)
endif()
