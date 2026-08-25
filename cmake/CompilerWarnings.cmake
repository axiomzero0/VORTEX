# =============================================================================
# VORTEX compiler warning & hardening configuration
#
# Laws:
#   Rule  6 — -fno-exceptions : no unwinding on any compiler/runtime path.
#   Rule  8 — -fno-rtti       : type switching via enum class NodeKind only.
#   Rule 48 — [[nodiscard]] discipline is compiler-enforced (-Wunused-result).
#   Rule 61 — perf matters: -O2 (O3 benchmarked worse on geometric mean of the
#             in-tree suite; see docs/adr/0003-build-flags.md).
# =============================================================================
set(VORTEX_WARNING_FLAGS
    -Wall -Wextra
    -Wshadow -Wformat=2 -Wnull-dereference
    -Wunused-result)
# Note: -Wpedantic is deliberately omitted: VORTEX_TRY (Rule 22) uses GCC
# statement expressions, a documented, sanctioned extension. See
# docs/adr/0001-try-macro.md.

function(vortex_set_target_flags target)
    target_compile_options(${target} PRIVATE
        -fno-exceptions      # Rule 6
        -fno-rtti            # Rule 8
        ${VORTEX_WARNING_FLAGS}
        -fno-omit-frame-pointer)   # deopt trampolines walk frame pointers

    # Optimized but debuggable: keep invariants cheap in Release, exact in Debug.
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_definitions(${target} PRIVATE VORTEX_DEBUG=1)
        target_compile_options(${target} PRIVATE -O0 -g)
    else()
        target_compile_definitions(${target} PRIVATE VORTEX_DEBUG=0 NDEBUG)
        target_compile_options(${target} PRIVATE -O2 -g1)
    endif()

    if(VORTEX_WERROR)
        target_compile_options(${target} PRIVATE -Werror)
    endif()

    if(VORTEX_ENABLE_ASSERTS)
        target_compile_definitions(${target} PRIVATE VORTEX_ENABLE_ASSERTS=1)
    else()
        target_compile_definitions(${target} PRIVATE VORTEX_ENABLE_ASSERTS=0)
    endif()

    if(VORTEX_ENABLE_T0_TRACES AND CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_definitions(${target} PRIVATE VORTEX_T0_TRACE=1)
    endif()
endfunction()
