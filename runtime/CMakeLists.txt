# =============================================================================
# runtime/ — libvortex_runtime
#
# PyObject model, Tier-0 direct-threaded register VM, deoptimizer, GC/tiering.
# Rule 6: -fno-exceptions (Python-level exceptions are explicit VM state).
# =============================================================================
add_library(vortex_runtime STATIC
    src/objects.cpp
    src/interp.cpp
    src/builtins.cpp
    src/driver.cpp
    src/deopt.cpp
    src/jit.cpp
    src/tracer.cpp
)
target_include_directories(vortex_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(vortex_runtime PUBLIC vortex_common vortex_compiler)
vortex_set_target_flags(vortex_runtime)
