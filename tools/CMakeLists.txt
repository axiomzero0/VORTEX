# =============================================================================
# tools/ — the `vortex` CLI
# =============================================================================
add_executable(vortex vortex_main.cpp)
target_link_libraries(vortex PRIVATE vortex_compiler vortex_runtime)
vortex_set_target_flags(vortex)
