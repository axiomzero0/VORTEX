# =============================================================================
# tools/ — the `vortex` CLI
# =============================================================================
add_executable(vortex vortex_main.cpp)
# codegen.cpp (in vortex_compiler) takes the address of vortex_deopt_entry
# / vortex_jit_bridge (defined in vortex_runtime). The single-pass static
# linker doesn't re-scan vortex_runtime.a after codegen.o's late-arriving
# undefined refs surface. --start-group / --end-group forces re-scanning
# until the undefined pool stabilizes (same pattern as vortex_bench).
target_link_libraries(vortex PRIVATE
    "-Wl,--start-group"
    vortex_compiler
    vortex_runtime
    "-Wl,--end-group"
)
vortex_set_target_flags(vortex)
