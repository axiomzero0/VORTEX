# =============================================================================
# common/ — header-only shared vocabulary (Value, ids) used by BOTH the
# compiler and runtime libraries. INTERFACE target: zero build cost (Rule 50).
# =============================================================================
add_library(vortex_common INTERFACE)
target_include_directories(vortex_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
