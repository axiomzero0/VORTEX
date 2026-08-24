# =============================================================================
# compiler/ — libvortex_compiler
#
# Tier-agnostic compiler infrastructure: support utilities, IR (Sea of Nodes),
# Python-subset frontend, analyses, the 55-pass pipeline, and the x86-64
# backend (machine code emission for Tiers 1/2/3).
#
# Rule 50: sources are split per-module so single-file edits rebuild fast.
# =============================================================================
add_library(vortex_compiler STATIC
    src/support/diagnostic.cpp
    src/support/symbol_table.cpp
    src/support/telemetry.cpp
    src/ir/graph.cpp
    src/ir/node_kind.cpp
    src/ir/printer.cpp
    src/ir/parser.cpp
    src/ir/verifier.cpp
    src/frontend/lexer.cpp
    src/frontend/parser.cpp
    src/frontend/lowering.cpp
    src/pipeline/scheduler.cpp
)

target_include_directories(vortex_compiler PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/runtime/include)
target_link_libraries(vortex_compiler PUBLIC vortex_common)
vortex_set_target_flags(vortex_compiler)

find_package(Threads REQUIRED)
target_link_libraries(vortex_compiler PUBLIC Threads::Threads)
