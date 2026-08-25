# =============================================================================
# compiler/ — libvortex_compiler
#
# Tier-agnostic compiler infrastructure: support utilities, IR (Sea of Nodes),
# Python-subset frontend, analyses, the 51-pass pipeline (one file per pass —
# Rule 50/51: modular targets, single-file edits rebuild fast), and the
# scheduler emitting Tier-0 bytecode.
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
    src/passes/p03_trivial_dce.cpp
    src/passes/p04_local_constfold.cpp
    src/passes/p05_algebraic_simplify.cpp
    src/passes/p06_cf_simplify.cpp
    src/passes/p07_local_cse.cpp
    src/passes/p08_sccp.cpp
    src/passes/p09_redundant_stores.cpp
    src/passes/p10_early_gvn.cpp
    src/passes/p11_andersen_points_to.cpp
    src/passes/p12_cfl_alias.cpp
    src/passes/p13_flow_sensitive_alias.cpp
    src/passes/p14_demand_driven_alias.cpp
    src/passes/p15_shape_analysis.cpp
    src/passes/p16_ic_monomorphism.cpp
    src/passes/p17_mro_linearization.cpp
    src/passes/p18_side_effect_analysis.cpp
    src/passes/p19_callgraph_pgo.cpp
    src/passes/p20_speculative_inlining.cpp
    src/passes/p21_partial_inlining.cpp
    src/passes/p22_recursive_inlining.cpp
    src/passes/p23_closure_devirt.cpp
    src/passes/p24_generator_deforestation.cpp
    src/passes/p25_exception_outlining.cpp
    src/passes/p26_ipcp.cpp
    src/passes/p27_licm.cpp
    src/passes/p28_speculative_licm.cpp
    src/passes/p29_induction_vars.cpp
    src/passes/p30_loop_unrolling.cpp
    src/passes/p31_slp_vectorization.cpp
    src/passes/p32_loop_vectorization.cpp
    src/passes/p33_polyhedral.cpp
    src/passes/p34_software_pipelining.cpp
    src/passes/p35_loop_fusion_fission.cpp
    src/passes/p36_bounds_check_elim.cpp
    src/passes/p37_none_check_elim.cpp
    src/passes/p38_gil_hoisting.cpp
    src/passes/p39_escape_analysis.cpp
    src/passes/p40_partial_escape_analysis.cpp
    src/passes/p41_object_inlining.cpp
    src/passes/p42_region_memory.cpp
    src/passes/p43_refcnt_opt.cpp
    src/passes/p44_write_barrier_elim.cpp
    src/passes/p45_string_interning.cpp
    src/passes/p46_dict_layout.cpp
    src/passes/p47_box_unbox_elim.cpp
    src/passes/p48_tlab_sizing.cpp
    src/passes/p49_effect_reordering.cpp
    src/passes/p50_late_gvn.cpp
    src/passes/p51_global_dce.cpp
    src/passes/pipeline.cpp
    src/passes/analyses/dominators.cpp
    src/passes/analyses/loops.cpp
)

target_include_directories(vortex_compiler PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/runtime/include)
target_link_libraries(vortex_compiler PUBLIC vortex_common)
vortex_set_target_flags(vortex_compiler)

find_package(Threads REQUIRED)
target_link_libraries(vortex_compiler PUBLIC Threads::Threads)
