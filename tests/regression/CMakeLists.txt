# =============================================================================
# tests/regression/ — Regression test infrastructure
#
# Distinct from tests/unit/ in scope and intent:
#   - tests/unit/  : correctness of each component in isolation
#   - tests/regression/ : end-to-end guards against behavioral drift across
#                          the full pipeline, the JIT bridge, and the opt-in
#                          flag surface. Each test below exists because a
#                          specific regression has happened before (the
#                          IBE-18 home-slot clobber is the canonical example)
#                          and we will not let it silently come back.
#
# A separate executable (vortex_regression_tests) and a separate ctest entry
# keep these runnable independently from the unit suite — useful when
# bisecting a behavioral change to a specific commit, since the regression
# suite's per-test granularity is by *regression class* (idempotency,
# verifier, determinism, JIT differential, etc.) not by component.
# =============================================================================

add_executable(vortex_regression_tests
    main.cpp
    pass_idempotency_regr.cpp
    verifier_after_each_pass_regr.cpp
    pipeline_ir_fingerprint_regr.cpp
    determinism_regr.cpp
    jit_differential_regr.cpp
    pass_budget_guards_regr.cpp
    opt_in_toggle_regr.cpp
)

# Make the existing self-registering harness from tests/unit/ reusable here
# — same TEST/CHECK/CHECK_EQ macros, same Registrar, so the regression
# files read identically to the unit files.
target_include_directories(vortex_regression_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../unit)

target_link_libraries(vortex_regression_tests PRIVATE vortex_compiler vortex_runtime)
vortex_set_target_flags(vortex_regression_tests)

add_test(NAME regression COMMAND vortex_regression_tests)
