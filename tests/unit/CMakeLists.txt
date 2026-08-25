# Unit tests: zero-dependency self-registering harness (Rule 52:
# self-contained — no network, no external test framework).
add_executable(vortex_unit_tests
    main.cpp
    support_test.cpp
    ir_test.cpp
    frontend_test.cpp
    lang_test.cpp
)
target_link_libraries(vortex_unit_tests PRIVATE vortex_compiler vortex_runtime)
vortex_set_target_flags(vortex_unit_tests)
add_test(NAME unit COMMAND vortex_unit_tests)
