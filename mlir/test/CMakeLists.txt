configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(L0_RUNNER_TEST_DEPENDS
        FileCheck
        level_zero_runner
        dpcomp-runtime
        dpcomp-gpu-runtime
        )

add_lit_testsuite(check-l0-runner "Running the level-zero-runner regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${L0_RUNNER_TEST_DEPENDS}
        )
set_target_properties(check-l0-runner PROPERTIES FOLDER "Tests")

add_lit_testsuites(L0_RUNNER ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${L0_RUNNER_TEST_DEPENDS})
