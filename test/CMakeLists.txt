configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(IMEX_OPT_TEST_DEPENDS
        FileCheck count not
        imex-opt
        )

add_lit_testsuite(check-imex-opt "Running the imex-opt regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${IMEX_OPT_TEST_DEPENDS}
        )
set_target_properties(check-imex-opt PROPERTIES FOLDER "Tests")

add_lit_testsuites(IMEX_OPT ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${IMEX_OPT_TEST_DEPENDS})
