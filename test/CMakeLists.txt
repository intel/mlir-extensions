configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(MLIR_RUNNER_UTILS_DIR ${LLVM_LIBRARY_DIR})
set(IMEX_TEST_DEPENDS
        FileCheck count not
        imex-opt
        imex-runner
        mlir-cpu-runner
        mlir_c_runner_utils
        mlir_runner_utils
        )

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

add_lit_testsuite(check-imex "Running the IMEX regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${IMEX_TEST_DEPENDS}
        )
set_target_properties(check-imex PROPERTIES FOLDER "Tests")

add_lit_testsuites(IMEX ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${IMEX_TEST_DEPENDS})
