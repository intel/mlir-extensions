configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

if(WIN32)
    set(MLIR_RUNNER_UTILS_DIR ${LLVM_BINARY_DIR}/bin)
else()
    set(MLIR_RUNNER_UTILS_DIR ${LLVM_LIBRARY_DIR})
endif()
set(IMEX_TEST_DEPENDS
        FileCheck count not
        imex-opt
        mlir-cpu-runner
        mlir_c_runner_utils
        mlir_runner_utils
        )

if(IMEX_ENABLE_VULKAN_RUNNER)
    list(APPEND IMEX_TEST_DEPENDS
        mlir-vulkan-runner
        vulkan-runtime-wrappers
        )
endif()

if(IMEX_ENABLE_SYCL_RUNTIME)
    list(APPEND IMEX_TEST_DEPENDS
        sycl-runtime
        )
endif()

if(IMEX_ENABLE_L0_RUNTIME)
    list(APPEND IMEX_TEST_DEPENDS
        level-zero-runtime
        )
endif()

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

add_lit_testsuite(check-ptensor "Running the IMEX/ptensor regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}/*/*PTensor*
        ${CMAKE_CURRENT_BINARY_DIR}/*/*Dist*
        ${CMAKE_CURRENT_BINARY_DIR}/imex-runner
        DEPENDS ${IMEX_TEST_DEPENDS}
        )
set_target_properties(check-ptensor PROPERTIES FOLDER "Tests")

add_lit_testsuites(IMEX ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${IMEX_TEST_DEPENDS})
