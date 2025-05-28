if(WIN32)
    set(MLIR_RUNNER_UTILS_DIR ${LLVM_BINARY_DIR}/bin)
else()
    set(MLIR_RUNNER_UTILS_DIR ${LLVM_LIBRARY_DIR})
endif()
set(IMEX_TEST_DEPENDS
        imex-opt
        imex-cpu-runner
        mlir-runner
        mlir_c_runner_utils
        mlir_runner_utils
        imex_runner_utils
        )

if(IMEX_EXTERNAL_PROJECT_BUILD)
    # llvm external project has targets FileCheck count and not
    list(APPEND IMEX_TEST_DEPENDS
        FileCheck
        count
        not
    )
elseif(DEFINED LLVM_BUILD_BINARY_DIR)
    # separate LLVM build tree
    # make sure that targets FileCheck count and not are exported
    if(TARGET FileCheck AND TARGET count AND TARGET not)
        list(APPEND IMEX_TEST_DEPENDS
            FileCheck
            count
            not
        )
    else()
        message(FATAL_ERROR "Targets FileCheck, not and count missing. Please build LLVM with -DLLVM_INSTALL_UTILS=ON")
    endif()
else()
    # separate LLVM install
    # make sure that LLVM_EXTERNAL_LIT is set or found
    if(NOT DEFINED LLVM_EXTERNAL_LIT)
        find_program (LLVM_EXTERNAL_LIT NAMES llvm-lit lit PATHS ${LLVM_TOOLS_BINARY_DIR} REQUIRED)
    endif()
    message(STATUS "IMEX using LLVM_EXTERNAL_LIT: ${LLVM_EXTERNAL_LIT}")
endif()

if(IMEX_ENABLE_VULKAN_RUNNER)
    list(APPEND IMEX_TEST_DEPENDS
        mlir-vulkan-runner
        vulkan-runtime-wrappers
        )
endif()

if(NOT DEFINED MLIR_ENABLE_SYCL_RUNNER)
    set(MLIR_ENABLE_SYCL_RUNNER 0)
endif()

if(MLIR_ENABLE_SYCL_RUNNER)
    list(APPEND IMEX_TEST_DEPENDS
        mlir_sycl_runtime
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

if(IMEX_ENABLE_L0_RUNTIME OR IMEX_ENABLE_SYCL_RUNTIME)
    list(APPEND IMEX_TEST_DEPENDS
        l0-fp64-checker
        )
endif()

if(IMEX_ENABLE_BINDINGS_PYTHON)
    list(APPEND IMEX_TEST_DEPENDS
        ImexPythonModules
        )
endif()

# "Gen" is the root of all generated test cases
add_subdirectory(Gen)

set(IMEX_LIT_CFG_PY ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${IMEX_LIT_CFG_PY}
  )

add_lit_testsuite(check-static "Running the IMEX regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${IMEX_TEST_DEPENDS}
        )
set_target_properties(check-static PROPERTIES FOLDER "Tests")

add_custom_target(check-imex
        DEPENDS check-static
        )

add_lit_testsuites(IMEX ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${IMEX_TEST_DEPENDS})
