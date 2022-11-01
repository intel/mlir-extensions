# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Needed for llvm lit to work.
find_package (Python3 COMPONENTS Interpreter REQUIRED)

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(IMEX_TEST_DEPENDS
        FileCheck
        imex-opt
        level_zero_runner
        dpcomp-runtime
        dpcomp-gpu-runtime
        )

add_lit_testsuite(check-imex "Running the imex regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${IMEX_TEST_DEPENDS}
        )
set_target_properties(check-imex PROPERTIES FOLDER "Tests")

add_lit_testsuites(imex-tests ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${IMEX_TEST_DEPENDS})
