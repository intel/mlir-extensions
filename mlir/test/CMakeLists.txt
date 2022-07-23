# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Needed for llvm lit to work.
find_package (Python3 COMPONENTS Interpreter REQUIRED)

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(L0_RUNNER_TEST_DEPENDS
        FileCheck
        level_zero_runner
        )

add_lit_testsuite(check-l0-runner "Running the level-zero-runner regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${L0_RUNNER_TEST_DEPENDS}
        )
set_target_properties(check-l0-runner PROPERTIES FOLDER "Tests")

add_lit_testsuites(L0_RUNNER ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${L0_RUNNER_TEST_DEPENDS})
