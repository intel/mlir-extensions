#===============================================================================
# Copyright 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

set(LLVM_VERSION_FILE "${PROJECT_SOURCE_DIR}/llvm-sha.txt")
file(READ "${LLVM_VERSION_FILE}" REVISION_FILE)
string(REGEX MATCH "([A-Za-z0-9]+)" _ ${REVISION_FILE})
set(MLIR_EXT_LLVM_COMMIT_ID ${CMAKE_MATCH_1})
message(STATUS "LLVM COMMIT ID: ${MLIR_EXT_LLVM_COMMIT_ID}")
set(MLIR_EXT_LLVM_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llvm-project")

if (NOT EXISTS "${MLIR_EXT_LLVM_SOURCE_DIR}")
    message(STATUS "Cloning LLVM git repo")
    execute_process(COMMAND git clone https://github.com/llvm/llvm-project.git
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND_ECHO STDOUT
        OUTPUT_VARIABLE LLVM_CLONE_OUTPUT
        ERROR_VARIABLE LLVM_CLONE_ERROR
        ECHO_OUTPUT_VARIABLE
        ECHO_ERROR_VARIABLE
        )
else()
    message(STATUS "LLVM git repo already cloned.")
endif()
execute_process(COMMAND git fetch --prune
    WORKING_DIRECTORY ${MLIR_EXT_LLVM_SOURCE_DIR}
    COMMAND_ECHO STDOUT
    OUTPUT_VARIABLE LLVM_PULL_OUTPUT
    ERROR_VARIABLE LLVM_PULL_ERROR
    ECHO_OUTPUT_VARIABLE
    ECHO_ERROR_VARIABLE
    )

message(STATUS "LLVM: checkout ${MLIR_EXT_LLVM_COMMIT_ID}")

execute_process(COMMAND git checkout ${MLIR_EXT_LLVM_COMMIT_ID}
    WORKING_DIRECTORY ${MLIR_EXT_LLVM_SOURCE_DIR}
    COMMAND_ECHO STDOUT
    OUTPUT_VARIABLE LLVM_CHECKOUT_OUTPUT
    ERROR_VARIABLE LLVM_CHECKOUT_ERROR
    ECHO_OUTPUT_VARIABLE
    ECHO_ERROR_VARIABLE
    )
