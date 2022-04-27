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

if (MLIR_EXT_USE_LLVM_INSTALL_TREE)
    message(STATUS "Using LLVM install tree.")
    return()
endif()

# Update MLIR source
include(update_mlir)

# Add MLIR/LLVM
if (LINUX)
    set(LLVM_ENABLE_LLD ON CACHE INTERNAL "")
endif()
set(LLVM_INSTALL_UTILS ON CACHE INTERNAL "")
set(LLVM_ENABLE_PROJECTS mlir CACHE INTERNAL "")
set(LLVM_TARGETS_TO_BUILD "host" CACHE INTERNAL "")
set(LLVM_INCLUDE_TOOLS ON CACHE INTERNAL "")
set(LLVM_BUILD_EXAMPLES ON CACHE INTERNAL "")
set(LLVM_ENABLE_ASSERTIONS ON CACHE INTERNAL "")
set(LLVM_ENABLE_RTTI ON CACHE INTERNAL "")
# ON by default
#set(LLVM_INCLUDE_TESTS ON CACHE INTERNAL "")
set(MLIR_INCLUDE_INTEGRATION_TESTS ON CACHE INTERNAL "")
add_subdirectory(llvm-project/llvm)

