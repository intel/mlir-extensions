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

find_package(LLVM REQUIRED CONFIG)

if (TARGET FileCheck)
  message(STATUS "FileCheck already exists, skipping local")
  return()
endif()

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include(HandleLLVMOptions)

add_executable(FileCheck FileCheck.cpp)

apply_llvm_compile_flags(FileCheck)

target_link_libraries(FileCheck PRIVATE LLVMFileCheck LLVMSupport)

target_include_directories(FileCheck PRIVATE
    ${LLVM_INCLUDE_DIRS}
    )

set(CMAKE_INSTALL_BINDIR ./tools/FileCheck)

install(TARGETS FileCheck
        PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
        )
