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

if(NOT LEVEL_ZERO_DIR)
    message(FATAL_ERROR "LEVEL_ZERO_DIR is not set")
endif()

set(CMAKE_INSTALL_RPATH ${LEVEL_ZERO_DIR}/lib)
add_mlir_library(level-zero-runtime
    SHARED
    LevelZeroRuntimeWrappers.cpp

    EXCLUDE_FROM_LIBMLIR
  )

target_compile_options (level-zero-runtime PUBLIC -fexceptions)

target_include_directories(level-zero-runtime PRIVATE
    ${MLIR_INCLUDE_DIRS}
    ${LEVEL_ZERO_DIR}/include/
    )

target_link_directories(level-zero-runtime PRIVATE
    ${LEVEL_ZERO_DIR}/lib
    )

target_link_libraries(level-zero-runtime PRIVATE ze_loader)
