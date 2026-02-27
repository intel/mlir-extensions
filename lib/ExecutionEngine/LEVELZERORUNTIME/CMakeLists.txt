# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

find_package(LevelZero)

if(NOT LevelZero_FOUND)
  message(FATAL_ERROR "LevelZero not found. Please set LEVEL_ZERO_DIR.")
endif()

add_mlir_library(level-zero-runtime
  SHARED
  LevelZeroRuntimeWrappers.cpp

  EXCLUDE_FROM_LIBMLIR

  STANDALONE

  LINK_LIBS
  imex_execution_engine_utils
)

target_compile_options(level-zero-runtime PUBLIC -fexceptions)

target_link_libraries(level-zero-runtime PRIVATE LevelZero::LevelZero)

set_property(TARGET level-zero-runtime APPEND PROPERTY BUILD_RPATH "${LevelZero_LIBRARIES_DIR}")
