# Copyright 2022 Intel Corporation
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

set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -Wno-return-type -Wno-cast-qual")

find_package(SyclRuntime)

if(NOT SyclRuntime_FOUND)
    message(FATAL_ERROR "syclRuntime not found. Please set check oneapi installation and run setvars.sh.")
endif()

find_package(LevelZero)

if(NOT LevelZero_FOUND)
    message(FATAL_ERROR "LevelZero not found. Please set LEVEL_ZERO_DIR.")
endif()

add_mlir_library(sycl-runtime
    SHARED
    SyclRuntimeWrappers.cpp

    EXCLUDE_FROM_LIBMLIR

    LINK_LIBS
    imex_execution_engine_utils
)

check_cxx_compiler_flag("-frtti" CXX_HAS_FRTTI_FLAG)

if(NOT CXX_HAS_FRTTI_FLAG)
    message(FATAL_ERROR "CXX compiler does not accept flag -frtti")
endif()

target_compile_options(sycl-runtime PUBLIC -fexceptions -frtti)

target_include_directories(sycl-runtime PRIVATE
    ${MLIR_INCLUDE_DIRS}
)

target_link_libraries(sycl-runtime PRIVATE LevelZero::LevelZero SyclRuntime::SyclRuntime)

set_property(TARGET sycl-runtime APPEND PROPERTY BUILD_RPATH "${LevelZero_LIBRARIES_DIR}" "${SyclRuntime_LIBRARIES_DIR}")
