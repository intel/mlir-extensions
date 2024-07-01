# Copyright 2024 Intel Corporation
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

find_package(OpenCL REQUIRED)

if(NOT OpenCL_FOUND)
    message(FATAL_ERROR "OpenCL not found.")
endif()

add_mlir_library(opencl-runtime
    SHARED
    OpenCLRuntimeWrappers.cpp

    EXCLUDE_FROM_LIBMLIR
  )

check_cxx_compiler_flag("-frtti" CXX_HAS_FRTTI_FLAG)
if(NOT CXX_HAS_FRTTI_FLAG)
    message(FATAL_ERROR "CXX compiler does not accept flag -frtti")
endif()
target_compile_options (opencl-runtime PUBLIC -fexceptions -frtti)

target_include_directories(opencl-runtime PRIVATE
    ${MLIR_INCLUDE_DIRS}
    ${OpenCL_INCLUDE_DIRS}
    )

message(STATUS "OpenCL Libraries: ${OpenCL_LIBRARIES}")
target_link_libraries(opencl-runtime PRIVATE ${OpenCL_LIBRARIES})

