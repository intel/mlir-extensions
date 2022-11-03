# Copyright 2022 Intel Corporation
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

if(NOT SYCL_DIR)
    message(FATAL_ERROR "SYCL_DIR is not set")
endif()

if(NOT LEVEL_ZERO_DIR)
    message(FATAL_ERROR "LEVEL_ZERO_DIR is not set")
endif()

add_mlir_library(sycl-runtime
    SHARED
    SyclRuntimeWrappers.cpp

    EXCLUDE_FROM_LIBMLIR
  )

target_compile_options (sycl-runtime PUBLIC -fexceptions)

target_include_directories(sycl-runtime PRIVATE
    ${MLIR_INCLUDE_DIRS}
    ${SYCL_DIR}/include/
    ${SYCL_DIR}/include/sycl
    ${LEVEL_ZERO_DIR}/include/
    )

target_link_directories(sycl-runtime PRIVATE
    ${SYCL_DIR}/lib
    ${LEVEL_ZERO_DIR}/lib
    )

target_link_libraries(sycl-runtime PRIVATE ze_loader sycl)
