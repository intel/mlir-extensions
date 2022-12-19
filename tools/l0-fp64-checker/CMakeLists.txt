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

add_imex_tool(l0-fp64-checker l0-fp64-checker.cpp)

target_compile_options (l0-fp64-checker PUBLIC -fexceptions)

target_include_directories(l0-fp64-checker PRIVATE
    ${LEVEL_ZERO_DIR}/include/
    )

target_link_directories(l0-fp64-checker PRIVATE
    ${LEVEL_ZERO_DIR}/lib
    )

target_link_libraries(l0-fp64-checker PRIVATE ze_loader)
