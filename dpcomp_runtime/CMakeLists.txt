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

project(dpcomp-runtime LANGUAGES CXX C)

include(GenerateExportHeader)

if(IMEX_ENABLE_TBB_SUPPORT)
    find_package(TBB REQUIRED)
endif()

find_package(MLIR REQUIRED CONFIG)

set(SOURCES_LIST
    lib/context.cpp
    lib/tbb_parallel.cpp
    lib/memory.cpp
    )
set(HEADERS_LIST
    )

add_library(${PROJECT_NAME} SHARED ${SOURCES_LIST} ${HEADERS_LIST})
generate_export_header(${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE
    ${MLIR_INCLUDE_DIRS}
    PRIVATE
    ${PROJECT_BINARY_DIR}
    )

if(IMEX_ENABLE_TBB_SUPPORT)
    target_compile_definitions(${PROJECT_NAME} PRIVATE IMEX_ENABLE_TBB_SUPPORT=1)
    target_link_libraries(${PROJECT_NAME} TBB::tbb)
endif()
