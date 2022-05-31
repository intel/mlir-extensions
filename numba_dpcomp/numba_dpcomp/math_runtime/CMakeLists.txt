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

project(dpcomp-math-runtime LANGUAGES CXX C)

include(GenerateExportHeader)

set(SOURCES_LIST
    lib/common.cpp
    lib/numpy_linalg.cpp
    )
set(HEADERS_LIST
    lib/common.hpp
    )

add_library(${PROJECT_NAME} SHARED ${SOURCES_LIST} ${HEADERS_LIST})
generate_export_header(${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_BINARY_DIR}
    )

if(IMEX_USE_DPNP)
    target_include_directories(${PROJECT_NAME} PRIVATE
        ${DPNP_INCLUDE_DIR}
        )

    target_link_directories(${PROJECT_NAME} PUBLIC ${DPNP_LIBRARY_DIR})
    target_link_libraries(${PROJECT_NAME} PUBLIC "dpnp_backend_c")

    set(DPNP_RPATH "${DPNP_LIBRARY_DIR}" "$ORIGIN/../dpnp")
    set_target_properties(${PROJECT_NAME}
        PROPERTIES
        INSTALL_RPATH "${DPNP_RPATH}"
    )

    target_compile_definitions(${PROJECT_NAME} PRIVATE IMEX_USE_DPNP=1)
endif()
