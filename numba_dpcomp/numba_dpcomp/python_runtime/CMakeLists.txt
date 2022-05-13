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

project(dpcomp-python-runtime LANGUAGES CXX C)

find_package (Python3 COMPONENTS Development NumPy REQUIRED)

include(GenerateExportHeader)

set(SOURCES_LIST
    lib/boxing.cpp
    lib/python-rt.cpp
    )
set(HEADERS_LIST
    lib/python-rt.hpp
    )

add_library(${PROJECT_NAME} SHARED ${SOURCES_LIST} ${HEADERS_LIST})
generate_export_header(${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME} PRIVATE Python3::Python)

if (NOT Python3_NumPy_INCLUDE_DIRS)
    message(FATAL_ERROR "Python3_NumPy_INCLUDE_DIRS is not set")
endif()
message("Python3_NumPy_INCLUDE_DIRS ${Python3_NumPy_INCLUDE_DIRS}")

target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_BINARY_DIR}
    ${Python3_NumPy_INCLUDE_DIRS}
    )
