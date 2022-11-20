# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

project(dpcomp-python-runtime LANGUAGES CXX C)

find_package (Python3 COMPONENTS Development REQUIRED)

include(GenerateExportHeader)

set(SOURCES_LIST
    lib/Boxing.cpp
    lib/PythonRt.cpp
    )
set(HEADERS_LIST
    lib/PythonRt.hpp
    )

add_library(${PROJECT_NAME} SHARED ${SOURCES_LIST} ${HEADERS_LIST})
generate_export_header(${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME} PRIVATE Python3::Python)

if (NOT Python3_NumPy_INCLUDE_DIRS)
    message(FATAL_ERROR "Python3_NumPy_INCLUDE_DIRS is not set")
endif()
message(STATUS "Python3_NumPy_INCLUDE_DIRS ${Python3_NumPy_INCLUDE_DIRS}")

target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_BINARY_DIR}
    ${Python3_NumPy_INCLUDE_DIRS}
    )
