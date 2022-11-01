# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

project(dpcomp-runtime LANGUAGES CXX C)

include(GenerateExportHeader)

if(IMEX_ENABLE_TBB_SUPPORT)
    find_package(TBB REQUIRED)
endif()

find_package(MLIR REQUIRED CONFIG)

set(SOURCES_LIST
    lib/Context.cpp
    lib/Memory.cpp
    lib/TbbParallel.cpp
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
