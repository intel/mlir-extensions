# SPDX-FileCopyrightText: 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

project(imex-opt LANGUAGES CXX C)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(SOURCES_LIST
    Main.cpp
    Passes.cpp
    )
set(HEADERS_LIST
    )

add_executable(${PROJECT_NAME} ${SOURCES_LIST} ${HEADERS_LIST})

apply_llvm_compile_flags(${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME} PRIVATE
    mlir-extensions
    ${dialect_libs}
    ${conversion_libs}
    MLIROptLib
    MLIRPass
    MLIRSupport
    )

target_include_directories(${PROJECT_NAME} PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
    )

set(CMAKE_INSTALL_BINDIR "./tools/${PROJECT_NAME}")

install(TARGETS ${PROJECT_NAME}
        PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
        )
