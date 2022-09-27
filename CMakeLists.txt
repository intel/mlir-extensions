cmake_minimum_required(VERSION 3.13.4)

# CMP0116: Ninja generators transform `DEPFILE`s from `add_custom_command()`
# New in CMake 3.20. https://cmake.org/cmake/help/latest/policy/CMP0116.html
if(POLICY CMP0116)
    cmake_policy(SET CMP0116 OLD)
endif()

set(IMEX_EXTERNAL_PROJECT_BUILD OFF)

if(NOT (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR) AND NOT MLIR_BINARY_DIR)
    # Building as part of LLVM via the external project mechanism.
    set(IMEX_EXTERNAL_PROJECT_BUILD ON)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
else()
    # Building standalone.
    project(mlir-extensions LANGUAGES CXX C)

    # Must go below project(..)
    include(GNUInstallDirs)

    set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
    list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")
endif()


if(IMEX_EXTERNAL_PROJECT_BUILD)
    message(STATUS "Building IMEX with LLVM_EXTERNAL_PROJECTS")
    set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir )
    set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include )
    set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
    include_directories(SYSTEM ${MLIR_INCLUDE_DIR})
    include_directories(SYSTEM ${MLIR_GENERATED_INCLUDE_DIR})
    include_directories(SYSTEM ${MLIR_TABLEGEN_OUTPUT_DIR})

    set(BACKEND_PACKAGE_STRING "${PACKAGE_STRING}")
    list(APPEND CMAKE_MODULE_PATH "${MLIR_MAIN_SRC_DIR}/cmake/modules")
else()
    message(STATUS "Building IMEX with external MLIR")
    find_package(MLIR REQUIRED CONFIG)

    message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
    message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

    set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
    set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
    set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

    list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
    list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
endif()

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

if(NOT IMEX_EXTERNAL_PROJECT_BUILD)
    find_package(Python3 ${LLVM_MINIMUM_PYTHON_VERSION} REQUIRED
        COMPONENTS Interpreter)
endif()

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

set(LLVM_LIT_ARGS "-sv" CACHE STRING "lit default options")

set(IMEX_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
# LLVM_EXTERNAL_PROJECTS build puts library, executables and tools in LLVM's CMAKE_BINARY_DIR
set(IMEX_BINARY_DIR ${CMAKE_BINARY_DIR})

include(sanitizers)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(test)

option(IMEX_INCLUDE_DOCS "Generate build targets for the IMEX docs." ON)
if (IMEX_INCLUDE_DOCS)
  add_subdirectory(docs)
endif()

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/mlir
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT mlir-headers
    FILES_MATCHING
    PATTERN "*.def"
    PATTERN "*.h"
    PATTERN "*.inc"
    PATTERN "*.td"
    PATTERN "LICENSE.TXT"
)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT mlir-headers
    FILES_MATCHING
    PATTERN "*.def"
    PATTERN "*.h"
    PATTERN "*.gen"
    PATTERN "*.inc"
    PATTERN "*.td"
    PATTERN "CMakeFiles" EXCLUDE
    PATTERN "config.h" EXCLUDE
)

add_subdirectory(cmake/modules)
