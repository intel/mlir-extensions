cmake_minimum_required(VERSION 3.13.4...3.18)

# CMP0116: Ninja generators transform `DEPFILE`s from `add_custom_command()`
# New in CMake 3.20. https://cmake.org/cmake/help/latest/policy/CMP0116.html
if(POLICY CMP0116)
    cmake_policy(SET CMP0116 OLD)
endif()

# Building standalone.
project(imex-integration-example LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

find_package(IMEX REQUIRED CONFIG)

message(STATUS "Using IMEXConfig.cmake in: ${IMEX_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# For redirecting llvm_add_library() and others
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

# LLVM_INCLUDE_DIRS is the same as MLIR_INCLUDE_DIRS in an MLIR install tree
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${IMEX_INCLUDE_DIRS})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(mlir_conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(mlir_extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(imex_dialect_libs GLOBAL PROPERTY IMEX_DIALECT_LIBS)
get_property(imex_conversion_libs GLOBAL PROPERTY IMEX_CONVERSION_LIBS)
get_property(imex_extension_libs GLOBAL PROPERTY IMEX_EXTENSION_LIBS)
set(LIBS
        ${mlir_dialect_libs}
        ${mlir_conversion_libs}
        ${mlir_extension_libs}
        ${imex_dialect_libs}
        ${imex_conversion_libs}
        ${imex_extension_libs}
        MLIROptLib
        IMEXTransforms
        IMEXUtil
        )
add_llvm_executable(sample-opt sample-opt.cpp)

llvm_update_compile_flags(sample-opt)
target_link_libraries(sample-opt PRIVATE ${LIBS} IMEXTransforms)
