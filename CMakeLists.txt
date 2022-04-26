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

# Match requirement from LLVM/MLIR
cmake_minimum_required(VERSION 3.13.4)

project(mlir-extensions)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

if(POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif()

if(POLICY CMP0116)
    cmake_policy(SET CMP0116 OLD)
endif()

# SDL
# set(CMAKE_VERBOSE_MAKEFILE on) # enable for debug
if (CMAKE_SYSTEM_NAME STREQUAL Windows)
    add_compile_options(-GS -D_CRT_SECURE_NO_WARNINGS)
    add_link_options(-DYNAMICBASE -NXCOMPAT -GUARD:CF)
    # add_link_options(-INTEGRITYCHECK) # require signatures of libs, only recommended
endif()
if (CMAKE_SYSTEM_NAME STREQUAL Linux)
    add_compile_options(-D_FORTIFY_SOURCE=2 -Wformat -Wformat-security -Werror=format-security -fno-delete-null-pointer-checks -fstack-protector-strong -fno-strict-overflow -Wall)
    add_compile_options(-fstack-clash-protection -fcf-protection=full) # v8.0 and newer
    # add_compile_options(-mcet) # v8.0 and newer # unrecognized command line option '-mcet', only recommended
    add_link_options(-Wl,-z,noexecstack,-z,relro,-z,now)
endif()
if (CMAKE_SYSTEM_NAME STREQUAL Darwin)
    add_compile_options(-D_FORTIFY_SOURCE=2 -Wformat -Wformat-security -Werror=format-security -fno-delete-null-pointer-checks -fstack-protector-strong -fno-strict-overflow -Wall)
    add_compile_options(-fcf-protection=full) # v8.0 and newer
    # add_compile_options(-mcet) # v8.0 and newer # unrecognized command line option '-mcet', only recommended
endif()

option(DPNP_ENABLE "Use DPNP for some math functions" OFF)
option(GPU_ENABLE "Enable GPU codegen" OFF)
option(NUMBA_ENABLE "Enable numba-based python frontend" OFF)
option(TBB_ENABLE "Enable TBB" OFF)

message(STATUS "DPNP_ENABLE ${DPNP_ENABLE}")
message(STATUS "GPU_ENABLE ${GPU_ENABLE}")
message(STATUS "BUILD_TESTING ${BUILD_TESTING}")
message(STATUS "NUMBA_ENABLE ${NUMBA_ENABLE}")
message(STATUS "TBB_ENABLE ${TBB_ENABLE}")

include(CTest)

macro(apply_llvm_compile_flags target)
    if (MSVC)
        target_compile_options(${target} PRIVATE /EHsc)
    endif ()

    target_compile_definitions(${target} PRIVATE ${LLVM_DEFINITIONS})
endmacro()

# Check if using external LLVM/MLIR install tree
set(MLIR_EXT_USE_LLVM_INSTALL_TREE (DEFINED LLVM_DIR) AND (DEFINED MLIR_DIR))
if (DEFINED MLIR_DIR)
    if (NOT (DEFINED LLVM_DIR))
        message(FATAL_ERROR "Need to define both LLVM_DIR and MLIR_DIR or none.")
    endif()
elseif (DEFINED LLVM_DIR)
    message(FATAL_ERROR "Need to define both LLVM_DIR and MLIR_DIR or none.")
endif()

# Check if using external level-zero
set(MLIR_EXT_USE_EXT_LEVEL_ZERO (DEFINED LEVEL_ZERO_DIR))

add_subdirectory(third-party)

if(NOT MLIR_EXT_USE_LLVM_INSTALL_TREE)
    set(LLVM_DIR ${PROJECT_BINARY_DIR}/lib/cmake/llvm)
    message(STATUS "LLVM_DIR set to: ${PROJECT_BINARY_DIR}/lib/cmake/llvm")
    set(MLIR_DIR ${PROJECT_BINARY_DIR}/lib/cmake/mlir)
    message(STATUS "MLIR_DIR set to: ${PROJECT_BINARY_DIR}/lib/cmake/mlir")
endif()

if(NOT MLIR_EXT_USE_EXT_LEVEL_ZERO)
    set(LEVEL_ZERO_DIR ${PROJECT_BINARY_DIR})
    message(STATUS "LEVEL_ZERO_DIR set to: ${PROJECT_BINARY_DIR}")
endif()

if(GPU_ENABLE)
    add_subdirectory(dpcomp_gpu_runtime)
endif()

add_subdirectory(mlir)
add_subdirectory(dpcomp_runtime)
add_subdirectory(tools)

if(NUMBA_ENABLE)
    add_subdirectory(numba_dpcomp)
endif()

if(BUILD_TESTING)
    add_subdirectory(test)
endif()
