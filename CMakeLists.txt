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

cmake_minimum_required(VERSION 3.5)

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
option(NUMBA_ENABLE "Enable Numba" ON)

message(STATUS "DPNP_ENABLE ${DPNP_ENABLE}")
message(STATUS "GPU_ENABLE ${GPU_ENABLE}")
message(STATUS "BUILD_TESTING ${BUILD_TESTING}")

include(CTest)

macro(apply_llvm_compile_flags target)
    if (MSVC)
        target_compile_options(${target} PRIVATE /EHsc)
    endif ()

    target_compile_definitions(${target} PRIVATE ${LLVM_DEFINITIONS})
endmacro()

if(GPU_ENABLE)
    add_subdirectory(dpcomp_gpu_runtime)
endif()

add_subdirectory(mlir)
add_subdirectory(dpcomp_runtime)

if(NUMBA_ENABLE)
    add_subdirectory(numba_dpcomp)
    add_subdirectory(dpcomp_math_runtime)
    add_subdirectory(tools)
endif()

if(BUILD_TESTING)
    add_subdirectory(test)
endif()
