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

cmake_minimum_required(VERSION 3.20)

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

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/")

# SDL
# set(CMAKE_VERBOSE_MAKEFILE on) # enable for debug
if (CMAKE_SYSTEM_NAME STREQUAL Windows)
    add_compile_options(-GS -D_CRT_SECURE_NO_WARNINGS)
    add_link_options(-DYNAMICBASE -NXCOMPAT -GUARD:CF)
    # add_link_options(-INTEGRITYCHECK) # require signatures of libs, only recommended
endif()
if (CMAKE_SYSTEM_NAME STREQUAL Linux)
    string(CONCAT WARN_FLAGS
        "-Wall "
        "-Wextra "
        "-Winit-self "
        "-Wunused-function "
        "-Wuninitialized "
        "-Wmissing-declarations "
        "-fdiagnostics-color=auto "
        "-Wno-deprecated-declarations "
    )
    string(CONCAT SDL_FLAGS
        "-D_FORTIFY_SOURCE=2 "
        "-Wformat "
        "-Wformat-security "
        "-Werror=format-security "
        "-fno-delete-null-pointer-checks "
        "-fstack-protector-strong  "
        "-fno-strict-overflow "
        "-fstack-clash-protection "
        "-fcf-protection=full "
    )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WARN_FLAGS} ${SDL_FLAGS}")
    # add_compile_options(-mcet) # v8.0 and newer # unrecognized command line option '-mcet', only recommended
    add_link_options(-Wl,-z,noexecstack,-z,relro,-z,now)
endif()
if (CMAKE_SYSTEM_NAME STREQUAL Darwin)
    add_compile_options(-D_FORTIFY_SOURCE=2 -Wformat -Wformat-security -Werror=format-security -fno-delete-null-pointer-checks -fstack-protector-strong -fno-strict-overflow -Wall)
    add_compile_options(-fcf-protection=full) # v8.0 and newer
    # add_compile_options(-mcet) # v8.0 and newer # unrecognized command line option '-mcet', only recommended
endif()

option(IMEX_USE_DPNP
    "Use the dpnp Python NumPy-like library for some math functions"
    OFF
)
option(IMEX_ENABLE_IGPU_DIALECT
    "Enable GPU codegen"
    OFF
)
option(IMEX_ENABLE_NUMBA_FE
    "Enable numba-based python frontend"
    OFF
)
option(IMEX_ENABLE_TBB_SUPPORT
    "Enable TBB"
    OFF
)
option(IMEX_ENABLE_TESTS
    "Enable CTests"
    OFF
)
option(IMEX_ENABLE_NUMBA_HOTFIX
    "Enable hotfix for numba dpcomp"
    OFF
)

message(STATUS "IMEX_USE_DPNP ${DPNIMEX_USE_DPNPP_ENABLE}")
message(STATUS "IMEX_ENABLE_IGPU_DIALECT ${IMEX_ENABLE_IGPU_DIALECT}")
message(STATUS "IMEX_ENABLE_TESTS ${IMEX_ENABLE_TESTS}")
message(STATUS "IMEX_ENABLE_NUMBA_FE ${IMEX_ENABLE_NUMBA_FE}")
message(STATUS "IMEX_ENABLE_NUMBA_HOTFIX ${IMEX_ENABLE_NUMBA_HOTFIX}")
message(STATUS "IMEX_ENABLE_TBB_SUPPORT ${IMEX_ENABLE_TBB_SUPPORT}")

macro(apply_llvm_compile_flags target)
    if (MSVC)
        target_compile_options(${target} PRIVATE /EHsc)
    endif ()
    target_compile_definitions(${target} PRIVATE ${LLVM_DEFINITIONS})
endmacro()

if(IMEX_ENABLE_IGPU_DIALECT)
    if(DEFINED ENV{LEVEL_ZERO_VERSION_CHECK_OFF})
        find_package(LevelZero REQUIRED)
    else()
        find_package(LevelZero 1.6 REQUIRED)
    endif()
    add_subdirectory(dpcomp_gpu_runtime)
endif()

if(IMEX_ENABLE_NUMBA_HOTFIX)
    add_definitions(-DIMEX_ENABLE_NUMBA_HOTFIX)
endif()

add_subdirectory(mlir)
add_subdirectory(dpcomp_runtime)
add_subdirectory(tools)

if(IMEX_ENABLE_NUMBA_FE)
    add_subdirectory(numba_dpcomp)
endif()

if(IMEX_ENABLE_TESTS)
    include(CTest)
    add_subdirectory(test)
endif()
