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

project(dpcomp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

# SDL
# set(CMAKE_VERBOSE_MAKEFILE on) # enable for debug
if (CMAKE_SYSTEM_NAME STREQUAL Windows)
    add_compile_options(-GS)
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

include(CTest)

macro(apply_llvm_compile_flags target)
    if (MSVC)
        target_compile_options(${target} PRIVATE /EHsc)
    endif ()

    target_compile_definitions(${target} PRIVATE ${LLVM_DEFINITIONS})
endmacro()

add_subdirectory(dpcomp)
add_subdirectory(dpcomp_math_runtime)
add_subdirectory(dpcomp_runtime)
add_subdirectory(tools)
add_subdirectory(numba_dpcomp/mlir_compiler)

if(${BUILD_TESTING})
    add_subdirectory(test)
endif()

set(CMAKE_INSTALL_BINDIR .)
set(CMAKE_INSTALL_LIBDIR .)
set(CMAKE_INSTALL_INCLUDEDIR .)

install(TARGETS dpcomp-runtime dpcomp-math-runtime mlir_compiler
        PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
        LIBRARY DESTINATION .
        )
