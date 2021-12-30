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

project(dpcomp LANGUAGES CXX C)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/../llvm-sha.txt EXPECTED_LLVM_SHA)
message(STATUS "Expected llvm sha: \"${EXPECTED_LLVM_SHA}\"")

file(STRINGS ${LLVM_INCLUDE_DIR}/llvm/Support/VCSRevision.h REVISION_FILE_DATA)
message(DEBUG "VCSRevision: ${REVISION_FILE_DATA}")
string(REGEX MATCH "\"([^\"]*)\"" LLVM_SHA ${REVISION_FILE_DATA})
string(REPLACE "\"" "" LLVM_SHA ${LLVM_SHA})
message(STATUS "llvm sha: \"${LLVM_SHA}\"")

if (NOT EXPECTED_LLVM_SHA STREQUAL LLVM_SHA)
    message(FATAL_ERROR "Invalid llvm version")
endif()

add_subdirectory(include/plier)

set(SOURCES_LIST
    src/Conversion/SCFToAffine/SCFToAffine.cpp
    src/analysis/memory_ssa.cpp
    src/analysis/memory_ssa_analysis.cpp
    src/compiler/compiler.cpp
    src/compiler/pipeline_registry.cpp
    src/dialect.cpp
    src/transforms/arg_lowering.cpp
    src/transforms/call_lowering.cpp
    src/transforms/canonicalize_reductions.cpp
    src/transforms/cast_lowering.cpp
    src/transforms/cast_utils.cpp
    src/transforms/common_opts.cpp
    src/transforms/const_utils.cpp
    src/transforms/cse.cpp
    src/transforms/func_utils.cpp
    src/transforms/if_rewrites.cpp
    src/transforms/index_type_propagation.cpp
    src/transforms/inline_utils.cpp
    src/transforms/loop_rewrites.cpp
    src/transforms/loop_utils.cpp
    src/transforms/memory_rewrites.cpp
    src/transforms/pipeline_utils.cpp
    src/transforms/promote_to_parallel.cpp
    src/transforms/type_conversion.cpp
    src/transforms/uplift_math_calls.cpp
    src/utils.cpp
    )
set(HEADERS_LIST
    include/plier/Conversion/SCFToAffine/SCFToAffine.h
    include/plier/PlierOps.td
    include/plier/analysis/memory_ssa.hpp
    include/plier/analysis/memory_ssa_analysis.hpp
    include/plier/compiler/compiler.hpp
    include/plier/compiler/pipeline_registry.hpp
    include/plier/dialect.hpp
    include/plier/transforms/arg_lowering.hpp
    include/plier/transforms/call_lowering.hpp
    include/plier/transforms/canonicalize_reductions.hpp
    include/plier/transforms/cast_lowering.hpp
    include/plier/transforms/cast_utils.hpp
    include/plier/transforms/common_opts.hpp
    include/plier/transforms/const_utils.hpp
    include/plier/transforms/cse.hpp
    include/plier/transforms/func_utils.hpp
    include/plier/transforms/if_rewrites.hpp
    include/plier/transforms/index_type_propagation.hpp
    include/plier/transforms/inline_utils.hpp
    include/plier/transforms/loop_rewrites.hpp
    include/plier/transforms/loop_utils.hpp
    include/plier/transforms/memory_rewrites.hpp
    include/plier/transforms/pipeline_utils.hpp
    include/plier/transforms/promote_to_parallel.hpp
    include/plier/transforms/rewrite_wrapper.hpp
    include/plier/transforms/type_conversion.hpp
    include/plier/transforms/uplift_math_calls.hpp
    include/plier/utils.hpp
    )

set(PLIER_LIB "plier")

add_library(${PLIER_LIB} STATIC ${SOURCES_LIST} ${HEADERS_LIST})

apply_llvm_compile_flags(${PLIER_LIB})

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG(-Winvalid-offsetof __offsetof_flag)
if (__offsetof_flag)
    set_source_files_properties(
        src/analysis/memory_ssa.cpp
        PROPERTIES
        COMPILE_FLAGS "-Wno-invalid-offsetof"
    )
endif()
unset(__offsetof_flag)

target_link_libraries(${PLIER_LIB} PRIVATE
    MLIRIR
    MLIRLLVMIR
    MLIRTransforms
    MLIRStandardOpsTransforms
    MLIRLinalgTransforms
    MLIRSCFToStandard
    MLIRTensorTransforms
    )

target_include_directories(${PLIER_LIB} SYSTEM PRIVATE
    ${MLIR_INCLUDE_DIRS}
    PRIVATE
    ./src
    )

target_include_directories(${PLIER_LIB} PUBLIC
    ./include
    ${PROJECT_BINARY_DIR}/include
    ${PROJECT_BINARY_DIR}/plier/include
    )

add_dependencies(${PLIER_LIB} MLIRPlierOpsIncGen)
