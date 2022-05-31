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

project(mlir-extensions LANGUAGES CXX C)

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

add_subdirectory(include/mlir-extensions/dialect/plier)
add_subdirectory(include/mlir-extensions/dialect/plier_util)
add_subdirectory(include/mlir-extensions/dialect/gpu_runtime/IR)

if (IMEX_ENABLE_IGPU_DIALECT)
    add_subdirectory(tools)
endif()

set(SOURCES_LIST
    lib/Conversion/SCFToAffine/SCFToAffine.cpp
    lib/Conversion/gpu_to_gpu_runtime.cpp
    lib/Conversion/gpu_runtime_to_llvm.cpp
    lib/analysis/memory_ssa.cpp
    lib/analysis/memory_ssa_analysis.cpp
    lib/compiler/compiler.cpp
    lib/compiler/pipeline_registry.cpp
    lib/dialect/gpu_runtime/IR/gpu_runtime_ops.cpp
    lib/dialect/plier/dialect.cpp
    lib/dialect/plier_util/dialect.cpp
    lib/transforms/arg_lowering.cpp
    lib/transforms/call_lowering.cpp
    lib/transforms/canonicalize_reductions.cpp
    lib/transforms/cast_lowering.cpp
    lib/transforms/cast_utils.cpp
    lib/transforms/common_opts.cpp
    lib/transforms/const_utils.cpp
    lib/transforms/cse.cpp
    lib/transforms/func_utils.cpp
    lib/transforms/if_rewrites.cpp
    lib/transforms/index_type_propagation.cpp
    lib/transforms/inline_utils.cpp
    lib/transforms/loop_rewrites.cpp
    lib/transforms/loop_utils.cpp
    lib/transforms/memory_rewrites.cpp
    lib/transforms/pipeline_utils.cpp
    lib/transforms/promote_bool_memref.cpp
    lib/transforms/promote_to_parallel.cpp
    lib/transforms/type_conversion.cpp
    lib/transforms/uplift_math.cpp
    lib/utils.cpp
    )
set(HEADERS_LIST
    include/mlir-extensions/Conversion/SCFToAffine/SCFToAffine.h
    include/mlir-extensions/Conversion/gpu_to_gpu_runtime.hpp
    include/mlir-extensions/Conversion/gpu_runtime_to_llvm.hpp
    include/mlir-extensions/analysis/memory_ssa.hpp
    include/mlir-extensions/analysis/memory_ssa_analysis.hpp
    include/mlir-extensions/compiler/compiler.hpp
    include/mlir-extensions/compiler/pipeline_registry.hpp
    include/mlir-extensions/dialect/gpu_runtime/IR/gpu_runtime_ops.hpp
    include/mlir-extensions/dialect/plier/dialect.hpp
    include/mlir-extensions/dialect/plier_util/dialect.hpp
    include/mlir-extensions/transforms/arg_lowering.hpp
    include/mlir-extensions/transforms/call_lowering.hpp
    include/mlir-extensions/transforms/canonicalize_reductions.hpp
    include/mlir-extensions/transforms/cast_lowering.hpp
    include/mlir-extensions/transforms/cast_utils.hpp
    include/mlir-extensions/transforms/common_opts.hpp
    include/mlir-extensions/transforms/const_utils.hpp
    include/mlir-extensions/transforms/cse.hpp
    include/mlir-extensions/transforms/func_utils.hpp
    include/mlir-extensions/transforms/if_rewrites.hpp
    include/mlir-extensions/transforms/index_type_propagation.hpp
    include/mlir-extensions/transforms/inline_utils.hpp
    include/mlir-extensions/transforms/loop_rewrites.hpp
    include/mlir-extensions/transforms/loop_utils.hpp
    include/mlir-extensions/transforms/memory_rewrites.hpp
    include/mlir-extensions/transforms/pipeline_utils.hpp
    include/mlir-extensions/transforms/promote_bool_memref.hpp
    include/mlir-extensions/transforms/promote_to_parallel.hpp
    include/mlir-extensions/transforms/rewrite_wrapper.hpp
    include/mlir-extensions/transforms/type_conversion.hpp
    include/mlir-extensions/transforms/uplift_math.hpp
    include/mlir-extensions/utils.hpp
    )

set(MLIR_EXTENSIONS_LIB "mlir-extensions")

add_library(${MLIR_EXTENSIONS_LIB} STATIC ${SOURCES_LIST} ${HEADERS_LIST})

apply_llvm_compile_flags(${MLIR_EXTENSIONS_LIB})

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG(-Winvalid-offsetof __offsetof_flag)
if (__offsetof_flag)
    set_source_files_properties(
        lib/analysis/memory_ssa.cpp
        PROPERTIES
        COMPILE_FLAGS "-Wno-invalid-offsetof"
    )
endif()
unset(__offsetof_flag)

target_link_libraries(${MLIR_EXTENSIONS_LIB} PRIVATE
    MLIRIR
    MLIRLLVMIR
    MLIRTransforms
    MLIRFuncTransforms
    MLIRLinalgTransforms
    MLIRTensorTransforms
    MLIRMathToSPIRV
    MLIRControlFlowToSPIRV
    )

target_include_directories(${MLIR_EXTENSIONS_LIB} SYSTEM PRIVATE
    ${MLIR_INCLUDE_DIRS}
    PRIVATE
    ./lib
    )

target_include_directories(${MLIR_EXTENSIONS_LIB} PUBLIC
    ./include
    ${PROJECT_BINARY_DIR}/include
    ${PROJECT_BINARY_DIR}/mlir-extensions/include
    )

add_dependencies(${MLIR_EXTENSIONS_LIB} MLIRPlierOpsIncGen MLIRPlierUtilOpsIncGen MLIRGpuRuntimeOpsIncGen)
