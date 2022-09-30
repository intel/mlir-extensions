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

add_subdirectory(include/imex/Dialect/plier)
add_subdirectory(include/imex/Dialect/imex_util)
add_subdirectory(include/imex/Dialect/gpu_runtime/IR)
add_subdirectory(include/imex/Dialect/ntensor/IR)

set(SOURCES_LIST
    lib/Conversion/SCFToAffine/SCFToAffine.cpp
    lib/Conversion/cfg_to_scf.cpp
    lib/Conversion/gpu_runtime_to_llvm.cpp
    lib/Conversion/gpu_to_gpu_runtime.cpp
    lib/Conversion/util_conversion.cpp
    lib/Conversion/util_to_llvm.cpp
    lib/Dialect/gpu_runtime/IR/gpu_runtime_ops.cpp
    lib/Dialect/imex_util/dialect.cpp
    lib/Dialect/ntensor/IR/NTensorOps.cpp
    lib/Dialect/ntensor/Transforms/ResolveArrayOps.cpp
    lib/Dialect/plier/dialect.cpp
    lib/ExecutionEngine/execution_engine.cpp
    lib/Transforms/arg_lowering.cpp
    lib/Transforms/call_lowering.cpp
    lib/Transforms/canonicalize_reductions.cpp
    lib/Transforms/cast_lowering.cpp
    lib/Transforms/cast_utils.cpp
    lib/Transforms/common_opts.cpp
    lib/Transforms/const_utils.cpp
    lib/Transforms/cse.cpp
    lib/Transforms/expand_tuple.cpp
    lib/Transforms/func_utils.cpp
    lib/Transforms/if_rewrites.cpp
    lib/Transforms/index_type_propagation.cpp
    lib/Transforms/inline_utils.cpp
    lib/Transforms/loop_rewrites.cpp
    lib/Transforms/loop_utils.cpp
    lib/Transforms/memory_rewrites.cpp
    lib/Transforms/pipeline_utils.cpp
    lib/Transforms/promote_bool_memref.cpp
    lib/Transforms/promote_to_parallel.cpp
    lib/Transforms/type_conversion.cpp
    lib/Transforms/uplift_math.cpp
    lib/analysis/memory_ssa.cpp
    lib/analysis/memory_ssa_analysis.cpp
    lib/compiler/compiler.cpp
    lib/compiler/pipeline_registry.cpp
    lib/utils.cpp
    )
set(HEADERS_LIST
    include/imex/Conversion/SCFToAffine/SCFToAffine.h
    include/imex/Conversion/cfg_to_scf.hpp
    include/imex/Conversion/gpu_runtime_to_llvm.hpp
    include/imex/Conversion/gpu_to_gpu_runtime.hpp
    include/imex/Conversion/util_conversion.hpp
    include/imex/Conversion/util_to_llvm.hpp
    include/imex/Dialect/gpu_runtime/IR/gpu_runtime_ops.hpp
    include/imex/Dialect/imex_util/dialect.hpp
    include/imex/Dialect/ntensor/IR/NTensorOps.hpp
    include/imex/Dialect/ntensor/Transforms/ResolveArrayOps.hpp
    include/imex/Dialect/plier/dialect.hpp
    include/imex/ExecutionEngine/execution_engine.hpp
    include/imex/Transforms/arg_lowering.hpp
    include/imex/Transforms/call_lowering.hpp
    include/imex/Transforms/canonicalize_reductions.hpp
    include/imex/Transforms/cast_lowering.hpp
    include/imex/Transforms/cast_utils.hpp
    include/imex/Transforms/common_opts.hpp
    include/imex/Transforms/const_utils.hpp
    include/imex/Transforms/cse.hpp
    include/imex/Transforms/expand_tuple.hpp
    include/imex/Transforms/func_utils.hpp
    include/imex/Transforms/if_rewrites.hpp
    include/imex/Transforms/index_type_propagation.hpp
    include/imex/Transforms/inline_utils.hpp
    include/imex/Transforms/loop_rewrites.hpp
    include/imex/Transforms/loop_utils.hpp
    include/imex/Transforms/memory_rewrites.hpp
    include/imex/Transforms/pipeline_utils.hpp
    include/imex/Transforms/promote_bool_memref.hpp
    include/imex/Transforms/promote_to_parallel.hpp
    include/imex/Transforms/rewrite_wrapper.hpp
    include/imex/Transforms/type_conversion.hpp
    include/imex/Transforms/uplift_math.hpp
    include/imex/analysis/memory_ssa.hpp
    include/imex/analysis/memory_ssa_analysis.hpp
    include/imex/compiler/compiler.hpp
    include/imex/compiler/pipeline_registry.hpp
    include/imex/utils.hpp
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
    MLIRControlFlowToSPIRV
    MLIRFuncTransforms
    MLIRIR
    MLIRLLVMDialect
    MLIRLinalgTransforms
    MLIRMathToSPIRV
    MLIRTensorTransforms
    MLIRTransforms
    )

target_include_directories(${MLIR_EXTENSIONS_LIB} SYSTEM PRIVATE
    ${MLIR_INCLUDE_DIRS}
    PRIVATE
    ./lib
    )

target_include_directories(${MLIR_EXTENSIONS_LIB} PUBLIC
    ./include
    ${PROJECT_BINARY_DIR}/mlir/include
    ${PROJECT_BINARY_DIR}/imex/include
    )

add_dependencies(${MLIR_EXTENSIONS_LIB} MLIRPlierOpsIncGen MLIRImexUtilOpsIncGen MLIRGpuRuntimeOpsIncGen MLIRNTensorOpsIncGen)

add_subdirectory(tools)
if(IMEX_ENABLE_TESTS)
    add_subdirectory(test)
endif()
