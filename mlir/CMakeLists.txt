# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
    lib/Analysis/MemorySsa.cpp
    lib/Analysis/MemorySsaAnalysis.cpp
    lib/Compiler/Compiler.cpp
    lib/Compiler/PipelineRegistry.cpp
    lib/Conversion/CfgToScf.cpp
    lib/Conversion/GpuRuntimeToLlvm.cpp
    lib/Conversion/GpuToGpuRuntime.cpp
    lib/Conversion/NtensorToLinalg.cpp
    lib/Conversion/NtensorToMemref.cpp
    lib/Conversion/SCFToAffine/SCFToAffine.cpp
    lib/Conversion/UtilConversion.cpp
    lib/Conversion/UtilToLlvm.cpp
    lib/Dialect/gpu_runtime/IR/GpuRuntimeOps.cpp
    lib/Dialect/gpu_runtime/Transforms/MakeBarriersUniform.cpp
    lib/Dialect/imex_util/Dialect.cpp
    lib/Dialect/ntensor/IR/NTensorOps.cpp
    lib/Dialect/ntensor/Transforms/PropagateEnvironment.cpp
    lib/Dialect/ntensor/Transforms/ResolveArrayOps.cpp
    lib/Dialect/plier/Dialect.cpp
    lib/ExecutionEngine/ExecutionEngine.cpp
    lib/Transforms/ArgLowering.cpp
    lib/Transforms/CallLowering.cpp
    lib/Transforms/CanonicalizeReductions.cpp
    lib/Transforms/CastLowering.cpp
    lib/Transforms/CastUtils.cpp
    lib/Transforms/CommonOpts.cpp
    lib/Transforms/ConstUtils.cpp
    lib/Transforms/Cse.cpp
    lib/Transforms/ExpandTuple.cpp
    lib/Transforms/FuncUtils.cpp
    lib/Transforms/IfRewrites.cpp
    lib/Transforms/IndexTypePropagation.cpp
    lib/Transforms/InlineUtils.cpp
    lib/Transforms/LoopRewrites.cpp
    lib/Transforms/LoopUtils.cpp
    lib/Transforms/MakeSignless.cpp
    lib/Transforms/MemoryRewrites.cpp
    lib/Transforms/PipelineUtils.cpp
    lib/Transforms/PromoteBoolMemref.cpp
    lib/Transforms/PromoteToParallel.cpp
    lib/Transforms/TypeConversion.cpp
    lib/Transforms/UpliftMath.cpp
    lib/Utils.cpp
    )
set(HEADERS_LIST
    include/imex/Analysis/MemorySsa.hpp
    include/imex/Analysis/MemorySsaAnalysis.hpp
    include/imex/Compiler/Compiler.hpp
    include/imex/Compiler/PipelineRegistry.hpp
    include/imex/Conversion/CfgToScf.hpp
    include/imex/Conversion/GpuRuntimeToLlvm.hpp
    include/imex/Conversion/GpuToGpuRuntime.hpp
    include/imex/Conversion/NtensorToLinalg.hpp
    include/imex/Conversion/NtensorToMemref.hpp
    include/imex/Conversion/SCFToAffine/SCFToAffine.h
    include/imex/Conversion/UtilConversion.hpp
    include/imex/Conversion/UtilToLlvm.hpp
    include/imex/Dialect/gpu_runtime/IR/GpuRuntimeOps.hpp
    include/imex/Dialect/gpu_runtime/Transforms/MakeBarriersUniform.hpp
    include/imex/Dialect/imex_util/Dialect.hpp
    include/imex/Dialect/imex_util/Utils.hpp
    include/imex/Dialect/ntensor/IR/NTensorOps.hpp
    include/imex/Dialect/ntensor/Transforms/PropagateEnvironment.hpp
    include/imex/Dialect/ntensor/Transforms/ResolveArrayOps.hpp
    include/imex/Dialect/plier/Dialect.hpp
    include/imex/ExecutionEngine/ExecutionEngine.hpp
    include/imex/Transforms/ArgLowering.hpp
    include/imex/Transforms/CallLowering.hpp
    include/imex/Transforms/CanonicalizeReductions.hpp
    include/imex/Transforms/CastLowering.hpp
    include/imex/Transforms/CastUtils.hpp
    include/imex/Transforms/CommonOpts.hpp
    include/imex/Transforms/ConstUtils.hpp
    include/imex/Transforms/Cse.hpp
    include/imex/Transforms/ExpandTuple.hpp
    include/imex/Transforms/FuncUtils.hpp
    include/imex/Transforms/IfRewrites.hpp
    include/imex/Transforms/IndexTypePropagation.hpp
    include/imex/Transforms/InlineUtils.hpp
    include/imex/Transforms/LoopRewrites.hpp
    include/imex/Transforms/LoopUtils.hpp
    include/imex/Transforms/MakeSignless.hpp
    include/imex/Transforms/MemoryRewrites.hpp
    include/imex/Transforms/PipelineUtils.hpp
    include/imex/Transforms/PromoteBoolMemref.hpp
    include/imex/Transforms/PromoteToParallel.hpp
    include/imex/Transforms/RewriteWrapper.hpp
    include/imex/Transforms/TypeConversion.hpp
    include/imex/Transforms/UpliftMath.hpp
    include/imex/Utils.hpp
    )

set(MLIR_EXTENSIONS_LIB "mlir-extensions")

add_library(${MLIR_EXTENSIONS_LIB} STATIC ${SOURCES_LIST} ${HEADERS_LIST})

apply_llvm_compile_flags(${MLIR_EXTENSIONS_LIB})

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG(-Winvalid-offsetof __offsetof_flag)
if (__offsetof_flag)
    set_source_files_properties(
        lib/Analysis/MemorySsa.cpp
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
