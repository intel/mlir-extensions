// Copyright 2022 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <llvm/Support/FormatVariadic.h>
#include <mlir/Analysis/BufferViewFlowAnalysis.h>
#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/ArithmeticToSPIRV/ArithmeticToSPIRV.h>
#include <mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h>
#include <mlir/Conversion/GPUCommon/GPUCommonPass.h>
#include <mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h>
#include <mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/MathToSPIRV/MathToSPIRV.h>
#include <mlir/Conversion/SCFToGPU/SCFToGPUPass.h>
#include <mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h>
#include <mlir/Conversion/StandardToSPIRV/StandardToSPIRV.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arithmetic/Transforms/Passes.h>
#include <mlir/Dialect/GPU/ParallelLoopMapper.h>
#include <mlir/Dialect/GPU/Passes.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <mlir/Dialect/SPIRV/IR/TargetAndABI.h>
#include <mlir/Dialect/SPIRV/Transforms/Passes.h>
#include <mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/Dominance.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/SPIRV/Serialization.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

#include "mlir-extensions/dialect/gpu_runtime/IR/gpu_runtime_ops.hpp"
#include "mlir-extensions/transforms/func_utils.hpp"

namespace gpu_runtime {

std::unique_ptr<mlir::Pass> createEnumerateEventsPass();
std::unique_ptr<mlir::Pass> createGPUToLLVMPass();

} // namespace gpu_runtime
