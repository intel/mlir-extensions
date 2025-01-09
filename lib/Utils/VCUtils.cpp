//===-- Utils.cpp - XeGPU to VC Lowering pass ---------*- C++ -*-===//
//
// Copyright 2024 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements some utils used in ConversionToVC passes
///
//===----------------------------------------------------------------------===//

#include "imex/Utils/VCUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

/// This function adds necessary Func Declaration for Imported VC-intrinsics
/// functions and sets linkage attributes to those declaration
/// to support SPIRV compilation
FlatSymbolRefAttr getFuncRefAttr(gpu::GPUModuleOp module, StringRef name,
                                 TypeRange resultType, ValueRange operands,
                                 bool isVectorComputeFunction,
                                 bool emitCInterface,
                                 bool emitSPIRVLinkage /* = true*/) {
  MLIRContext *context = module.getContext();
  auto result = SymbolRefAttr::get(context, name);

  // Look up for existing VC instrinsics Function declaration and
  // create it if not present Module level
  auto func = module.lookupSymbol<func::FuncOp>(result.getAttr());
  if (!func) {
    OpBuilder moduleBuilder(module.getBodyRegion());
    func = moduleBuilder.create<func::FuncOp>(
        module.getLoc(), name,
        FunctionType::get(context, operands.getTypes(), resultType));

    func.setPrivate();
    if (emitCInterface)
      func->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                    UnitAttr::get(context));

    // Set spirv attributes.
    // Set VectorComputeFunctionINTEL atribute if it is a VectorComputeFunction.
    if (isVectorComputeFunction)
      func->setAttr(spirv::stringifyDecoration(
                        spirv::Decoration::VectorComputeFunctionINTEL),
                    UnitAttr::get(context));
    // Emit linkage attributes needed for SPIR-V dialect path
    if (emitSPIRVLinkage) {
      auto linkageNameAttr = StringAttr::get(context, name);
      auto linkageTypeAttr =
          spirv::LinkageTypeAttr::get(context, spirv::LinkageType::Import);
      auto linkageAttribute = spirv::LinkageAttributesAttr::get(
          context, linkageNameAttr, linkageTypeAttr);
      func->setAttr("linkage_attributes", linkageAttribute);
    }
  }
  return result;
}

/// Create VC-Intrinsics function declaration and insert func.call Op
func::CallOp createFuncCall(PatternRewriter &rewriter, Location loc,
                            StringRef funcName, TypeRange resultType,
                            ValueRange operands, bool emitCInterface) {
  auto module =
      rewriter.getBlock()->getParentOp()->getParentOfType<gpu::GPUModuleOp>();
  FlatSymbolRefAttr fn = getFuncRefAttr(
      module, funcName, resultType, operands,
      true /*isVectorComputeFunctionINTEL=true*/, emitCInterface);
  return rewriter.create<func::CallOp>(loc, fn, resultType, operands);
}

Value getOffsetInUnitOfBytes(PatternRewriter &rewriter, Location loc,
                             Type addrTy, Value offset, unsigned eTyBitWidth) {
  if (eTyBitWidth >= 8) {
    unsigned eTyBytes = eTyBitWidth / 8;
    Value factor = integer_val(eTyBytes, addrTy);
    return muli(offset, factor);
  } else {
    Value eight = integer_val(8, addrTy);
    Value bw = integer_val(eTyBitWidth, addrTy);
    return divi(muli(offset, bw), eight);
  }
}

Value getVecOffsetInUnitOfBytes(PatternRewriter &rewriter, Location loc,
                                unsigned vecSize, Type addrTy, Value offset,
                                unsigned eTyBitWidth) {
  if (eTyBitWidth >= 8) {
    unsigned eTyBytes = eTyBitWidth / 8;
    Value factor = dense_vector_int_val(eTyBytes, addrTy, vecSize);
    return muli(offset, factor);
  } else {
    Value eight = dense_vector_int_val(8, addrTy, vecSize);
    Value bw = dense_vector_int_val(eTyBitWidth, addrTy, vecSize);
    return divi(muli(offset, bw), eight);
  }
}
