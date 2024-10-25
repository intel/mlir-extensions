//===-- Utils.h ------- XeGPU to VC Lowering pass --------------*- C++ -*-===//
//
// Copyright 2024 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines some utils used in ConversionToVC passes
///
//===----------------------------------------------------------------------===//

#ifndef VC_UTILS_H
#define VC_UTILS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

#define i1Ty rewriter.getI1Type()
#define i8Ty rewriter.getI8Type()
#define i16Ty rewriter.getI16Type()
#define i32Ty rewriter.getI32Type()
#define i64Ty rewriter.getI64Type()
#define indexTy rewriter.getIndexType()
#define vecTy(n, ty) VectorType::get(n, ty)

#define i1_val(value)                                                          \
  rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(value))

#define i8_val(value)                                                          \
  rewriter.create<arith::ConstantOp>(loc, rewriter.getI8IntegerAttr(value))

#define i16_val(value)                                                         \
  rewriter.create<arith::ConstantOp>(loc, rewriter.getI16IntegerAttr(value))

#define i32_val(value)                                                         \
  rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(value))

#define i64_val(value)                                                         \
  rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(value))

#define index_val(value)                                                       \
  rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(value))

#define constant_val(attr) rewriter.create<arith::ConstantOp>(loc, attr)

#define integer_val(value, type)                                               \
  rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(type, value))

#define dense_vector_int_val(value, elemTy, vecSize)                           \
  rewriter.create<arith::ConstantOp>(                                          \
      loc, DenseElementsAttr::get(vecTy(vecSize, elemTy),                      \
                                  IntegerAttr::get(elemTy, value)))

#define dense_vector_val(attr, vecTy)                                          \
  rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vecTy, attr))

/// This function adds necessary Func Declaration for Imported VC-intrinsics
/// functions and sets linkage attributes to those declaration
/// to support SPIRV compilation
FlatSymbolRefAttr getFuncRefAttr(gpu::GPUModuleOp module, StringRef name,
                                 TypeRange resultType, ValueRange operands,
                                 bool isVectorComputeFunction,
                                 bool emitCInterface,
                                 bool emitSPIRVLinkage = true);

/// Create VC-Intrinsics function declaration and insert func.call Op
func::CallOp createFuncCall(PatternRewriter &rewriter, Location loc,
                            StringRef funcName, TypeRange resultType,
                            ValueRange operands, bool emitCInterface);

#endif // XEGPU_VC_UTILS_H
