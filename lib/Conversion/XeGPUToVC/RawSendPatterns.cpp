//===-- RawSendPatterns.cpp - XeGPU to VC Lowering pass ---------*- C++ -*-===//
//
// Copyright 2024 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements patterns to lower load/store to RawSend messages
///
//===----------------------------------------------------------------------===//

#include <imex/Conversion/XeGPUToVC/XeGPUToVC.h>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"

#include "Utils.h"

using namespace mlir;
using mlir::xegpu::NbarrierArriveOp;

namespace imex {
namespace RawSend {

class NbarrierArrivePattern : public OpConversionPattern<NbarrierArriveOp> {
public:
  using OpConversionPattern<NbarrierArriveOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(NbarrierArriveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto payload = adaptor.getNbarrier();

    std::string funcName = "llvm.genx.raw.send2.noresult.i1.v8i32";

    // desc format
    Value modifier = i8_val(0);
    Value exec_size = i8_val(0);
    Value predicate = i1_val(1);
    Value numsrc1 = i8_val(1); // register nums of payload
    Value sfid = i8_val(3);
    Value etDesc = i32_val(0);
    Value msg_desc = i32_val(0x2000004);

    SmallVector<Value> args{modifier, exec_size, predicate, numsrc1,
                            sfid,     etDesc,    msg_desc,  payload};

    createFuncCall(rewriter, loc, funcName, TypeRange{}, args, false);

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace RawSend

void populateNbarrierArriveRawSendPatterns(TypeConverter &converter,
                                           RewritePatternSet &patterns) {
  patterns.add<RawSend::NbarrierArrivePattern>(patterns.getContext());
}

} // namespace imex
