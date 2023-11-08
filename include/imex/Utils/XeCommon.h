
//===- XeUtils.h - XeTile/XeGPU Utility Functions --------------------*- C++
//-*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines utility functions used by XeTile/XeGPU dialects.
//
//===----------------------------------------------------------------------===//

#ifndef _IMEX_XECOMMON_H_
#define _IMEX_XECOMMON_H_

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/OneToNTypeConversion.h>

namespace imex {

/**
 * None: Not associated with dpas
 * DPASA: asscociated with dpas A operand
 * DPASB: asscociated with dpas B operand
 * DPASC: asscociated with dpas C operand
 * DPASR: asscociated with dpas result value.
 */
enum class OperandType { None = 0, DPASA = 1, DPASB = 2, DPASC = 4, DPASR = 8 };

class ValueAttributeMap {
public:
  ValueAttributeMap() {}

  void add(mlir::BlockArgument arg, imex::OperandType type);
  void add(mlir::Operation *op, imex::OperandType type);

  imex::OperandType get(mlir::Operation *op);
  imex::OperandType get(mlir::BlockArgument arg);

private:
  llvm::DenseMap<mlir::Operation *, int> operationMap;
  llvm::DenseMap<mlir::BlockArgument, int> argumentMap;
};

void markDefChainValues(mlir::Value value, imex::OperandType type,
                        imex::ValueAttributeMap &map);
void markUseChainValues(mlir::Value value, imex::OperandType type,
                        imex::ValueAttributeMap &map);

mlir::ValueRange buildUnrealizedCast(mlir::OpBuilder &builder,
                                     mlir::TypeRange resultTypes,
                                     mlir::ValueRange inputs);

class XeTypeConverter : public mlir::OneToNTypeConverter {
public:
  using mlir::OneToNTypeConverter::convertType;

  XeTypeConverter(mlir::MLIRContext &context, imex::ValueAttributeMap &map)
      : context(context), map(map) {
    addConversion([&](xetile::TileType tileTy,
                      llvm::SmallVectorImpl<mlir::Type> &resultTypes)
                      -> std::optional<mlir::LogicalResult> {
      return convertTileType(tileTy, resultTypes);
    });

    addConversion([&](mlir::VectorType vectorTy,
                      llvm::SmallVectorImpl<mlir::Type> &resultTypes)
                      -> std::optional<mlir::LogicalResult> {
      return convertVectorType(vectorTy, resultTypes);
    });
  }

  virtual std::optional<mlir::LogicalResult>
  convertTileType(xetile::TileType tileTy,
                  llvm::SmallVectorImpl<mlir::Type> &resultTypes) {
    llvm_unreachable("Pending Implementation for convertTileType.");
  }

  virtual std::optional<mlir::LogicalResult>
  convertVectorType(mlir::VectorType vectorTy,
                    llvm::SmallVectorImpl<mlir::Type> &resultTypes) {
    llvm_unreachable("Pending Implementation for convertVectorType.");
  }

  imex::OperandType get(mlir::Operation *op) { return map.get(op); }

  imex::OperandType get(mlir::BlockArgument arg) { return map.get(arg); }

  bool isA(mlir::Operation *op) {
    return (int(map.get(op)) & int(imex::OperandType::DPASA));
  }

  bool isA(mlir::BlockArgument arg) {
    return (int(map.get(arg)) & int(imex::OperandType::DPASA));
  }

  bool isAOnly(mlir::Operation *op) { return isA(op) && !isB(op) && !isRC(op); }

  bool isAOnly(mlir::BlockArgument arg) {
    return isA(arg) && !isB(arg) && !isRC(arg);
  }

  bool isB(mlir::Operation *op) {
    return (int(map.get(op)) & int(imex::OperandType::DPASB));
  }

  bool isB(mlir::BlockArgument arg) {
    return (int(map.get(arg)) & int(imex::OperandType::DPASB));
  }

  bool isBOnly(mlir::Operation *op) { return isB(op) && !isA(op) && !isRC(op); }

  bool isBOnly(mlir::BlockArgument arg) {
    return isB(arg) && !isB(arg) && !isRC(arg);
  }

  bool isRC(mlir::Operation *op) {
    return int(map.get(op)) &
           (int(imex::OperandType::DPASC) | int(imex::OperandType::DPASR));
  }

  bool isRC(mlir::BlockArgument arg) {
    return int(map.get(arg)) &
           (int(imex::OperandType::DPASC) | int(imex::OperandType::DPASR));
  }

  bool isRCOnly(mlir::Operation *op) {
    return isRC(op) && !isA(op) && !isB(op);
  }

  bool isRCOnly(mlir::BlockArgument arg) {
    return isRC(arg) && !isA(arg) && !isB(arg);
  }

private:
  mlir::MLIRContext &context;
  imex::ValueAttributeMap &map;
};

// A simple mlir::RewritePattern wrapper with methods for accessing OperandType
class XeConversionPattern : public mlir::RewritePattern {
public:
  using mlir::RewritePattern::RewritePattern;

  template <typename... Args>
  XeConversionPattern(imex::XeTypeConverter &typeConverter, Args &&...args)
      : mlir::RewritePattern(std::forward<Args>(args)...),
        typeConverter(typeConverter) {}

  virtual mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    llvm_unreachable("must override matchAndRewrite or a rewrite method");
  };

  imex::OperandType getOperandType(mlir::Operation *op) const {
    return typeConverter.get(op);
  }

  imex::OperandType getOperandType(mlir::BlockArgument arg) const {
    return typeConverter.get(arg);
  }

  bool isA(mlir::Operation *op) const { return typeConverter.isAOnly(op); }

  bool isA(mlir::BlockArgument arg) const { return typeConverter.isAOnly(arg); }

  bool isB(mlir::Operation *op) const { return typeConverter.isBOnly(op); }

  bool isB(mlir::BlockArgument arg) const { return typeConverter.isBOnly(arg); }

  bool isRC(mlir::Operation *op) const { return typeConverter.isRCOnly(op); }

  bool isRC(mlir::BlockArgument arg) const {
    return typeConverter.isRCOnly(arg);
  }

  imex::XeTypeConverter &getTypeConverter() const { return typeConverter; }

  template <typename ConverterTy>
  std::enable_if_t<std::is_base_of<mlir::TypeConverter, ConverterTy>::value,
                   ConverterTy &>
  getTypeConverter() const {
    return static_cast<ConverterTy &>(typeConverter);
  }

protected:
  imex::XeTypeConverter &typeConverter;
};

} // namespace imex

#endif
