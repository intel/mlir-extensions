// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

namespace mlir {
class Value;
class Location;
class OpBuilder;
class Type;
class IntegerType;
} // namespace mlir

namespace imex {
mlir::Value indexCast(mlir::OpBuilder &builder, mlir::Location loc,
                      mlir::Value src, mlir::Type dst_type);
mlir::Value indexCast(mlir::OpBuilder &builder, mlir::Location loc,
                      mlir::Value src);

mlir::Type makeSignlessType(mlir::Type type);
mlir::IntegerType makeSignlessType(mlir::IntegerType type);
} // namespace imex
