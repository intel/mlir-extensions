//===- MeshShardingExtensions.h - -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_NDARRAY_EXTENSIONS_SHARDINGEXTENSIONS_H_
#define MLIR_DIALECT_NDARRAY_EXTENSIONS_SHARDINGEXTENSIONS_H_

namespace mlir {
class DialectRegistry;
}

namespace imex {
namespace ndarray {

void registerShardingInterfaceExternalModels(mlir::DialectRegistry &registry);

} // namespace ndarray
} // namespace imex

#endif // MLIR_DIALECT_NDARRAY_EXTENSIONS_SHARDINGEXTENSIONS_H_
