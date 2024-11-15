//===- AllExtensions.cpp - All NDArray Dialect Extensions ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "imex/Dialect/NDArray/Extensions/AllExtensions.h"
#include "imex/Dialect/NDArray/Extensions/MeshShardingExtensions.h"

using namespace mlir;

void imex::ndarray::registerAllExtensions(DialectRegistry &registry) {
  registerShardingInterfaceExternalModels(registry);
}