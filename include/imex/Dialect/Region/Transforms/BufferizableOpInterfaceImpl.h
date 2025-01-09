//===- BufferizableOpInterfaceImpl.h - Impl. of BufferizableOpInterface ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef IMEX_DIALECT_REGION_BUFFERIZABLEOPINTERFACEIMPL_H
#define IMEX_DIALECT_REGION_BUFFERIZABLEOPINTERFACEIMPL_H

#include <mlir/IR/MLIRContext.h>

namespace imex {
namespace region {
void registerBufferizableOpInterfaceExternalModels(
    ::mlir::DialectRegistry &registry);
} // namespace region
} // namespace imex

#endif // IMEX_DIALECT_REGION_BUFFERIZABLEOPINTERFACEIMPL_H
