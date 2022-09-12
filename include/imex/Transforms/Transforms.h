//===- Transforms.h - Transforms Pass declaration ---------------------*- C++
//-*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares all the Transforms Passes.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>

namespace mlir {
class Pass;
}

namespace imex {

std::unique_ptr<mlir::Pass> createInsertGPUAllocsPass();
std::unique_ptr<mlir::Pass> createSetSPIRVCapabilitiesPass();
std::unique_ptr<mlir::Pass> createSetSPIRVAbiAttribute();
} // namespace imex
