//===- InitIMEXPasses.h - IMEX Registration ---------------------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all IMEX
// dialects and passes to the system.
//
//===----------------------------------------------------------------------===//

#ifndef _IMEX_INITALLPASSES_H_
#define _IMEX_INITALLPASSES_H_

#include <imex/Conversion/Passes.h>
// #include <imex/Transforms/IMEXPasses.h>
#include <imex/Dialect/Dist/Transforms/Passes.h>
#include <imex/Dialect/DistRuntime/Transforms/Passes.h>
#include <imex/Dialect/NDArray/Transforms/Passes.h>
// #include <imex/Dialect/*/Transforms/Passes.h>
#include "imex/Transforms/Passes.h"
#include <imex/Dialect/XeTile/Transforms/Passes.h>

#include <cstdlib>

namespace imex {

/// This function may be called to register the IMEX passes with the
/// global registry.
/// If you're building a compiler, you likely don't need this: you would build a
/// pipeline programmatically without the need to register with the global
/// registry, since it would already be calling the creation routine of the
/// individual passes.
/// The global registry is interesting to interact with the command-line tools.
inline void registerAllPasses() {
  // General passes
  registerTransformsPasses();

  // Conversion passes
  registerConversionPasses();

  // Dialect passes
  registerNDArrayPasses();
  registerDistPasses();
  registerDistRuntimePasses();
  registerXeTilePasses();
  // register*Passes();

  // Dialect pipelines
}

} // namespace imex

#endif // _IMEX_INITALLPASSES_H_
