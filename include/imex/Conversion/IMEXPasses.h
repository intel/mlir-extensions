//===- IMEXPasses.h - Conversion Pass Construction and Registration -------===//
//
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file includes all IMEX conversion pass declarations.
///
//===----------------------------------------------------------------------===//

#ifndef _IMEX_CONVERSION_PASSES_H_INCLUDED_
#define _IMEX_CONVERSION_PASSES_H_INCLUDED_

#include <imex/Conversion/PTensorToLinalg/DistElim.h>
#include <imex/Conversion/PTensorToLinalg/PTensorToLinalg.h>

namespace imex {

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include <imex/Conversion/IMEXPasses.h.inc>

} // namespace imex

#endif // _IMEX_CONVERSION_PASSES_H_INCLUDED_
