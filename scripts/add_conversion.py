# Adding stubs for a new dialect conversion, such as CMakeLists, and basic
# headers/cpp files.  If this goes upstream we probably want this cleaned up,
# like source code should go in separate template files.
# More graceful bail-out/overwriting behavior would be nice, too.

import os
from os.path import join as jp
import argparse

parser = argparse.ArgumentParser(
    description="Create subfolders for a new dialect conversion and create/extend "
    + "CMakeLists and stub sources"
)
parser.add_argument("source", help="source dialect's name")
parser.add_argument("target", help="target dialect's name")

args = parser.parse_args()

name = f"{args.source}To{args.target}"

incroot = jp("include", "imex", "Conversion")
libroot = jp("lib", "Conversion")
testroot = jp("test", "Conversion")

# quick check that we are in the right dir
if not os.path.isdir(incroot) or not os.path.isdir(libroot):
    raise Exception(
        f"'{incroot}' and/or '{libroot}' not found. "
        + "Are you in the right directory?"
    )

# create dialect subdirs and default CMakeLists
for d in [incroot, libroot, testroot]:
    # This raises an exception if already exists -> no overwriting
    os.makedirs(jp(d, name))
    if d != testroot:
        # empty CMakeLists.txt (cpp will be filled below)
        open(jp(d, name, "CMakeLists.txt"), "a").close()
        # we append in the root CMakeLists, other dialects exist
        with open(jp(d, "CMakeLists.txt"), "a") as f:
            f.write(f"add_subdirectory({name})\n")

# add test file stub
fn = jp(testroot, name, f"{name}.mlir")
with open(fn, "w") as f:
    f.write(f'// RUN: imex-opt --split-input-file --convert-{args.source.lower()}-to-{args.target.lower()} %s -verify-diagnostics -o -| FileCheck %s\n')

# Create CMakeLists.txt
with open(jp(libroot, name, f"CMakeLists.txt"), "w") as f:
    f.write(f"""add_mlir_conversion_library(IMEX{name}
  {name}.cpp

  ADDITIONAL_HEADER_DIRS
  ${{MLIR_MAIN_INCLUDE_DIR}}/imex/Conversion/{name}

  DEPENDS
  IMEXConversionPassIncGen

  LINK_LIBS PUBLIC
  IMEX{args.target}Dialect
)
""")

# Create header stub
fn = jp(incroot, name, f"{name}.h")
header = fn
with open(fn, "w") as f:
    f.write(f"""//===- {os.path.basename(fn)} - {name} conversion  -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \\file
/// This file defines the {name} conversion, converting the {args.source}
/// dialect to the {args.target} dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _{name}_H_INCLUDED_
#define _{name}_H_INCLUDED_

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir {{
class LLVMTypeConverter;
class MLIRContext;
class ModuleOp;
template <typename T>
class OperationPass;
class RewritePatternSet;
}}

namespace imex {{
/// Populate the given list with patterns rewrite {args.source} Ops
void populate{name}ConversionPatterns(::mlir::LLVMTypeConverter &converter,
                                        ::mlir::RewritePatternSet &patterns);

/// Create a pass to convert the {args.source} dialect to the {args.target} dialect.
std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>> createConvert{name}Pass();

}} // namespace imex

#endif // _{name}_H_INCLUDED_
""")

# Create cpp stub
fn = jp(libroot, name, f"{name}.cpp")
with open(fn, "w") as f:
    f.write(f"""//===- {os.path.basename(fn)} - {name} conversion  -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \\file
/// This file implements the {name} conversion, converting the {args.source}
/// dialect to the {args.target} dialect.
///
//===----------------------------------------------------------------------===//

#include <imex/Conversion/{name}/{name}.h>
#include <imex/Dialect/{args.source}/IR/{args.source}Ops.h>
#include <imex/Dialect/{args.target}/IR/{args.target}Ops.h>
#include <imex/internal/PassWrapper.h>

#include <mlir/IR/BuiltinOps.h>

namespace imex {{

namespace {{
// *******************************
// ***** Individual patterns *****
// *******************************

// Some{args.source}Op -> Some{args.target}Op
struct Some{args.source}OpRewriter
    : public mlir::OpRewritePattern<::imex::{args.source.lower()}::FIXME> {{
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite({args.source.lower()}::FIXME op,
                  mlir::PatternRewriter &rewriter) const override {{

      FIXME fill in rewriting code

      return ::mlir::success();
  }}
}};

// *******************************
// ***** Pass infrastructure *****
// *******************************

// Full Pass
struct Convert{name}Pass that convert {args.source} to {args.target}
    : public ::imex::Convert{name}PassBase<Convert{name}Pass>
{{
  Convert{name}Pass() = default;

  void runOnOperation() override {{
    ::mlir::FrozenRewritePatternSet patterns;
    insertPatterns<FIXME patterns>(getContext(), patterns);
    (void)::mlir::applyPatternsAndFoldGreedily(this->getOperation(), patterns);
  }}
}};

}} // namespace

/// Populate the given list with patterns that convert {args.source} to {args.target}
void populate{name}ConversionPatterns(::mlir::LLVMTypeConverter &converter,
                                      ::mlir::RewritePatternSet &patterns)
{{
  FIXME
}}

/// Create a pass that convert {args.source} to {args.target}
std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>> createConvert{name}Pass() {{
    return std::make_unique<{name}Pass>();
}}

}} // namespace imex
""")

# add header-include to Passes.h
fn = jp(incroot, "Passes.h")
with open(fn, "r") as f:
    lines = f.readlines()
done = False
with open(fn, "w") as f:
    for l in lines:
        if not done and l.startswith("#include "):
            f.write(f"#include <imex/Conversion/{name}/{name}.h>\n")
            done = True
        f.write(l)

# add Pass to Passes.td
fn = jp(incroot, "Passes.td")
with open(fn, "r") as f:
    lines = f.readlines()
done = False
with open(fn, "w") as f:
    for l in lines:
        if not done and l.startswith("#endif // _IMEX_CONVERSION_PASSES_"):
            f.write(f"""//===----------------------------------------------------------------------===//
// {name}
//===----------------------------------------------------------------------===//

def Convert{name}: Pass<"convert-{args.source.lower()}-to-{args.target.lower()}", "::mlir::ModuleOp"> {{
  let summary = "Convert from the {args.source} dialect to the {args.target} dialect.";
  let description = [{{
    Convert {args.source} dialect operations into the {args.target} dialect operations.

    #### Input invariant

    - FIXME

    #### Output IR

    - FIXME
  }}];
  let constructor = "::imex::create{name}ConvertPass()";
  let dependentDialects = ["::imex::{args.target.lower()}::{args.target}Dialect"];
  let options = [];
}}

""")
            done = True
        f.write(l)
