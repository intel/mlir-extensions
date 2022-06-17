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

# quick check that we are in the right dir
if not os.path.isdir(incroot) or not os.path.isdir(libroot):
    raise Exception(
        f"'{incroot}' and/or '{libroot}' not found. "
        + "Are you in the right directory?"
    )

# create dialect subdirs and default CMakeLists
for d in [incroot, libroot]:
    # This raises an exception if already exists -> no overwriting
    os.makedirs(jp(d, name))
    # empty CMakeLists.txt (cpp will be filled below)
    open(jp(d, name, "CMakeLists.txt"), "a").close()
    # we append in the root CMakeLists, other dialects exist
    with open(jp(d, "CMakeLists.txt"), "a") as f:
        f.write(f"add_subdirectory({name})\n")

# Create CMakeLists.txt
with open(jp(libroot, name, f"CMakeLists.txt"), "w") as f:
    f.write(f"""add_mlir_conversion_library(IMEX{name}
  {name}.cpp

  ADDITIONAL_HEADER_DIRS
  ${{MLIR_MAIN_INCLUDE_DIR}}/mlir/Conversion/{name}

  LINK_LIBS PUBLIC
  IMEX{args.target}
)
""")

# Create header stub
with open(jp(incroot, name, f"{name}.h"), "w") as f:
    f.write(f"""// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Converting {args.source} to {args.target}

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
/// Populate the given list with patterns that eliminate Dist ops
void populate{name}ConversionPatterns(::mlir::LLVMTypeConverter &converter,
                                        ::mlir::RewritePatternSet &patterns);

/// Create a pass to eliminate Dist ops
std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>> createConvert{name}Pass();

}} // namespace imex

#endif // _{name}_H_INCLUDED_
""")

# Create cpp stub
with open(jp(libroot, name, f"{name}.cpp"), "w") as f:
    f.write(f"""// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Converting {args.source} to {args.target}

#include <imex/Conversion/{name}/{name}.h>
#include <imex/Dialect/{args.source}/IR/{args.source}Ops.h>
#include <imex/Dialect/{args.target}/IR/{args.target}Ops.h>
#include <imex/internal/PassWrapper.h>

#include <mlir/IR/BuiltinOps.h>

// *******************************
// ***** Individual patterns *****
// *******************************

// RegisterPTensorOp -> no-op
struct ElimRegisterPTensorOp
    : public mlir::OpRewritePattern<::{args.source.lower()}::FIXME> {{
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

namespace imex {{

/// Populate the given list with patterns that convert {args.source} to {args.target}
void populate{name}ConversionPatterns(::mlir::LLVMTypeConverter &converter,
                                      ::mlir::RewritePatternSet &patterns);

// Full Pass
// FIXME if you need typ conversion or other more complicated stuff you have to write your own
struct {name}Pass that convert {args.source} to {args.target}
    : public ::imex::PassWrapper<{name}Pass, ::mlir::ModuleOp,
                                 ::imex::DialectList<::imex::{args.target.lower()}::{args.target}Dialect,
                                                     FIXME more target dialects>,
                                 FIXMEOp,
                                 ...>
{{}};

/// Create a pass that convert {args.source} to {args.target}
std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>> createConvert{name}Pass() {{
    return std::make_unique<{name}Pass>();
}}

}} // namespace imex
""")
