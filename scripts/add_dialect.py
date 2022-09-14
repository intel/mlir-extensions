# Adding stubs for a new dialect, such as CMakeLists, .td files and basic
# headers/cpp files.  If this goes upstream we probably want this cleaned up,
# like source code should go in separate template files.
# More graceful bail-out/overwriting behavior would be nice, too.

import os
from os.path import join as jp
import argparse

parser = argparse.ArgumentParser(
    description="Create subfolders for a new dialect and create/extend "
    + "CMakeLists and stub sources"
)
parser.add_argument("name", help="new dialect's name")

args = parser.parse_args()

incroot = jp("include", "imex", "Dialect")
libroot = jp("lib", "Dialect")
testroot = jp("test", "Dialect")

# quick check that we are in the right dir
if not os.path.isdir(incroot) or not os.path.isdir(libroot):
    raise Exception(
        f"'{incroot}' and/or '{libroot}' not found. "
        + "Are you in the right directory?"
    )

# create dialect subdirs and default CMakeLists
for d in [incroot, libroot, testroot]:
    # This raises an exception if already exists -> no overwriting
    os.makedirs(jp(d, args.name, "IR"))
    os.makedirs(jp(d, args.name, "Transforms"))
    if d != testroot:
        # we append in the root CMakeLists, other dialects exist
        with open(jp(d, "CMakeLists.txt"), "a") as f:
            f.write(f"add_subdirectory({args.name})\n")
        with open(jp(d, args.name, "CMakeLists.txt"), "w") as f:
            f.write("add_subdirectory(IR)\nadd_subdirectory(Transforms)\n")

# add test file stub
fn = jp(testroot, args.name, "IR", f"{args.name}Ops.mlir")
with open(fn, "w") as f:
    f.write("""// RUN: imex-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: imex-opt %s | imex-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: imex-opt -mlir-print-op-generic %s | imex-opt | FileCheck %s
""")

# Default rules for IR tablegen and alike
with open(jp(incroot, args.name, "IR", "CMakeLists.txt"), "w") as f:
    f.write(f"add_mlir_dialect({args.name}Ops {args.name.lower()})\n")
    f.write(
        f"add_mlir_doc({args.name}Ops {args.name}Dialect Dialects/ "
        + "-gen-dialect-doc)\n"
    )

# Default rules for transforms/passes tablegen and alike
with open(jp(incroot, args.name, "Transforms", "CMakeLists.txt"), "w") as f:
    f.write(f"""set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name {args.name})
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix {args.name})
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix {args.name})
add_public_tablegen_target(IMEX{args.name}PassIncGen)\n
add_mlir_doc(Passes {args.name}Passes ./ -gen-pass-doc)
""")

# Default rules for tblgen'erated cpps
with open(jp(libroot, args.name, "IR", "CMakeLists.txt"), "w") as f:
    f.write(f"""add_mlir_dialect_library(IMEX{args.name}Dialect
  {args.name}Ops.cpp

  ADDITIONAL_HEADER_DIRS
  ${{PROJECT_SOURCE_DIR}}/include/imex/Dialect/{args.name}

  DEPENDS
  IMEX{args.name}OpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
""")

# Default rules for Transforms
with open(jp(libroot, args.name, "Transforms", "CMakeLists.txt"), "w") as f:
    f.write(f"""add_mlir_dialect_library(IMEX{args.name}Transforms
  # FIXME.cpp

  ADDITIONAL_HEADER_DIRS
  ${{PROJECT_SOURCE_DIR}}/include/imex/Dialect/{args.name}

  DEPENDS
  IMEX{args.name}PassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  IMEX{args.name}Dialect
)
""")

# Generate default tablegen defs
fn = jp(incroot, args.name, "IR", f"{args.name}Ops.td")
with open(fn, "w") as f:
    f.write(f"""//===- {os.path.basename(fn)} - {args.name} dialect  -------*- tablegen -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \\file
/// This file defines the basic operations for the {args.name} dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _{args.name}_OPS_TD_INCLUDED_
#define _{args.name}_OPS_TD_INCLUDED_

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Provide a definition of the '{args.name}' dialect in the ODS framework so that we
// can define our operations.
def {args.name}_Dialect : Dialect {{
    // The namespace of our dialect
    let name = "{args.name.lower()}";

    // A short one-line summary of our dialect.
    let summary = "FIXME insert summary";

    // A longer description of our dialect.
    let description = [{{
            FIXME insert discription
        }}];

    // The C++ namespace that the dialect class definition resides in.
    let cppNamespace = "::imex::{args.name.lower()}";
}}

// Base class for dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class {args.name}_Op<string mnemonic, list<Trait> traits = []> :
    Op<{args.name}_Dialect, mnemonic, traits>;

#endif // _{args.name}_OPS_TD_INCLUDED_
""")


# Generate default IR include file
fn = jp(incroot, args.name, "IR", f"{args.name}Ops.h")
with open(fn, "w") as f:
    f.write(f"""//===- {os.path.basename(fn)} - {args.name} dialect  -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \\file
/// This file declares the {args.name} dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _{args.name}_OPS_H_INCLUDED_
#define _{args.name}_OPS_H_INCLUDED_

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace imex {{
namespace {args.name.lower()} {{

}} // namespace {args.name.lower()}
}} // namespace mlir

#include <imex/Dialect/{args.name}/IR/{args.name}OpsDialect.h.inc>
#define GET_TYPEDEF_CLASSES
#include <imex/Dialect/{args.name}/IR/{args.name}OpsTypes.h.inc>
#define GET_OP_CLASSES
#include <imex/Dialect/{args.name}/IR/{args.name}Ops.h.inc>

#endif // _{args.name}_OPS_H_INCLUDED_
""")

# Generate default IR cpp file
fn = jp(libroot, args.name, "IR", f"{args.name}Ops.cpp")
with open(fn, "w") as f:
    f.write(f"""//===- {os.path.basename(fn)} - {args.name} dialect -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \\file
/// This file implements the {args.name} dialect and its basic operations.
///
//===----------------------------------------------------------------------===//

#include <imex/Dialect/{args.name}/IR/{args.name}Ops.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/DialectImplementation.h>

namespace imex {{
namespace {args.name.lower()} {{

    void {args.name}Dialect::initialize()
    {{
        addTypes<
#define GET_TYPEDEF_LIST
#include <imex/Dialect/{args.name}/IR/{args.name}OpsTypes.cpp.inc>
            >();
        addOperations<
#define GET_OP_LIST
#include <imex/Dialect/{args.name}/IR/{args.name}Ops.cpp.inc>
            >();
    }}

}} // namespace {args.name.lower()}
}} // namespace mlir

#include <imex/Dialect/{args.name}/IR/{args.name}OpsDialect.cpp.inc>
#define GET_TYPEDEF_CLASSES
#include <imex/Dialect/{args.name}/IR/{args.name}OpsTypes.cpp.inc>
#define GET_OP_CLASSES
#include <imex/Dialect/{args.name}/IR/{args.name}Ops.cpp.inc>
""")

##### Transforms and Passes
# add Passes.td
fn = jp(incroot, args.name, "Transforms", "Passes.td")
with open(fn, "w") as f:
    f.write(f"""//===-- {os.path.basename(fn)} - {args.name} pass definition file --------*- tablegen -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \\file
/// This file defines passes/transformations of the {args.name} dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _{args.name}_PASSES_TD_INCLUDED_
#define _{args.name}_PASSES_TD_INCLUDED_

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// FIXME pass
//===----------------------------------------------------------------------===//

// def FIXME-passname: Pass<"FIXME-command-line", ""> {{
//   let summary = "FIXME";
//   let description = [{{
//     FIXME
//
//     #### Input invariant
//
//     - FIXME
//
//     #### Output IR
//
//     - FIXME
//   }}];
//   let constructor = "::imex::createFIXMEPass()";
//   let dependentDialects = ["::imex::{args.name.lower()}::{args.name}Dialect"];
//   let options = [];
// }}

#endif // _{args.name}_PASSES_TD_INCLUDED_
""")

# add Passes.h
fn = jp(incroot, args.name, "Transforms", "Passes.h")
with open(fn, "w") as f:
    f.write(f"""//===-- {os.path.basename(fn)} - {args.name} pass declaration file --------*- tablegen -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \\file
/// This header file defines prototypes that expose pass constructors for the
/// {args.name} dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _{args.name}_PASSES_H_INCLUDED_
#define _{args.name}_PASSES_H_INCLUDED_

#include <mlir/Pass/Pass.h>

namespace mlir {{
class LLVMTypeConverter;
class MLIRContext;
class ModuleOp;
template <typename T> class OperationPass;
class RewritePatternSet;
}} // namespace mlir

namespace imex {{

//===----------------------------------------------------------------------===//
/// {args.name} passes.
//===----------------------------------------------------------------------===//

/// FIXME
std::unique_ptr<::mlir::Pass> createFIXMEPass();

/// Populate the given list with patterns that eliminate Dist ops
void populateFIXMEPatterns(::mlir::LLVMTypeConverter &converter,
                           ::mlir::RewritePatternSet &patterns);

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include <imex/Dialect/{args.name}/Transforms/Passes.h.inc>

}} // namespace imex

#endif // _{args.name}_PASSES_H_INCLUDED_
""")

# add PassDetail.h
fn = jp(libroot, args.name, "Transforms", "PassDetail.h")
with open(fn, "w") as f:
    f.write(f"""//===-- {os.path.basename(fn)} - {args.name} pass details --------*- tablegen -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \\file
/// This header file defines prototypes for {args.name} dialect passes.
///
//===----------------------------------------------------------------------===//

#ifndef _{args.name}_PASSDETAIL_H_INCLUDED_
#define _{args.name}_PASSDETAIL_H_INCLUDED_


#include <mlir/Pass/Pass.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/FunctionInterfaces.h>

namespace mlir {{

class AffineDialect;

namespace arith {{
class ArithmeticDialect;
}} // namespace arith

// FIXME define other dependent MLIR dialects

}} // namespace mlir

namespace imex {{

#define GEN_PASS_CLASSES
#include <imex/Dialect/{args.name}/Transforms/Passes.h.inc>

}} // namespace imex

#endif // _{args.name}_PASSDETAIL_H_INCLUDED_
""")
