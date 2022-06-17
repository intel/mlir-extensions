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

# quick check that we are in the right dir
if not os.path.isdir(incroot) or not os.path.isdir(libroot):
    raise Exception(
        f"'{incroot}' and/or '{libroot}' not found. "
        + "Are you in the right directory?"
    )

# create dialect subdirs and default CMakeLists
for d in [incroot, libroot]:
    # This raises an exception if already exists -> no overwriting
    os.makedirs(jp(d, args.name, "IR"))
    # we append in the root CMakeLists, other dialects exist
    with open(jp(d, "CMakeLists.txt"), "a") as f:
        f.write(f"add_subdirectory({args.name})\n")
    with open(jp(d, args.name, "CMakeLists.txt"), "w") as f:
        f.write("add_subdirectory(IR)\n")

# Default rules for tablegen and alike
with open(jp(incroot, args.name, "IR", "CMakeLists.txt"), "w") as f:
    f.write(f"add_mlir_dialect({args.name}Ops {args.name.lower()})\n")
    f.write(
        f"add_mlir_doc({args.name}Dialect {args.name}Dialect {args.name}/"
        + " -gen-dialect-doc)\n"
    )
    f.write(
        f"add_mlir_doc({args.name}Ops {args.name}Ops {args.name}/ "
        + "-gen-op-doc)\n"
    )

# Default rules for tblgen'erated cpps
with open(jp(libroot, args.name, "IR", "CMakeLists.txt"), "w") as f:
    f.write(
        """add_mlir_dialect_library(IMEX{0}
        {0}Ops.cpp

        ADDITIONAL_HEADER_DIRS
        ${{PROJECT_SOURCE_DIR}}/include/imex/Dialect/{0}

        DEPENDS
        IMEX{0}OpsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    )
""".format(
            args.name
        )
    )

# Generate default tablegen defs
tgdef = """// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef _{0}_OPS_TD_INCLUDED_
#define _{0}_OPS_TD_INCLUDED_

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Provide a definition of the '{0}' dialect in the ODS framework so that we
// can define our operations.
def {0}_Dialect : Dialect {{
    // The namespace of our dialect
    let name = "{1}";

    // A short one-line summary of our dialect.
    let summary = "FIXME insert summary";

    // A longer description of our dialect.
    let description = [{{
            FIXME insert discription
        }}];

    // The C++ namespace that the dialect class definition resides in.
    let cppNamespace = "::{1}";
}}

// Base class for dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class {0}_Op<string mnemonic, list<Trait> traits = []> :
    Op<{0}_Dialect, mnemonic, traits>;

#endif // _{0}_OPS_TD_INCLUDED_
""".format(
    args.name, args.name.lower()
)

with open(jp(incroot, args.name, "IR", f"{args.name}Ops.td"), "w") as f:
    f.write(tgdef)

# Generate default IR include file
irinc = """// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef _{0}_OPS_H_INCLUDED_
#define _{0}_OPS_H_INCLUDED_

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace {1} {{

}}

#include <imex/Dialect/{0}/IR/{0}OpsDialect.h.inc>
#define GET_TYPEDEF_CLASSES
#include <imex/Dialect/{0}/IR/{0}OpsTypes.h.inc>
#define GET_OP_CLASSES
#include <imex/Dialect/{0}/IR/{0}Ops.h.inc>

#endif // _{0}_OPS_H_INCLUDED_
""".format(
    args.name, args.name.lower()
)

with open(jp(incroot, args.name, "IR", f"{args.name}Ops.h"), "w") as f:
    f.write(irinc)

# Generate default IR cpp file
irlib = """// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <imex/Dialect/{0}/IR/{0}Ops.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/DialectImplementation.h>

namespace {1} {{

    void {0}Dialect::initialize()
    {{
        addTypes<
#define GET_TYPEDEF_LIST
#include <imex/Dialect/{0}/IR/{0}OpsTypes.cpp.inc>
            >();
        addOperations<
#define GET_OP_LIST
#include <imex/Dialect/{0}/IR/{0}Ops.cpp.inc>
            >();
    }}

}} // namespace {1}

#include <imex/Dialect/{0}/IR/{0}OpsDialect.cpp.inc>
#define GET_TYPEDEF_CLASSES
#include <imex/Dialect/{0}/IR/{0}OpsTypes.cpp.inc>
#define GET_OP_CLASSES
#include <imex/Dialect/{0}/IR/{0}Ops.cpp.inc>
""".format(
    args.name, args.name.lower()
)

with open(jp(libroot, args.name, "IR", f"{args.name}Ops.cpp"), "w") as f:
    f.write(irlib)
