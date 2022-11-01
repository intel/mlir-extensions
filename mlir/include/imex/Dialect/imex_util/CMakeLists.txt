# SPDX-FileCopyrightText: 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include_directories(${MLIR_INCLUDE_DIRS})
set(dialect ImexUtilOps)
set(dialect_namespace imex_util)
set(LLVM_TARGET_DEFINITIONS ${dialect}.td)
mlir_tablegen(${dialect}Enums.h.inc -gen-enum-decls)
mlir_tablegen(${dialect}Enums.cpp.inc -gen-enum-defs)
mlir_tablegen(${dialect}Attributes.h.inc -gen-attrdef-decls -attrdefs-dialect=${dialect_namespace})
mlir_tablegen(${dialect}Attributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=${dialect_namespace})
mlir_tablegen(${dialect}.h.inc -gen-op-decls)
mlir_tablegen(${dialect}.cpp.inc -gen-op-defs)
mlir_tablegen(${dialect}Dialect.h.inc -gen-dialect-decls -dialect=${dialect_namespace})
mlir_tablegen(${dialect}Dialect.cpp.inc -gen-dialect-defs -dialect=${dialect_namespace})
mlir_tablegen(${dialect}Types.h.inc -gen-typedef-decls)
mlir_tablegen(${dialect}Types.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIR${dialect}IncGen)
add_dependencies(mlir-headers MLIR${dialect}IncGen)
