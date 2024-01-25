add_mlir_dialect(DistOps dist)
add_mlir_doc(DistOps DistDialect Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS DistOps.td)
mlir_tablegen(DistOpsAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=dist)
mlir_tablegen(DistOpsAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=dist)
add_public_tablegen_target(MLIRDistIncGen)
add_dependencies(mlir-headers MLIRDistIncGen)
