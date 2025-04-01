add_mlir_dialect(NDArrayOps ndarray)
add_mlir_doc(NDArrayOps NDArrayDialect Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS NDArrayOps.td)
mlir_tablegen(NDArrayOpsAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=ndarray)
mlir_tablegen(NDArrayOpsAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=ndarray)
add_public_tablegen_target(MLIRNDArrayIncGen)
add_dependencies(mlir-headers MLIRNDArrayIncGen)
