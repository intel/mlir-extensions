add_mlir_dialect(RegionOps region)
add_mlir_doc(RegionOps RegionDialect Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS RegionOps.td)
mlir_tablegen(RegionOpsAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=region)
mlir_tablegen(RegionOpsAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=region)
add_public_tablegen_target(MLIRRegionIncGen)
add_dependencies(mlir-headers MLIRRegionIncGen)
