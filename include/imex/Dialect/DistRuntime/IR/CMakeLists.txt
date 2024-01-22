add_mlir_dialect(DistRuntimeOps distruntime)
add_mlir_doc(DistRuntimeOps DistRuntimeDialect Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS DistRuntimeOps.td)
mlir_tablegen(DistRuntimeOpsIFaces.h.inc -gen-op-interface-decls)
mlir_tablegen(DistRuntimeOpsIFaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRDistRuntimeOpsIFacesIncGen)
add_dependencies(mlir-generic-headers MLIRDistRuntimeOpsIFacesIncGen)
