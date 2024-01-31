set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Region)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Region)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Region)
add_public_tablegen_target(IMEXRegionPassIncGen)

add_mlir_doc(Passes RegionPasses ./ -gen-pass-doc)
