set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Dist)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Dist)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Dist)
add_public_tablegen_target(IMEXDistPassIncGen)

add_mlir_doc(Passes DistPasses ./ -gen-pass-doc)
