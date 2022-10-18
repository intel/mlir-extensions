set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name PTensor)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix PTensor)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix PTensor)
add_public_tablegen_target(IMEXPTensorPassIncGen)

add_mlir_doc(Passes PTensorPasses ./ -gen-pass-doc)
