set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Transforms)
mlir_tablegen(Transforms.capi.h.inc -gen-pass-capi-header --prefix Transforms)
mlir_tablegen(Transforms.capi.cpp.inc -gen-pass-capi-impl --prefix Transforms)
add_public_tablegen_target(IMEXTransformsPassIncGen)

add_mlir_doc(Passes IMEXGeneralPasses ./ -gen-pass-doc)
