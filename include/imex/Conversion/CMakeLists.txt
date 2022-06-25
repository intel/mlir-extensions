set(LLVM_TARGET_DEFINITIONS IMEXPasses.td)
mlir_tablegen(IMEXPasses.h.inc -gen-pass-decls -name Conversion)
mlir_tablegen(IMEXPasses.capi.h.inc -gen-pass-capi-header --prefix Conversion)
mlir_tablegen(IMEXPasses.capi.cpp.inc -gen-pass-capi-impl --prefix Conversion)
add_public_tablegen_target(IMEXConversionPassIncGen)

add_mlir_doc(IMEXPasses ConversionPasses ./ -gen-pass-doc)
