set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name XeTile)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix XeTile)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix XeTile)
add_public_tablegen_target(IMEXXeTilePassIncGen)

add_mlir_doc(Passes XeTilePasses ./ -gen-pass-doc)
