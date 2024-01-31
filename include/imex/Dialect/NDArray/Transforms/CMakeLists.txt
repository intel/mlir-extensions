set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name NDArray)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix NDArray)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix NDArray)
add_public_tablegen_target(IMEXNDArrayPassIncGen)

add_mlir_doc(Passes NDArrayPasses ./ -gen-pass-doc)
