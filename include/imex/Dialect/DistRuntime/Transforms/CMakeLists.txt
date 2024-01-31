set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name DistRuntime)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix DistRuntime)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix DistRuntime)
add_public_tablegen_target(IMEXDistRuntimePassIncGen)

add_mlir_doc(Passes DistRuntimePasses ./ -gen-pass-doc)
