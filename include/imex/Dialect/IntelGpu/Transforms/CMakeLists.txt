set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name IntelGpu)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix IntelGpu)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix IntelGpu)
add_public_tablegen_target(IMEXIntelGpuPassIncGen)

add_mlir_doc(Passes IntelGpuPasses ./ -gen-pass-doc)
