set(LLVM_TARGET_DEFINITIONS Passes.td)

# Add preprocessor definition for XeGPU layout passes if enabled
if(IMEX_ENABLE_XEGPU_LAYOUT_PASSES)
  set(IMEX_XEGPU_LAYOUT_FLAG "-DIMEX_ENABLE_XEGPU_LAYOUT_PASSES")
else()
  set(IMEX_XEGPU_LAYOUT_FLAG "")
endif()

mlir_tablegen(Passes.h.inc -gen-pass-decls -name Transforms ${IMEX_XEGPU_LAYOUT_FLAG})
mlir_tablegen(Transforms.capi.h.inc -gen-pass-capi-header --prefix Transforms ${IMEX_XEGPU_LAYOUT_FLAG})
mlir_tablegen(Transforms.capi.cpp.inc -gen-pass-capi-impl --prefix Transforms ${IMEX_XEGPU_LAYOUT_FLAG})
add_public_tablegen_target(IMEXTransformsPassIncGen)

add_mlir_doc(Passes IMEXGeneralPasses ./ -gen-pass-doc)
