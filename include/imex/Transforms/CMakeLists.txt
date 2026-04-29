set(LLVM_TARGET_DEFINITIONS Passes.td)

# Add preprocessor definition for VC conversions if enabled
if(IMEX_BUILD_VC_CONVERSIONS)
  set(IMEX_VC_CONVERSIONS_FLAG "-DIMEX_BUILD_VC_CONVERSIONS")
else()
  set(IMEX_VC_CONVERSIONS_FLAG "")
endif()

mlir_tablegen(Passes.h.inc -gen-pass-decls -name Transforms ${IMEX_VC_CONVERSIONS_FLAG})
mlir_tablegen(Transforms.capi.h.inc -gen-pass-capi-header --prefix Transforms ${IMEX_VC_CONVERSIONS_FLAG})
mlir_tablegen(Transforms.capi.cpp.inc -gen-pass-capi-impl --prefix Transforms ${IMEX_VC_CONVERSIONS_FLAG})
add_public_tablegen_target(IMEXTransformsPassIncGen)

add_mlir_doc(Passes IMEXGeneralPasses ./ -gen-pass-doc)
