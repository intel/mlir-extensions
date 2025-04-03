add_imex_conversion_library(MLIRXeVMToLLVM
  XeVMToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/imex/Conversion/XeVMToLLVM

  DEPENDS
  IMEXConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRXeVMDialect
  MLIRPass
  MLIRTransforms
)
