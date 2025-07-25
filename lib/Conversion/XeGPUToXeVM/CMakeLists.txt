add_imex_conversion_library(MLIRXeGPUToXeVM
  XeGPUToXeVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/imex/Conversion/XeGPUToXeVM

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
  MLIRVectorDialect
  MLIRArithDialect
  MLIRXeGPUDialect
  MLIRPass
  MLIRTransforms
)
