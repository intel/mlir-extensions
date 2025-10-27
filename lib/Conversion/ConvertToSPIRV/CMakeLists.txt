add_imex_conversion_library(IMEXConvertToSPIRV
  ConvertToSPIRVPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/imex/Conversion/ConvertToSPIRV

  DEPENDS
  IMEXConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithToSPIRV
  MLIRControlFlowToSPIRV
  MLIRFuncToSPIRV
  MLIRGPUDialect
  MLIRGPUToSPIRV
  MLIRIR
  MLIRMathToSPIRV
  MLIRPass
  MLIRSCFToSPIRV
  MLIRSPIRVDialect
  MLIRSPIRVConversion
  MLIRSupport
  MLIRTransforms
  )
