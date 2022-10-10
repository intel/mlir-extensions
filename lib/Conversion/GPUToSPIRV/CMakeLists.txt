add_mlir_conversion_library(IMEXGPUToSPIRV
  GPUToSPIRVPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/GPUToSPIRV

  DEPENDS
  IMEXConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithmeticToSPIRV
  MLIRGPUOps
  MLIRFuncToSPIRV
  MLIRIR
  MLIRPass
  MLIRSCFToSPIRV
  MLIRSPIRVDialect
  MLIRSPIRVConversion
  MLIRSupport
  MLIRTransforms
  )
