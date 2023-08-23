add_mlir_conversion_library(XeGPUToSPIRV
  XeGPUToSPIRV.cpp

  DEPENDS
  IMEXConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSPIRVDialect
  MLIRSPIRVConversion
  MLIRSupport
  MLIRTransforms
  )
