add_mlir_conversion_library(IMEXLinalgToSPIRV
  LinalgToSPIRV.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/LinalgToSPIRV

  DEPENDS
  IMEXConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRSPIRVDialect
)
