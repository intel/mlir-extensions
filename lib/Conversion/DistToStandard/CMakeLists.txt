add_imex_conversion_library(IMEXDistToStandard
  DistToStandard.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/imex/Conversion/DistToStandard

  DEPENDS
  IMEXConversionPassIncGen

  LINK_LIBS PUBLIC
  IMEXNDArrayDialect
  MLIRLinalgDialect
)
