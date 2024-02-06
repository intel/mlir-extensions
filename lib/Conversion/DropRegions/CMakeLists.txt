add_imex_conversion_library(IMEXDropRegions
  DropRegions.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/imex/Conversion/DropRegions

  DEPENDS
  IMEXConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRGPUDialect
  IMEXGPUXDialect
)
