add_mlir_conversion_library(IMEXGPUToGPUX
  GPUToGPUX.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/imex/Conversion/GPUToGPUX

  DEPENDS
  IMEXConversionPassIncGen

  LINK_LIBS PUBLIC
  IMEXGPUXDialect
)
