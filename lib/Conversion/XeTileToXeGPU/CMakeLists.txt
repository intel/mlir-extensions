add_imex_conversion_library(IMEXXeTileToXeGPU
  XeTileToXeGPU.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/imex/Conversion/XeTileToXeGPU

  DEPENDS
  IMEXConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRXeGPUDialect
)
