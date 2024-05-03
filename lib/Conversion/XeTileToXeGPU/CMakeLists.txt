add_imex_conversion_library(IMEXXeTileToXeGPU
  ArithOpConversion.cpp
  SCFOpConversion.cpp
  XeTileToXeGPU.cpp
  XeTileOpConversion.cpp
  XeTileToXeGPUConversion.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/imex/Conversion/XeTileToXeGPU

  DEPENDS
  IMEXConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRXeGPUDialect
)
