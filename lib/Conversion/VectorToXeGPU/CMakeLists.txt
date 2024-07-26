add_imex_conversion_library(IMEXVectorToXeGPU
  VectorToXeGPU.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/imex/Conversion/VectorToXeGPU

  DEPENDS
  IMEXConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRXeGPUDialect
)
