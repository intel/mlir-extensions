add_mlir_conversion_library(IMEXPTensorToLinalg
  PTensorToLinalg.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/PTensorToLinalg

  DEPENDS
  IMEXConversionPassIncGen

  LINK_LIBS PUBLIC
  IMEXPTensor
  MLIRLinalgDialect
)
