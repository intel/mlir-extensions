add_imex_conversion_library(IMEXNDArrayToLinalg
  NDArrayToLinalg.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/NDArrayToLinalg

  DEPENDS
  IMEXConversionPassIncGen

  LINK_LIBS PUBLIC
  IMEXNDArrayDialect
  MLIRLinalgDialect
)
