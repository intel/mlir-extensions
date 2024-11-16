add_imex_conversion_library(IMEXArithToVC
  ArithToVC.cpp


  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ArithToVC

  DEPENDS
  IMEXConversionPassIncGen

  #LINK_COMPONENTS

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  # MLIRTransforms
  MLIRLLVMCommonConversion

  MLIRGPUDialect
  MLIRPass
  )
