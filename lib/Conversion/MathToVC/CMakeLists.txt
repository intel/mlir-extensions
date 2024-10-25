add_imex_conversion_library(IMEXMathToVC
  MathToVC.cpp


  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/MathToVC

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
