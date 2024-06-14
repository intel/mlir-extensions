add_imex_conversion_library(IMEXXeGPUToVC
  XeGPUToVC.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/XeGPUToVC

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
