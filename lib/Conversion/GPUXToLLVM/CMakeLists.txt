add_imex_conversion_library(IMEXGPUXToLLVM
  GPUXToLLVMPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/GPUXToLLVM

  DEPENDS
  IMEXConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRGPUDialect
  MLIRIR
  MLIRPass
  MLIRSCFToSPIRV
  MLIRSupport
  MLIRTransforms
  )
