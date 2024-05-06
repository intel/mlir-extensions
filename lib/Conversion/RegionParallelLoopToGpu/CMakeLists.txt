add_imex_conversion_library(IMEXSCFToGPU
  RegionParallelLoopToGpu.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/imex/Conversion/SCFToGPU

  DEPENDS
  IMEXConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRSCFToGPU
)
