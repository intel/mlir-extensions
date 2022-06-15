add_mlir_conversion_library(MLIRDistElim
  DistElim.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/DistElim

  LINK_LIBS PUBLIC
  MLIRDist
  )
