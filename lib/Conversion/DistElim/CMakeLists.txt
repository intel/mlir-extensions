add_mlir_conversion_library(IMEXDistElim
  DistElim.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/DistElim

  DEPENDS
  IMEXConversionPassIncGen

  LINK_LIBS PUBLIC
  IMEXDist
)
