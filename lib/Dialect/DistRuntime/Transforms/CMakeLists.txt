add_imex_dialect_library(IMEXDistRuntimeTransforms
  DistRuntimeToIDTR.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/imex/Dialect/DistRuntime

  DEPENDS
  IMEXDistRuntimePassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
)
