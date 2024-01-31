add_mlir_dialect_library(IMEXDistRuntimeTransforms
  DistRuntimeToIDTR.cpp
  OverlapCommAndCompute.cpp
  AddCommCacheKeys.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/imex/Dialect/DistRuntime

  DEPENDS
  IMEXDistRuntimePassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  IMEXDistRuntimeDialect
)
