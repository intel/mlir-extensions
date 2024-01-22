add_mlir_dialect_library(IMEXDistRuntimeDialect
  DistRuntimeOps.cpp
  GetHaloOp.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/imex/Dialect/DistRuntime

  DEPENDS
  MLIRDistRuntimeOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
