add_imex_dialect_library(IMEXDistRuntimeDialect
  DistRuntimeOps.cpp
  CopyReshapeOp.cpp
  CopyPermuteOp.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/imex/Dialect/DistRuntime

  DEPENDS
  MLIRDistRuntimeOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  IMEXDistRuntimeTransforms
)
