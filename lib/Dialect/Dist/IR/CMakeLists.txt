add_mlir_dialect_library(IMEXDistDialect
  DistOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mlir/Dialect/Dist

  DEPENDS
  MLIRDistOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
