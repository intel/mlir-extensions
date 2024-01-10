add_mlir_dialect_library(IMEXRegionDialect
  RegionOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/imex/Dialect/Region

  DEPENDS
  MLIRRegionOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
