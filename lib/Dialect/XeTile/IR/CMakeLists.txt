add_mlir_dialect_library(IMEXXeTileDialect
  XeTileOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/imex/Dialect/XeTile

  DEPENDS
  MLIRXeTileOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
