add_imex_dialect_library(IMEXXeTileDialect
  XeTileOps.cpp
  XeTileDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/imex/Dialect/XeTile

  DEPENDS
  MLIRXeTileOpsIncGen
  XeTileOpsAttrsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
