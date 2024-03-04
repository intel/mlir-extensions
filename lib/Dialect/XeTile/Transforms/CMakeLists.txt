add_imex_dialect_library(IMEXXeTileTransforms
  Blocking.cpp
  InitDuplicate.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/imex/Dialect/XeTile

  DEPENDS
  IMEXXeTilePassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  IMEXXeTileDialect
)
