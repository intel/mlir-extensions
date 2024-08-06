add_imex_dialect_library(IMEXXeTileTransforms
  Blocking.cpp
  InitDuplicate.cpp
  BlockAligning.cpp
  WgToSg.cpp
  OptimizeTranspose.cpp
  Canonicalization.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/imex/Dialect/XeTile

  DEPENDS
  IMEXXeTilePassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  IMEXXeTileDialect
)
