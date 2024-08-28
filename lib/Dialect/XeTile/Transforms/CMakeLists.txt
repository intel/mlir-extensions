add_imex_dialect_library(IMEXXeTileTransforms
  BlockAligning.cpp
  Blocking.cpp
  BlockingAnalysis.cpp
  BlockingRewrite.cpp
  InitDuplicate.cpp
  OptimizeTranspose.cpp
  Canonicalization.cpp
  WgToSg.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/imex/Dialect/XeTile

  DEPENDS
  IMEXXeTilePassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  IMEXXeTileDialect
)
