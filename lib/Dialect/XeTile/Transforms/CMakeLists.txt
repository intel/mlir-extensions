add_imex_dialect_library(IMEXXeTileTransforms
  Blocking.cpp
  BlockingAnalysis.cpp
  BlockOpFallback.cpp
  InitDuplicate.cpp
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
