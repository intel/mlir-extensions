add_imex_dialect_library(IMEXRegionTransforms
  RegionConversions.cpp
  RegionBufferize.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/imex/Dialect/Region

  DEPENDS
  IMEXRegionPassIncGen

  LINK_LIBS PUBLIC
  IMEXRegionDialect
  MLIRPass
  MLIRIR
)
