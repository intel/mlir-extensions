add_imex_dialect_library(IMEXNDArrayTransforms
  AddGPURegions.cpp
  CoalesceShardOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/imex/Dialect/NDArray

  DEPENDS
  IMEXNDArrayPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  IMEXNDArrayDialect
)
