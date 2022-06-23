add_mlir_dialect_library(IMEXDistTransforms
  DistElim.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/imex/Dialect/Dist

  DEPENDS
  IMEXDistPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  IMEXDistDialect
)
