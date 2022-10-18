add_mlir_dialect_library(IMEXPTensorTransforms
  PTensorDist.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/imex/Dialect/PTensor

  DEPENDS
  IMEXPTensorPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  IMEXPTensorDialect
  IMEXDistDialect
)
