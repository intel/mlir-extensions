add_imex_dialect_library(IMEXDistTransforms
  DistCoalesce.cpp
  DistInferElementwiseCores.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/imex/Dialect/Dist

  DEPENDS
  IMEXDistPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  IMEXDistDialect
)
