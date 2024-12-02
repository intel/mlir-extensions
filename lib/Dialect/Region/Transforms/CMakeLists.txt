add_imex_dialect_library(IMEXRegionTransforms
  BufferizableOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/imex/Dialect/Region

  LINK_LIBS PUBLIC
  IMEXRegionDialect
  MLIRPass
  MLIRIR
)
