add_mlir_dialect_library(IMEXIntelGpuTransforms
  # FIXME.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/imex/Dialect/IntelGpu

  DEPENDS
  IMEXIntelGpuPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  IMEXIntelGpuDialect
)
