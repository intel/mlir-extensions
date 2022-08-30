add_mlir_dialect_library(IMEXIntelGpuDialect
  IntelGpuOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/imex/Dialect/IntelGpu

  DEPENDS
  MLIRIntelGpuOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
