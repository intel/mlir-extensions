add_mlir_dialect_library(IMEXGPUXDialect
  GPUXOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/imex/Dialect/GPUX

  DEPENDS
  MLIRGPUXOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
