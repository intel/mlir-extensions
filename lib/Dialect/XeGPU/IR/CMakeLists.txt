add_mlir_dialect_library(IMEXXeGPUDialect
  XeGPUOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/imex/Dialect/XeGPU

  DEPENDS
  MLIRXeGPUOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
