add_mlir_dialect_library(IMEXXeGPUDialect
  XeGPUDialect.cpp
  XeGPUOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/imex/Dialect/XeGPU

  DEPENDS
  MLIRXeGPUIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
