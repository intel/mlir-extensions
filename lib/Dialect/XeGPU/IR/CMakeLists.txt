add_imex_dialect_library(IMEXXeGPUDialect
  XeGPUDialect.cpp
  XeGPUOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/imex/Dialect/XeGPU

  DEPENDS
  MLIRXeGPUIncGen
  MLIRXeGPUAttrsIncGen
  MLIRXeGPUEnumsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
