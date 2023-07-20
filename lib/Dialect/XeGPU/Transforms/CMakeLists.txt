add_mlir_dialect_library(IMEXXeGPUTransforms
  XeGPUToSPIRVWithVCIntrinsics.cpp
  XeGPUToSPIRVWithJointMatrix.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/imex/Dialect/XeGPU

  DEPENDS
  IMEXXeGPUPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  IMEXXeGPUDialect
)
