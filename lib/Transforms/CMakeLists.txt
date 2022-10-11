add_mlir_library(IMEXTransforms
  SerializeSPIRV.cpp
  InsertGpuAllocs.cpp
  SetSPIRVCapabilities.cpp
  SetSPIRVAbiAttribute.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/imex/Transforms

  LINK_LIBS PUBLIC
  MLIRGPUOps
  MLIRSPIRVDialect
  MLIRFuncDialect
  MLIRPass
  MLIRSupport
  MLIRTransformUtils

  DEPENDS
  IMEXTransformsPassIncGen
)
