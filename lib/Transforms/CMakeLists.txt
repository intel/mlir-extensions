add_mlir_library(IMEXTransforms
  SerializeSPIRV.cpp
  InsertGPUAllocs.cpp
  SetSPIRVCapabilities.cpp
  SetSPIRVAbiAttribute.cpp
  AddOuterParallelLoop.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/imex/Transforms

  LINK_LIBS PUBLIC
  MLIRSCFDialect
  MLIRGPUOps
  MLIRSPIRVDialect
  MLIRFuncDialect
  MLIRPass
  MLIRSupport
  MLIRTransformUtils

  DEPENDS
  IMEXTransformsPassIncGen
)
