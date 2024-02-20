add_mlir_library(IMEXTransforms
  AddOuterParallelLoop.cpp
  BF16ToGPU.cpp
  InsertGPUAllocs.cpp
  LowerMemRefCopy.cpp
  RemoveTemporaries.cpp
  SerializeSPIRV.cpp
  SetSPIRVAbiAttribute.cpp
  SetSPIRVCapabilities.cpp
  VectorLinearize.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/imex/Transforms

  LINK_LIBS PUBLIC
  MLIRSCFDialect
  MLIRGPUDialect
  MLIRSPIRVDialect
  MLIRFuncDialect
  MLIRPass
  MLIRSupport
  MLIRTransformUtils

  DEPENDS
  IMEXTransformsPassIncGen
)
