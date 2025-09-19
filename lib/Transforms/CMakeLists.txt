add_mlir_library(IMEXTransforms
  AddOuterParallelLoop.cpp
  BF16ToGPU.cpp
  CastIndex.cpp
  EmulateNonNativeBF16.cpp
  InsertGPUAllocs.cpp
  InsertGPUCopy.cpp
  LowerMemRefCopy.cpp
  RemoveSingleElemVector.cpp
  RemoveTemporaries.cpp
  SerializeSPIRV.cpp
  SetSPIRVAbiAttribute.cpp
  SetSPIRVCapabilities.cpp
  VectorLinearize.cpp
  VnniTransformation.cpp
  OptimizeTranspose.cpp
  HoistTranspose.cpp
  TileLoops.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/imex/Transforms

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSPIRVDialect
  MLIRSupport
  MLIRTransformUtils
  MLIRVectorTransforms

  DEPENDS
  IMEXTransformsPassIncGen
)
