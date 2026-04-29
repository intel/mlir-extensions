# Conditionally add obsolete VC backend transforms
if(IMEX_BUILD_VC_CONVERSIONS)
  set(VC_TRANSFORM_SOURCES
    RemoveSingleElemVector.cpp
    MaterializeMatrixOp.cpp
    OptimizeTranspose.cpp
    VnniTransformation.cpp
    HoistTranspose.cpp
  )
  message(STATUS "Including obsolete VC backend passes (RemoveSingleElemVector, MaterializeMatrixOp, OptimizeTranspose, VnniTransformation, HoistTranspose)")
else()
  set(VC_TRANSFORM_SOURCES "")
  message(STATUS "Excluding obsolete VC backend passes (set IMEX_BUILD_VC_CONVERSIONS=ON to enable)")
endif()

add_mlir_library(IMEXTransforms
  AddOuterParallelLoop.cpp
  BF16ToGPU.cpp
  CastIndex.cpp
  EmulateNonNativeBF16.cpp
  InsertGPUAllocs.cpp
  InsertGPUCopy.cpp
  LowerMemRefCopy.cpp
  RemoveTemporaries.cpp
  SerializeSPIRV.cpp
  SetSPIRVAbiAttribute.cpp
  SetSPIRVCapabilities.cpp
  TileLoops.cpp
  ${VC_TRANSFORM_SOURCES}

  PARTIAL_SOURCES_INTENDED

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
