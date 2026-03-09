# Conditionally add XeGPU layout transform sources
if(IMEX_ENABLE_XEGPU_LAYOUT_PASSES)
  set(XEGPU_LAYOUT_SOURCES
    MaterializeMatrixOp.cpp
    OptimizeTranspose.cpp
  )
  message(STATUS "Including XeGPU layout transformation passes")
else()
  set(XEGPU_LAYOUT_SOURCES "")
  message(STATUS "Excluding XeGPU layout transformation passes (set IMEX_ENABLE_XEGPU_LAYOUT_PASSES=ON to enable)")
endif()

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
  VnniTransformation.cpp
  HoistTranspose.cpp
  TileLoops.cpp
  ${XEGPU_LAYOUT_SOURCES}

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
