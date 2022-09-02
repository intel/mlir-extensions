add_mlir_library(IMEXTransforms
  InsertGpuAllocs.cpp
  SetSPIRVCapabalities.cpp
  SetSPIRVAbiAttribute.cpp

  #ADDITIONAL_HEADER_DIRS
  #${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms

  DEPENDS
  IMEXTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  )
