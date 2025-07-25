add_mlir_library(IMEXUtil
  FuncUtils.cpp
  PassUtils.cpp
  TypeConversion.cpp
  VCUtils.cpp
  XeCommon.cpp
  XeArch.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/imex/Utils

  LINK_LIBS PUBLIC
  MLIRGPUDialect
  MLIRSPIRVDialect
  MLIRFuncDialect
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
)
