add_mlir_library(IMEXUtil
    FuncUtils.cpp
    TypeConversion.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/imex/Util

  LINK_LIBS PUBLIC
  MLIRGPUOps
  MLIRSPIRVDialect
  MLIRFuncDialect
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
)
