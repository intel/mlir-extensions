add_imex_dialect_library(MLIRXeVMTarget
  XeVM/Target.cpp

  OBJECT

  ADDITIONAL_HEADER_DIRS
  ${IMEX_SOURCE_DIR}/include/imex/Target/LLVM

  LINK_COMPONENTS
  SPIRVCodeGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRExecutionEngineUtils
  MLIRSupport
  MLIRGPUDialect
  MLIRTargetLLVM
)

# Ensure SPIRV headers are included
target_include_directories(MLIRXeVMTarget PRIVATE
  ${LLVM_INCLUDE_DIR}/../lib/Target/SPIRV
  ${LLVM_BINARY_DIR}/lib/Target/SPIRV
)
