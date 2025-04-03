if ("SPIRV" IN_LIST LLVM_TARGETS_TO_BUILD)
  set(SPIRV_LIBS
    SPIRVCodeGen

  )
endif()

add_imex_dialect_library(MLIRXeVMTarget
  XeVM/Target.cpp

  OBJECT

  ADDITIONAL_HEADER_DIRS
  ${IMEX_SOURCE_DIR}/include/imex/Target/LLVM

  LINK_COMPONENTS
  ${SPIRV_LIBS}

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRExecutionEngineUtils
  MLIRSupport
  MLIRGPUDialect
  MLIRTargetLLVM
)

# Ensure SPIRV headers are included. Warning: references build directory!
target_include_directories(MLIRXeVMTarget PRIVATE
  ${LLVM_MAIN_SRC_DIR}/lib/Target/SPIRV
  ${LLVM_BINARY_DIR}/lib/Target/SPIRV
)
