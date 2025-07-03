add_imex_dialect_library(MLIRXeVMDialect
  IR/XeVMDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/LLVMIR
  ${PROJECT_SOURCE_DIR}/include/imex/Dialect/LLVMIR

  DEPENDS
  # MLIRGPUCompilationAttrInterfacesIncGen
  MLIRXeVMOpsIncGen
  MLIRXeVMConversionsIncGen
  # intrinsics_gen

  LINK_COMPONENTS
  AsmParser
  Core

  LINK_LIBS PUBLIC
  MLIRDialectUtils
  MLIRIR
  MLIRLLVMDialect
  MLIRSideEffectInterfaces
)
