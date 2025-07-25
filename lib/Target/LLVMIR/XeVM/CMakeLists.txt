add_imex_translation_library(MLIRXeVMToLLVMIRTranslation
  XeVMToLLVMIRTranslation.cpp

  DEPENDS
  MLIRXeVMDialect

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRXeVMDialect
  MLIRSupport
  MLIRTargetLLVMIRExport
)
