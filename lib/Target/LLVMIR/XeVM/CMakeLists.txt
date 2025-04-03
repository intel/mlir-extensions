add_imex_translation_library(MLIRXeVMToLLVMIRTranslation
  XeVMToLLVMIRTranslation.cpp

  DEPENDS
  MLIRXeVMConversionsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRXeVMDialect
  MLIRSupport
  MLIRTargetLLVMIRExport
)
