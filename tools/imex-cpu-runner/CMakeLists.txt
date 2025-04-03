set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  native
  )

add_imex_tool(imex-cpu-runner
  imex-cpu-runner.cpp
  )
llvm_update_compile_flags(imex-cpu-runner)

if (CMAKE_SYSTEM_NAME STREQUAL Linux)
    target_link_options(imex-cpu-runner PRIVATE "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/unexported_symbols.txt")
endif()

target_link_libraries(imex-cpu-runner PRIVATE
  MLIRAnalysis
  MLIRExecutionEngine
  MLIRIR
  MLIRJitRunner
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRToLLVMIRTranslationRegistration
  MLIRParser
  MLIRTargetLLVMIRExport
  MLIRSupport
  MLIRXeVMToLLVMIRTranslation
  )
