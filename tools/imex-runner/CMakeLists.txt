get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LLVM_LINK_COMPONENTS
    Core
    Support
    nativecodegen
    native
    )
set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    MLIROptLib
    MLIRAnalysis
    MLIRExecutionEngine
    MLIRIR
    MLIRJitRunner
    MLIRLLVMIR
    MLIRLLVMToLLVMIRTranslation
    MLIRToLLVMIRTranslationRegistration
    MLIRParser
    MLIRTargetLLVMIRExport
    MLIRSupport
    )

add_llvm_executable(imex-runner imex-runner.cpp)

llvm_update_compile_flags(imex-runner)
target_link_libraries(imex-runner PRIVATE ${LIBS})
