# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LLVM_LINK_COMPONENTS
    Core
    Support
    nativecodegen
    native
    )

add_llvm_executable(level_zero_runner LevelZeroRunner.cpp)

llvm_update_compile_flags(level_zero_runner)

target_link_libraries(level_zero_runner PRIVATE
    ${MLIR_EXTENSIONS_LIB}
    ${dialect_libs}
    ${conversion_libs}
    MLIROptLib
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
    )

target_include_directories(level_zero_runner PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
    )

install(TARGETS level_zero_runner
        PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
        )
