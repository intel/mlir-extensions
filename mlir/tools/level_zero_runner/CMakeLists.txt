# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
