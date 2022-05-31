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

project(mlir_compiler LANGUAGES CXX C)

find_package(pybind11 REQUIRED)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

set(SOURCES_LIST
    lib/pipelines/base_pipeline.cpp
    lib/pipelines/lower_to_gpu.cpp
    lib/pipelines/lower_to_llvm.cpp
    lib/pipelines/parallel_to_tbb.cpp
    lib/pipelines/plier_to_linalg.cpp
    lib/pipelines/plier_to_std.cpp
    lib/pipelines/plier_to_scf.cpp
    lib/pipelines/pre_low_simplifications.cpp
    lib/loop_utils.cpp
    lib/lowering.cpp
    lib/mangle.cpp
    lib/py_func_resolver.cpp
    lib/py_linalg_resolver.cpp
    lib/py_map_types.cpp
    lib/py_module.cpp
    )
set(HEADERS_LIST
    lib/pipelines/base_pipeline.hpp
    lib/pipelines/lower_to_gpu.hpp
    lib/pipelines/lower_to_llvm.hpp
    lib/pipelines/parallel_to_tbb.hpp
    lib/pipelines/plier_to_linalg.hpp
    lib/pipelines/plier_to_std.hpp
    lib/pipelines/plier_to_scf.hpp
    lib/pipelines/pre_low_simplifications.hpp
    lib/loop_utils.hpp
    lib/lowering.hpp
    lib/mangle.hpp
    lib/py_func_resolver.hpp
    lib/py_linalg_resolver.hpp
    lib/py_map_types.hpp
    lib/py_module.hpp
    )

pybind11_add_module(${PROJECT_NAME} ${SOURCES_LIST} ${HEADERS_LIST})

if (CMAKE_SYSTEM_NAME STREQUAL Linux)
    target_link_options(${PROJECT_NAME} PRIVATE "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/export.txt")
endif()

if (CMAKE_SYSTEM_NAME STREQUAL Darwin)
    target_link_libraries(${PROJECT_NAME} PRIVATE "-Wl,-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/export_darwin.txt")
endif()

apply_llvm_compile_flags(${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME} PRIVATE
    mlir-extensions
    LLVM${LLVM_NATIVE_ARCH}CodeGen
    LLVM${LLVM_NATIVE_ARCH}Desc
    LLVMTarget
    MLIRIR
    MLIRLLVMIR
    MLIRLLVMToLLVMIRTranslation
    MLIRTransforms
    MLIRFuncTransforms
    MLIRLinalgTransforms
    MLIRLinalgToLLVM
    MLIRMathToLLVM
    MLIRMathToLibm
    MLIRSCFToGPU
    MLIRGPUToSPIRV
    MLIRGPUToGPURuntimeTransforms
    MLIRSPIRVTransforms
    MLIRSPIRVSerialization
    MLIRTensorTransforms
    MLIRReconcileUnrealizedCasts
    )

target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE
    ${MLIR_INCLUDE_DIRS}
    PRIVATE
    ./lib)

if(IMEX_USE_DPNP)
    target_compile_definitions(${PROJECT_NAME} PRIVATE IMEX_USE_DPNP=1)
endif()

if(IMEX_ENABLE_IGPU_DIALECT)
    target_compile_definitions(${PROJECT_NAME} PRIVATE IMEX_ENABLE_IGPU_DIALECT=1)
endif()

set(CMAKE_INSTALL_BINDIR ./numba_dpcomp/numba_dpcomp)
set(CMAKE_INSTALL_LIBDIR ./numba_dpcomp/numba_dpcomp)
set(CMAKE_INSTALL_INCLUDEDIR ./numba_dpcomp/numba_dpcomp)

install(TARGETS dpcomp-runtime dpcomp-math-runtime dpcomp-python-runtime mlir_compiler
        LIBRARY DESTINATION numba_dpcomp/numba_dpcomp
        )

if(IMEX_ENABLE_IGPU_DIALECT)
    install(TARGETS dpcomp-gpu-runtime
            LIBRARY DESTINATION numba_dpcomp/numba_dpcomp
            )
endif()

get_filename_component(_tmp_path1 "${CMAKE_SOURCE_DIR}" REALPATH)
get_filename_component(_tmp_path2 "${CMAKE_INSTALL_PREFIX}" REALPATH)

if (NOT _tmp_path1 STREQUAL _tmp_path2)
    install(
        DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../
        DESTINATION ./numba_dpcomp
        FILES_MATCHING
        PATTERN "__pycache__" EXCLUDE
        PATTERN "*.py"
        PATTERN "*.cfg"
        PATTERN "*.in"
        )
endif()
