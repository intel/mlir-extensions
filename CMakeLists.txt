
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
    src/pipelines/base_pipeline.cpp
    src/pipelines/lower_to_llvm.cpp
    src/pipelines/parallel_to_tbb.cpp
    src/pipelines/plier_to_linalg.cpp
    src/pipelines/plier_to_std.cpp
    src/lowering.cpp
    src/mangle.cpp
    src/py_func_resolver.cpp
    src/py_linalg_resolver.cpp
    src/py_map_types.cpp
    src/py_module.cpp
    )
set(HEADERS_LIST
    src/pipelines/base_pipeline.hpp
    src/pipelines/lower_to_llvm.hpp
    src/pipelines/parallel_to_tbb.hpp
    src/pipelines/plier_to_linalg.hpp
    src/pipelines/plier_to_std.hpp
    src/lowering.hpp
    src/mangle.hpp
    src/py_func_resolver.hpp
    src/py_linalg_resolver.hpp
    src/py_map_types.hpp
    src/py_module.hpp
    )

pybind11_add_module(${PROJECT_NAME} ${SOURCES_LIST} ${HEADERS_LIST})

if (MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /EHsc)
endif ()

if(UNIX)
    target_link_options(${PROJECT_NAME} PRIVATE "LINKER:--version-script=export.txt")
endif()

target_compile_definitions(${PROJECT_NAME} PRIVATE ${LLVM_DEFINITIONS})

target_link_libraries(${PROJECT_NAME} PRIVATE
    plier
    LLVM${LLVM_NATIVE_ARCH}CodeGen
    LLVM${LLVM_NATIVE_ARCH}Desc
    LLVMTarget
    MLIRIR
    MLIRLLVMIR
    MLIRLLVMToLLVMIRTranslation
    MLIRTransforms
    MLIRStandardOpsTransforms
    MLIRLinalgTransforms
    MLIRSCFToStandard
    MLIRTensorTransforms
    )

target_include_directories(${PROJECT_NAME} PRIVATE
    ./src
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
    )
