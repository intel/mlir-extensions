
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

add_subdirectory(include/plier)

set(SOURCES_LIST
    src/compiler/compiler.cpp
    src/compiler/pipeline_registry.cpp
    src/dialect.cpp
    src/rewrites/call_lowering.cpp
    src/rewrites/canonicalize_reductions.cpp
    src/rewrites/cast_lowering.cpp
    src/rewrites/common_opts.cpp
    src/rewrites/cse.cpp
    src/rewrites/force_inline.cpp
    src/rewrites/index_type_propagation.cpp
    include/plier/rewrites/if_rewrites.cpp
    src/rewrites/loop_rewrites.cpp
    src/rewrites/promote_to_parallel.cpp
    src/rewrites/type_conversion.cpp
    src/transforms/cast_utils.cpp
    src/transforms/const_utils.cpp
    src/transforms/func_utils.cpp
    src/transforms/loop_utils.cpp
    src/transforms/pipeline_utils.cpp
    src/utils.cpp
    )
set(HEADERS_LIST
    include/plier/compiler/compiler.hpp
    include/plier/compiler/pipeline_registry.hpp
    include/plier/dialect.hpp
    include/plier/PlierOps.td
    include/plier/rewrites/call_lowering.hpp
    include/plier/rewrites/canonicalize_reductions.hpp
    include/plier/rewrites/cast_lowering.hpp
    include/plier/rewrites/common_opts.hpp
    include/plier/rewrites/cse.hpp
    include/plier/rewrites/force_inline.hpp
    include/plier/rewrites/index_type_propagation.hpp
    include/plier/rewrites/if_rewrites.hpp
    include/plier/rewrites/loop_rewrites.hpp
    include/plier/rewrites/promote_to_parallel.hpp
    include/plier/rewrites/type_conversion.hpp
    include/plier/transforms/cast_utils.hpp
    include/plier/transforms/const_utils.hpp
    include/plier/transforms/func_utils.hpp
    include/plier/transforms/loop_utils.hpp
    include/plier/transforms/pipeline_utils.hpp
    include/plier/utils.hpp
    )

set(PLIER_LIB "plier")

add_library(${PLIER_LIB} STATIC ${SOURCES_LIST} ${HEADERS_LIST})

if (MSVC)
    target_compile_options(${PLIER_LIB} PRIVATE /EHsc)
endif ()

target_compile_definitions(${PLIER_LIB} PRIVATE ${LLVM_DEFINITIONS})

target_link_libraries(${PLIER_LIB} PRIVATE
    MLIRIR
    MLIRLLVMIR
    MLIRTargetLLVMIR
    MLIRTransforms
    MLIRStandardOpsTransforms
    MLIRLinalgTransforms
    MLIRSCFToStandard
    MLIRTensorTransforms
    )

target_include_directories(${PLIER_LIB} PRIVATE
    ./src
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
    )

target_include_directories(${PLIER_LIB} PUBLIC
    ./include
    ${PROJECT_BINARY_DIR}/plier/include
    )

add_dependencies(${PLIER_LIB} MLIRPlierOpsIncGen)
