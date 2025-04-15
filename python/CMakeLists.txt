if(IMEX_ENABLE_BINDINGS_PYTHON)
  if(NOT MLIR_ENABLE_BINDINGS_PYTHON)
    message(STATUS "Failed to enable Python API due to the 'MLIR_ENABLE_BINDINGS_PYTHON' for LLVM is not ON.")
    set(IMEX_ENABLE_BINDINGS_PYTHON OFF CACHE BOOL "" FORCE)
    return()
  endif()
  message(STATUS "Enabling Python API")
else()
  return()
endif()

include(MLIRDetectPythonEnv)
mlir_configure_python_dev_packages()
include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the `imex_mlir`
# top level package (the API has been embedded in a relocatable way).
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=imex_mlir.")


################################################################################
# Sources
################################################################################

declare_mlir_python_sources(ImexPythonSources)

declare_mlir_python_sources(ImexPythonSources.Common
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/imex_mlir"
  ADD_TO_PARENT ImexPythonSources
  SOURCES
    _mlir_libs/_site_initialize_0.py
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT ImexPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/imex_mlir"
  TD_FILE dialects/RegionOps.td
  SOURCES
    dialects/region.py
  DIALECT_NAME region)

declare_mlir_python_extension(ImexPythonSources.Extension
  MODULE_NAME _imex_mlir
  ADD_TO_PARENT ImexPythonSources
  SOURCES
    ImexExtension.cpp
  EMBED_CAPI_LINK_LIBS
    ImexCAPI
  PYTHON_BINDINGS_LIBRARY nanobind
)


################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(ImexPythonCAPI
  INSTALL_COMPONENT ImexPythonModules
  INSTALL_DESTINATION python_packages/imex_mlir/_mlir_libs
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/python_packages/imex_mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    ImexPythonSources
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
    MLIRPythonSources.ExecutionEngine
)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(ImexPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python_packages/imex_mlir"
  INSTALL_PREFIX "python_packages/imex_mlir"
  DECLARED_SOURCES
    ImexPythonSources
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources
    MLIRPythonSources.ExecutionEngine
  COMMON_CAPI_LINK_LIBS
    ImexPythonCAPI
  )
