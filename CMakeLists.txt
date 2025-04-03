cmake_minimum_required(VERSION 3.20.0)

# CMP0116: Ninja generators transform `DEPFILE`s from `add_custom_command()`
# New in CMake 3.20. https://cmake.org/cmake/help/latest/policy/CMP0116.html
if(POLICY CMP0116)
    cmake_policy(SET CMP0116 OLD)
endif()

set(IMEX_EXTERNAL_PROJECT_BUILD OFF)

set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -Wall -Werror -Wnon-virtual-dtor")

if(NOT (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR) AND NOT DEFINED MLIR_BINARY_DIR)
    # Building as part of LLVM via the external project mechanism.
    set(IMEX_EXTERNAL_PROJECT_BUILD ON)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
else()
    # Building standalone.
    project(mlir-extensions LANGUAGES CXX C)

    # Must go below project(..)
    include(GNUInstallDirs)

    set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
    list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")
endif()

# CMAKE_SYSTEM_NAME needs to be checked after "project"
if(${CMAKE_VERSION} VERSION_LESS 3.25)
    # CMake starting with version 3.25 sets var LINUX to true if target system is linux
    if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
        set(LINUX TRUE)
    endif()
endif()
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
if(LINUX)
    message(STATUS "Building for target: LINUX")
endif()


string(FIND "${LLVM_TARGETS_TO_BUILD}" "SPIRV" SPIRV_FOUND)
if(SPIRV_FOUND GREATER -1)
  message(STATUS "SPIRV is part of the LLVM targets")
  set(IMEX_SPIRV_BACKEND_ENABLED 1)
else()
  message(STATUS "SPIRV is not part of the LLVM targets")
  set(IMEX_SPIRV_BACKEND_ENABLED 0)
endif()

# option(IMEX_BUILD_SPIRV_BACKEND "Append SPIRV to LLVM_TARGETS_TO_BUILD" ON)
# if(IMEX_BUILD_SPIRV_BACKEND)
#   set(LLVM_TARGETS_TO_BUILD "${LLVM_TARGETS_TO_BUILD};SPIRV" CACHE STRING "LLVM targets to build" FORCE)
#   message(STATUS "IMEX adds SPIRV target to LLVM, LLVM_TARGETS_TO_BUILD = ${LLVM_TARGETS_TO_BUILD}")
# endif()

# Expected LLVM SHA
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/build_tools/llvm_version.txt EXPECTED_LLVM_SHA)
message(STATUS "Expected llvm sha: \"${EXPECTED_LLVM_SHA}\"")
option(IMEX_CHECK_LLVM_VERSION "Check expected llvm version." ON)

if(IMEX_EXTERNAL_PROJECT_BUILD)
    message(STATUS "Building IMEX with LLVM_EXTERNAL_PROJECTS")
    # Check if the llvm SHA is consistent with llvm_version.txt
    list(APPEND CMAKE_MODULE_PATH "${LLVM_MAIN_SRC_DIR}/cmake/modules")
    include(VersionFromVCS)
    get_source_info(${LLVM_MAIN_SRC_DIR}/.. LLVM_SHA repository)
    message(STATUS "Actual llvm sha:   \"${LLVM_SHA}\"")

    if (IMEX_CHECK_LLVM_VERSION AND (NOT EXPECTED_LLVM_SHA STREQUAL LLVM_SHA))
        message(FATAL_ERROR "Invalid llvm version")
    endif()

    # Variables needed for mlir_tablegen() to function
    set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir )
    set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include )

    set(LLVM_INCLUDE_DIR ${LLVM_MAIN_SRC_DIR}/include )

    # LLVM headers in source tree are already included
    # as part of LLVM_EXTERNAL_PROJECTS build

    # MLIR headers in source tree
    include_directories(${MLIR_INCLUDE_DIR})
    # Generated MLIR headers
    set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
    include_directories(${MLIR_GENERATED_INCLUDE_DIR})

    list(APPEND CMAKE_MODULE_PATH "${MLIR_MAIN_SRC_DIR}/cmake/modules")
else()
    message(STATUS "Building IMEX with external MLIR")
    find_package(MLIR REQUIRED CONFIG)

    # Check if the llvm SHA is consistent with llvm_version.txt
    file(STRINGS ${LLVM_INCLUDE_DIR}/llvm/Support/VCSRevision.h REVISION_FILE_DATA)
    message(DEBUG "VCSRevision: ${REVISION_FILE_DATA}")
    string(REGEX MATCH "\"([^\"]*)\"" LLVM_SHA ${REVISION_FILE_DATA})
    string(REPLACE "\"" "" LLVM_SHA ${LLVM_SHA})
    message(STATUS "llvm sha: \"${LLVM_SHA}\"")

    if (IMEX_CHECK_LLVM_VERSION AND (NOT EXPECTED_LLVM_SHA STREQUAL LLVM_SHA))
        message(FATAL_ERROR "Invalid llvm version")
    endif()

    message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
    message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

    # For redirecting llvm_add_library() to IMEX's bin and lib folders
    set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
    set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

    include_directories(${MLIR_INCLUDE_DIRS})
    if(DEFINED LLVM_BUILD_BINARY_DIR)
        message(STATUS "Building IMEX with external MLIR build tree")
        # LLVM_INCLUDE_DIRS needs to added if MLIR build tree is used.
        include_directories(${LLVM_INCLUDE_DIRS})
    else()
        message(STATUS "Building IMEX with external MLIR install tree")
        # LLVM_INCLUDE_DIRS is the same as MLIR_INCLUDE_DIRS in an MLIR install tree
    endif()

    list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
    list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
endif()

# For redirecting add_mlir_doc() to IMEX's doc folder
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

include(TableGen)
include(AddLLVM)
include(AddMLIR)

# Functions to generate IMEXConfig.cmake and IMEXTargets.cmake.
function(add_imex_library_install name)
  get_target_export_arg(${name} IMEX export_to_mlirtargets UMBRELLA imex-libraries)
  install(TARGETS ${name}
    COMPONENT ${name}
    ${export_to_mlirtargets}
    LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX}
    ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX}
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    # Note that CMake will create a directory like:
    #   objects-${CMAKE_BUILD_TYPE}/obj.LibName
    # and put object files there.
    OBJECTS DESTINATION lib${LLVM_LIBDIR_SUFFIX}
  )

  add_llvm_install_targets(install-${name}
                           DEPENDS ${name}
                           COMPONENT ${name})
  set_property(GLOBAL APPEND PROPERTY IMEX_ALL_LIBS ${name})
  set_property(GLOBAL APPEND PROPERTY IMEX_EXPORTS ${name})
endfunction()

# Declare the library associated with a dialect.
function(add_imex_dialect_library name)
  set_property(GLOBAL APPEND PROPERTY IMEX_DIALECT_LIBS ${name})
  add_mlir_library(${ARGV} DEPENDS mlir-headers)
endfunction(add_imex_dialect_library)

# Declare the library associated with a conversion.
function(add_imex_conversion_library name)
  set_property(GLOBAL APPEND PROPERTY IMEX_CONVERSION_LIBS ${name})
  add_mlir_library(${ARGV} DEPENDS mlir-headers)
endfunction(add_imex_conversion_library)

# Declare the library associated with an extension.
function(add_imex_extension_library name)
  set_property(GLOBAL APPEND PROPERTY IMEX_EXTENSION_LIBS ${name})
  add_mlir_library(${ARGV} DEPENDS mlir-headers)
endfunction(add_imex_extension_library)

# Declare the library associated with a translation.
function(add_imex_translation_library name)
  set_property(GLOBAL APPEND PROPERTY IMEX_TRANSLATION_LIBS ${name})
  add_mlir_library(${ARGV} DEPENDS mlir-headers)
endfunction(add_imex_translation_library)

macro(add_imex_tool name)
    add_llvm_executable(${name} ${ARGN})
    get_target_export_arg(${name} IMEX export_to_imexexports)
    install(TARGETS ${name}
            ${export_to_imexexports}
            RUNTIME DESTINATION ${IMEX_TOOLS_INSTALL_DIR}
            COMPONENT ${name})

    add_llvm_install_targets(install-${name}
                             DEPENDS ${name}
                             COMPONENT ${name})
    set_property(GLOBAL APPEND PROPERTY IMEX_EXPORTS ${name})
    set_target_properties(${name} PROPERTIES FOLDER "Tools")
endmacro()

include(HandleLLVMOptions)

set(IMEX_INCLUDE_TESTS 1 CACHE BOOL "Include targets for IMEX tests")
set(IMEX_ENABLE_SYCL_RUNTIME 0 CACHE BOOL "Enable the Sycl Runtime")
set(IMEX_ENABLE_L0_RUNTIME 0 CACHE BOOL "Enable the Level Zero Runtime")
set(IMEX_ENABLE_BENCHMARK 0 CACHE BOOL "Enable the IMEX Benchmark (Depending on SYCL Runtime)")
# Useful when building IMEX as an LLVM external project.
# LLVM Debug build is huge. This option override IMEX build type to Debug.
set(IMEX_ENABLE_DEBUG_BUILD 0 CACHE BOOL "Override CMAKE_BUILD_TYPE if not set to Debug")

if (IMEX_INCLUDE_TESTS)
    set(IMEX_INCLUDE_TESTS 1)
else ()
    set(IMEX_INCLUDE_TESTS 0)
endif()

# Normalize IMEX_ENABLE_SYCL_RUNTIME and IMEX_ENABLE_L0_RUNTIME
# These values are expected to be passed as 0 or 1 to imex-runner.py
if (IMEX_ENABLE_SYCL_RUNTIME)
    set(IMEX_ENABLE_SYCL_RUNTIME 1)
else ()
    set(IMEX_ENABLE_SYCL_RUNTIME 0)
endif()

if (IMEX_ENABLE_L0_RUNTIME)
    set(IMEX_ENABLE_L0_RUNTIME 1)
else ()
    set(IMEX_ENABLE_L0_RUNTIME 0)
endif()

if (IMEX_ENABLE_BENCHMARK)
    set(IMEX_ENABLE_BENCHMARK 1)
else ()
    set(IMEX_ENABLE_BENCHMARK 0)
endif()

if (IMEX_ENABLE_DEBUG_BUILD)
    set(IMEX_ENABLE_DEBUG_BUILD 1)
else ()
    set(IMEX_ENABLE_DEBUG_BUILD 0)
endif()

# Using flags to control bf16 tests since we don't have a good way to test
# if GPU supports bf16
set(IMEX_ENABLE_BF16_TESTS 1 CACHE BOOL "Enable bf16 execution tests")

if (IMEX_ENABLE_BF16_TESTS)
    set(IMEX_ENABLE_BF16_TESTS 1)
else ()
    set(IMEX_ENABLE_BF16_TESTS 0)
endif()

# Using flags to control running tests that are excluded for various reasons
set(IMEX_ENABLE_EXCLUDED_TESTS 0 CACHE BOOL "Enable excluded tests")

if (IMEX_ENABLE_EXCLUDED_TESTS)
    set(IMEX_ENABLE_EXCLUDED_TESTS 1)
else ()
    set(IMEX_ENABLE_EXCLUDED_TESTS 0)
endif()

# Check if mlir vulkan runner was configured from mlir
# target mlir-vulkan-runner is only set if vulkan runner was enabled during
# mlir configuration
if (TARGET mlir-vulkan-runner)
    set(IMEX_ENABLE_VULKAN_RUNNER 1)
else ()
    set(IMEX_ENABLE_VULKAN_RUNNER 0)
endif()

# Run tests using FS simulator
set(IMEX_ENABLE_FS_SIMULATOR 0 CACHE BOOL "Enable FS simulator mode for tests")

if (IMEX_ENABLE_FS_SIMULATOR)
    set(IMEX_ENABLE_FS_SIMULATOR 1)
else ()
    set(IMEX_ENABLE_FS_SIMULATOR 0)
endif()

if (IMEX_ENABLE_ATS_TARGET)
    set(IMEX_ENABLE_ATS_TARGET 1)
else ()
    set(IMEX_ENABLE_ATS_TARGET 0)
endif()

if (IMEX_ENABLE_PVC_TARGET)
    set(IMEX_ENABLE_PVC_TARGET 1)
else ()
    set(IMEX_ENABLE_PVC_TARGET 0)
endif()

if (NOT (CMAKE_BUILD_TYPE MATCHES Debug))
    if (IMEX_ENABLE_DEBUG_BUILD)
        set (CMAKE_BUILD_TYPE Debug)
    endif()
endif()

if(NOT IMEX_EXTERNAL_PROJECT_BUILD)
    find_package(Python3 ${LLVM_MINIMUM_PYTHON_VERSION} REQUIRED
        COMPONENTS Interpreter)
endif()

# IMEX headers in source tree
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
# Generated IMEX headers
set(IMEX_GENERATED_HEADER_DIR ${CMAKE_CURRENT_BINARY_DIR})
include_directories(${IMEX_GENERATED_HEADER_DIR}/include)
message(STATUS "LLVM_DEFINITIONS: ${LLVM_DEFINITIONS}")
add_definitions(${LLVM_DEFINITIONS})

set(LLVM_LIT_ARGS "-sv" CACHE STRING "lit default options")
if (IMEX_ENABLE_SYCL_RUNTIME OR IMEX_ENABLE_L0_RUNTIME)
    set(LLVM_LIT_ARGS "-j 4 ${LLVM_LIT_ARGS}") # do not stress GPU
endif()

set(IMEX_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
# LLVM_EXTERNAL_PROJECTS build puts library, executables and tools in LLVM's CMAKE_BINARY_DIR
set(IMEX_BINARY_DIR ${CMAKE_BINARY_DIR})
if(WIN32)
    set(IMEX_LIB_DIR ${IMEX_BINARY_DIR}/bin)
else()
    set(IMEX_LIB_DIR ${IMEX_BINARY_DIR}/lib)
endif()

set(IMEX_TOOLS_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}" CACHE STRING
    "Path for binary subdirectory (defaults to '${CMAKE_INSTALL_BINDIR}')")

include(sanitizers)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
if(IMEX_INCLUDE_TESTS)
    add_subdirectory(test)
endif()

if((IMEX_ENABLE_BENCHMARK) AND (IMEX_ENABLE_SYCL_RUNTIME OR IMEX_ENABLE_L0_RUNTIME))
    add_subdirectory(benchmarks)
endif()

option(IMEX_INCLUDE_DOCS "Generate build targets for the IMEX docs." ON)
if (IMEX_INCLUDE_DOCS)
    add_subdirectory(docs)
endif()

# Custom target to install all imex headers
add_custom_target(imex-headers)
set_target_properties(imex-headers PROPERTIES FOLDER "Misc")

# Headers in source tree
install(DIRECTORY include/imex
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT imex-headers
    FILES_MATCHING
    PATTERN "*.def"
    PATTERN "*.h"
    PATTERN "*.inc"
    PATTERN "*.td"
    PATTERN "LICENSE.TXT"
)

# Generated headers
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/imex
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT imex-headers
    FILES_MATCHING
    PATTERN "*.def"
    PATTERN "*.h"
    PATTERN "*.gen"
    PATTERN "*.inc"
    PATTERN "*.td"
    PATTERN "CMakeFiles" EXCLUDE
    PATTERN "config.h" EXCLUDE
)

add_llvm_install_targets(install-imex-headers
                         DEPENDS imex-headers
                         COMPONENT imex-headers)


# Custom target to install all imex libraries
add_custom_target(imex-libraries)
set_target_properties(imex-libraries PROPERTIES FOLDER "Misc")

add_llvm_install_targets(install-imex-libraries
                         DEPENDS imex-libraries
                         COMPONENT imex-libraries)

get_property(IMEX_LIBS GLOBAL PROPERTY IMEX_ALL_LIBS)
if(IMEX_LIBS)
  list(REMOVE_DUPLICATES IMEX_LIBS)
  foreach(lib ${IMEX_LIBS})
    add_dependencies(imex-libraries ${lib})
    add_dependencies(install-imex-libraries install-${lib})
  endforeach()
endif()

add_subdirectory(cmake/modules)
