include(LLVMDistributionSupport)

if(NOT IMEX_EXTERNAL_PROJECT_BUILD)
# ----------- Copied over from llvm-project/cmake/Modules ---------------------#

# Extend the path in `base_path` with the path in `current_segment`, returning
# the result in `joined_path`. If `current_segment` is an absolute path then
# just return it, in effect overriding `base_path`, and issue a warning.
#
# Note that the code returns a relative path (avoiding introducing leading
# slashes) if `base_path` is empty.
function(extend_path joined_path base_path current_segment)
    if("${current_segment}" STREQUAL "")
        set(temp_path "${base_path}")
    elseif("${base_path}" STREQUAL "")
        set(temp_path "${current_segment}")
    elseif(IS_ABSOLUTE "${current_segment}")
        message(WARNING
            "Since \"${current_segment}\" is absolute, it "
            "overrides base path: \"${base_path}\"."
        )
        set(temp_path "${current_segment}")
    else()
        set(temp_path "${base_path}/${current_segment}")
    endif()

    set(${joined_path} "${temp_path}" PARENT_SCOPE)
endfunction()

# Find the prefix from the `*Config.cmake` file being generated.
#
# When generating an installed `*Config.cmake` file, we often want to be able
# to refer to the ancestor directory which contains all the installed files.
#
# We want to do this without baking in an absolute path when the config file is
# generated, in order to allow for a "relocatable" binary distribution that
# doesn't need to know what path it ends up being installed at when it is
# built.
#
# The solution that we know the relative path that the config file will be at
# within that prefix, like `"${prefix_var}/lib/cmake/${project}"`, so we count
# the number of components in that path to figure out how many parent dirs we
# need to traverse from the location of the config file to get to the prefix
# dir.
#
# out_var:
# variable to set the "return value" of the function, which is the code to
# include in the config file under construction.
#
# prefix_var:
# Name of the variable to define in the returned code (not directory for the
# faller!) that will contain the prefix path.
#
# path_to_leave:
# Path from the prefix to the config file, a relative path which we wish to
# go up and out from to find the prefix directory.
function(find_prefix_from_config out_var prefix_var path_to_leave)
    set(config_code
        "# Compute the installation prefix from this LLVMConfig.cmake file location."
        "get_filename_component(${prefix_var} \"\${CMAKE_CURRENT_LIST_FILE}\" PATH)")

    # Construct the proper number of get_filename_component(... PATH)
    # calls to compute the installation prefix.
    string(REGEX REPLACE "/" ";" _count "${path_to_leave}")

    foreach(p ${_count})
        list(APPEND config_code
            "get_filename_component(${prefix_var} \"\${${prefix_var}}\" PATH)")
    endforeach(p)

    string(REPLACE ";" "\n" config_code "${config_code}")
    set("${out_var}" "${config_code}" PARENT_SCOPE)
endfunction()
endif() # if(NOT IMEX_EXTERNAL_PROJECT_BUILD)

# -----------------------------------------------------------------------------#

# Generate a list of CMake library targets so that other CMake projects can
# link against them. LLVM calls its version of this file LLVMExports.cmake, but
# the usual CMake convention seems to be ${Project}Targets.cmake.
set(IMEX_INSTALL_PACKAGE_DIR lib${LLVM_LIBDIR_SUFFIX}/cmake/imex)
set(imex_cmake_builddir "${IMEX_BINARY_DIR}/${IMEX_INSTALL_PACKAGE_DIR}")

# Export build tree targets
get_property(IMEX_EXPORTS GLOBAL PROPERTY IMEX_EXPORTS)
export(TARGETS ${IMEX_EXPORTS} FILE ${imex_cmake_builddir}/IMEXTargets.cmake)

get_property(IMEX_ALL_LIBS GLOBAL PROPERTY IMEX_ALL_LIBS)
get_property(IMEX_DIALECT_LIBS GLOBAL PROPERTY IMEX_DIALECT_LIBS)
get_property(IMEX_CONVERSION_LIBS GLOBAL PROPERTY IMEX_CONVERSION_LIBS)
get_property(IMEX_EXTENSION_LIBS GLOBAL PROPERTY IMEX_EXTENSION_LIBS)
get_property(IMEX_TRANSLATION_LIBS GLOBAL PROPERTY IMEX_TRANSLATION_LIBS)

# Generate IMEXConfig.cmake for the build tree.
set(IMEX_CONFIG_CMAKE_DIR "${imex_cmake_builddir}")
set(IMEX_CONFIG_INCLUDE_EXPORTS "include(\"\${IMEX_CMAKE_DIR}/IMEXTargets.cmake\")")
set(IMEX_CONFIG_INCLUDE_DIRS
    "${IMEX_SOURCE_DIR}/include"
    "${IMEX_BINARY_DIR}/include"
    "${IMEX_GENERATED_HEADER_DIR}/include"
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/IMEXConfig.cmake.in
    ${imex_cmake_builddir}/IMEXConfig.cmake
    @ONLY)
set(IMEX_CONFIG_CMAKE_DIR)
set(IMEX_CONFIG_INCLUDE_DIRS)

# For compatibility with projects that include(IMEXConfig)
# via CMAKE_MODULE_PATH, place API modules next to it.
# This should be removed in the future.
file(COPY .
    DESTINATION ${imex_cmake_builddir}
    FILES_MATCHING PATTERN *.cmake
    PATTERN CMakeFiles EXCLUDE
)

message(STATUS "IMEX_INSTALL_PACKAGE_DIR ${IMEX_INSTALL_PACKAGE_DIR}")

# Generate IMEXConfig.cmake for the install tree.
find_prefix_from_config(IMEX_CONFIG_CODE IMEX_INSTALL_PREFIX "${IMEX_INSTALL_PACKAGE_DIR}")
set(IMEX_CONFIG_CMAKE_DIR "\${IMEX_INSTALL_PREFIX}/${IMEX_INSTALL_PACKAGE_DIR}")
get_config_exports_includes(IMEX IMEX_CONFIG_INCLUDE_EXPORTS)
extend_path(base_includedir "\${IMEX_INSTALL_PREFIX}" "${CMAKE_INSTALL_INCLUDEDIR}")
set(IMEX_CONFIG_INCLUDE_DIRS
    "${base_includedir}"
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/IMEXConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/IMEXConfig.cmake
    @ONLY)
set(IMEX_CONFIG_CODE)
set(IMEX_CONFIG_CMAKE_DIR)
set(IMEX_CONFIG_INCLUDE_DIRS)

install_distribution_exports(IMEX)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/IMEXConfig.cmake
    DESTINATION ${IMEX_INSTALL_PACKAGE_DIR}
    COMPONENT imex-cmake-exports
)
