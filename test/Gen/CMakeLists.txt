# How it works:
# - "Gen" is the root of all generated test cases
# - recursively crawl entire source tree under "Gen" by calling
#       "add_subdirectory" for all child directories
# - copy over file structure except .mlir.in
# - "*.mlir.in" is a template format for generating multi-variant test cases.
# - First line of the file needs to provide in comment(//) the following,
#       1) Number of placeholder - (Syntax: NUMPLACEHOLDERS <num>)
#       2) Number of variants for each placeholders - (Syntax: NUMVARIANTS <num>)
#   For example,
#   // NUMPLACEHOLDERS 2 NUMVARIANTS 3
#   Means two placeholder and each placeholder has three variants
# - The first line is followed by one line per placeholder of variant definitions
#   (Syntax: // PLACEHOLDER <name> <var1> .. <varN>
#   For example,
#   // PLACEHOLDER DTYPE f32 f16 bf16
#   // PLACEHOLDER PRINTDTYPE @printMemrefF32 @printMemrefF16 @printMemrefBF16

if(POLICY CMP0121)
    cmake_policy(SET CMP0121 OLD)
endif()

get_filename_component(GEN_TEST_ROOT ${CMAKE_CURRENT_BINARY_DIR}/../.. ABSOLUTE)
# Set the root of source and binary directory for generated test cases
# parallel to regular test binary directory
set(GEN_TEST_SRC_ROOT ${GEN_TEST_ROOT}/gen_test_src)
set(GEN_TEST_BIN_ROOT ${GEN_TEST_ROOT}/gen_test_bin)
file(MAKE_DIRECTORY ${GEN_TEST_SRC_ROOT})
file(MAKE_DIRECTORY ${GEN_TEST_BIN_ROOT})
set(GEN_TEST_PREFIX ${CMAKE_CURRENT_SOURCE_DIR})

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/../lit.cfg.py
    ${GEN_TEST_SRC_ROOT}/lit.cfg.py COPYONLY)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/../lit.site.cfg.py.in
    ${GEN_TEST_SRC_ROOT}/lit.site.cfg.py.in COPYONLY)
set(IMEX_LIT_CFG_PY ${GEN_TEST_SRC_ROOT}/lit.cfg.py)
configure_lit_site_cfg(
    ${GEN_TEST_SRC_ROOT}/lit.site.cfg.py.in
    ${GEN_TEST_BIN_ROOT}/lit.site.cfg.py
    MAIN_CONFIG
    ${IMEX_LIT_CFG_PY}
    )

function(imex_gen_set_num_ph_var)
    set(oneValueArgs NUMPLACEHOLDERS NUMVARIANTS)
    cmake_parse_arguments(GEN_SET "" ${oneValueArgs} "" ${ARGN})
    set(genNumPlaceholders ${GEN_SET_NUMPLACEHOLDERS} PARENT_SCOPE)
    set(genNumVariants ${GEN_SET_NUMVARIANTS} PARENT_SCOPE)
endfunction()

function(imex_gen_set_ph)
    set(multiValueArgs PLACEHOLDER)
    cmake_parse_arguments(GEN_SET "" "" ${multiValueArgs} ${ARGN})
    set(genPlaceholder ${GEN_SET_PLACEHOLDER} PARENT_SCOPE)
endfunction()

function(imex_gen_test src_base tgt_base)
    # Make target directory for generated files
    string(REPLACE ${src_base} ${tgt_base} _TGT_PATH ${CMAKE_CURRENT_SOURCE_DIR})
    file(MAKE_DIRECTORY ${_TGT_PATH})
    # Copy pass pipeline files
    file(GLOB PP_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.pp)
    foreach(_PP ${PP_FILES})
        get_filename_component(_PP ${_PP} NAME)
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/${_PP}
            ${_TGT_PATH}/${_PP} COPYONLY)
    endforeach()
    # Copy configuration files
    file(GLOB CFG_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cfg)
    foreach(_CFG ${CFG_FILES})
        get_filename_component(_CFG ${_CFG} NAME)
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/${_CFG}
            ${_TGT_PATH}/${_CFG} COPYONLY)
    endforeach()
    # Process templates and copy generated test cases
    file(GLOB SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.mlir.in)
    foreach(_SRC ${SRC_FILES})
        # initialize counts
        set(genNumPlaceholders 0)
        set(genNumVariants 0)
        # initialize placeholder name list
        set(genPlaceholderList "")
        get_filename_component(_SRC ${_SRC} NAME)
        file(STRINGS ${_SRC} _LINES)
        foreach(_LINE ${_LINES})
            # Remove "//", leading and trailing spaces
            string(REPLACE "//" "" _LINE ${_LINE})
            string(STRIP ${_LINE} _LINE)
            #message(STATUS "Line to process: ${_LINE}")
            if(_LINE MATCHES "^NUMPLACEHOLDERS")
                string(REPLACE " " ";" _LINE_ARGS ${_LINE})
                imex_gen_set_num_ph_var(${_LINE_ARGS})
            elseif(_LINE MATCHES "^PLACEHOLDER")
                string(REPLACE " " ";" _LINE_ARGS ${_LINE})
                imex_gen_set_ph(${_LINE_ARGS})
                list(GET genPlaceholder 0 PH_NAME)
                list(REMOVE_ITEM genPlaceholder ${PH_NAME})
                list(LENGTH genPlaceholder phLen)
                if(NOT (phLen EQUAL genNumVariants))
                    message(FATAL_ERROR "Num elements in ${genPlaceholder} does not match ${genNumVariants}")
                endif()
                list(APPEND genPlaceholderList "${PH_NAME}")
                set("${PH_NAME}_LIST" "${genPlaceholder}")
            else()
                break()
            endif()
        endforeach()
        list(LENGTH genPlaceholderList phListLen)
        if(NOT (phListLen EQUAL genNumPlaceholders))
            message(FATAL_ERROR "Num placeholders in ${genPlaceholderList} does no match ${genNumPlaceholders}")
        endif()
        set(genLoopCount 0)
        while(genLoopCount LESS genNumVariants)
            # assign default extension
            set(_TGT_EXT "${genLoopCount}")
            foreach(PH_NAME ${genPlaceholderList})
                list(GET "${PH_NAME}_LIST" ${genLoopCount} PH)
                # assign better extension name if DTYPE is given
                if("${PH_NAME}" STREQUAL "DTYPE")
                    set(_TGT_EXT ${PH})
                endif()
                set("${PH_NAME}" "${PH}")
            endforeach()
            string(REPLACE "mlir.in" "${_TGT_EXT}.mlir" _TGT ${_SRC})
            configure_file(
                ${CMAKE_CURRENT_SOURCE_DIR}/${_SRC}
                ${_TGT_PATH}/${_TGT} @ONLY)
            MATH(EXPR genLoopCount "${genLoopCount}+1")
        endwhile()
    endforeach()
endfunction()

add_subdirectory(PlaidML)
# Add other directories in the future

add_lit_testsuite(check-gen "Running the IMEX generated regression tests in ${GEN_TEST_SRC_ROOT}"
    ${GEN_TEST_BIN_ROOT}
    DEPENDS ${IMEX_TEST_DEPENDS}
    )
set_target_properties(check-gen PROPERTIES FOLDER "Tests")

add_lit_testsuites(GEN ${GEN_TEST_SRC_ROOT} DEPENDS ${IMEX_TEST_DEPENDS})
