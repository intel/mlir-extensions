# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

project(dpcomp-math-runtime LANGUAGES CXX C)

include(GenerateExportHeader)

set(SOURCES_LIST
    lib/Common.cpp
    lib/NumpyLinalg.cpp
    )
set(HEADERS_LIST
    lib/Common.hpp
    )

find_package(MKL CONFIG REQUIRED)

add_library(${PROJECT_NAME} SHARED ${SOURCES_LIST} ${HEADERS_LIST})
generate_export_header(${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_BINARY_DIR}
    $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>
    )

target_link_libraries(${PROJECT_NAME} PUBLIC $<LINK_ONLY:MKL::MKL> ${MKL_SYCL})
target_compile_options(${PROJECT_NAME} PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)

if(IMEX_USE_DPNP)
    target_include_directories(${PROJECT_NAME} PRIVATE
        ${DPNP_INCLUDE_DIR}
        )

    target_link_directories(${PROJECT_NAME} PUBLIC ${DPNP_LIBRARY_DIR})
    target_link_libraries(${PROJECT_NAME} PUBLIC "dpnp_backend_c")

    set(DPNP_RPATH "${DPNP_LIBRARY_DIR}" "$ORIGIN/../dpnp")
    set_target_properties(${PROJECT_NAME}
        PROPERTIES
        INSTALL_RPATH "${DPNP_RPATH}"
    )

    target_compile_definitions(${PROJECT_NAME} PRIVATE IMEX_USE_DPNP=1)
endif()
