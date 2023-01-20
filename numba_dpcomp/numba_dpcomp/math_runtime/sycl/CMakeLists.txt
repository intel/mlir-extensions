# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.20)

project(dpcomp-math-sycl-runtime LANGUAGES CXX C)

include(GenerateExportHeader)

set(SOURCES_LIST
    lib/NumpyLinalg.cpp
    )
set(HEADERS_LIST
    ../include/Common.hpp
    )

add_library(${PROJECT_NAME} SHARED ${SOURCES_LIST} ${HEADERS_LIST})
generate_export_header(${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_BINARY_DIR} "../include")

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

option(IMEX_USE_MKL
    "Enable mkl support"
    ON
)

option(IMEX_USE_SYCL
    "Enable sycl support"
    OFF
)

message(STATUS "IMEX_USE_SYCL ${IMEX_USE_SYCL}")
message(STATUS "IMEX_USE_MKL ${IMEX_USE_MKL}")

if(IMEX_USE_SYCL AND IMEX_USE_MKL)
    find_package(IntelDPCPP REQUIRED)
    if (NOT DEFINED(MKL_THREADING))
        set(MKL_THREADING tbb_thread)
    endif()

    find_package(MKL CONFIG REQUIRED)

    target_compile_options(${PROJECT_NAME} PUBLIC
        $<TARGET_PROPERTY:MKL::MKL_DPCPP,INTERFACE_COMPILE_OPTIONS>)

    target_include_directories(${PROJECT_NAME} PRIVATE
        $<TARGET_PROPERTY:MKL::MKL_DPCPP,INTERFACE_INCLUDE_DIRECTORIES>)

    target_link_libraries(${PROJECT_NAME} PUBLIC $<LINK_ONLY:MKL::MKL_DPCPP>)
    target_compile_definitions(${PROJECT_NAME} PRIVATE IMEX_USE_SYCL_MKL=1)
endif()

install(TARGETS ${PROJECT_NAME}
        DESTINATION "${MATH_SYCL_RUNTIME_INSTALL_PATH}"
        )
