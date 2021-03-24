cmake_minimum_required(VERSION 3.5)

project(dpcomp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

add_subdirectory(dpcomp)
add_subdirectory(dpcomp_runtime)
add_subdirectory(numba_dpcomp/mlir_compiler)
