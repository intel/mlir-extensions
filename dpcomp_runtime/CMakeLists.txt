project(dpcomp-runtime LANGUAGES CXX C)

include(GenerateExportHeader)

find_package(TBB REQUIRED)

set(SOURCES_LIST
    src/tbb_parallel.cpp
    )
set(HEADERS_LIST
    )

add_library(${PROJECT_NAME} SHARED ${SOURCES_LIST} ${HEADERS_LIST})
generate_export_header(${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_BINARY_DIR}
    )

target_link_libraries(${PROJECT_NAME} TBB::tbb)

if (CMAKE_SYSTEM_NAME STREQUAL Darwin)
    target_link_libraries(${PROJECT_NAME} "-bundle")
endif()
