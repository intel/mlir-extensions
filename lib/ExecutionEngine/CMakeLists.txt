if(IMEX_ENABLE_L0_RUNTIME)
    add_subdirectory(LEVELZERORUNTIME)
endif()

if(IMEX_ENABLE_SYCL_RUNTIME)
    add_subdirectory(SYCLRUNTIME)
endif()
