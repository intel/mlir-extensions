add_subdirectory(ExecutionEngineUtils)

if(IMEX_ENABLE_L0_RUNTIME)
  add_subdirectory(LEVELZERORUNTIME)
endif()

if(IMEX_ENABLE_SYCL_RUNTIME)
  add_subdirectory(SYCLRUNTIME)
endif()

add_mlir_library(imex_runner_utils
  SHARED
  ImexRunnerUtils.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  mlir_float16_utils
)
target_compile_definitions(imex_runner_utils PRIVATE imex_runner_utils_EXPORTS)
