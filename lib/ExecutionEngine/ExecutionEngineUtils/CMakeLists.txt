add_mlir_library(imex_execution_engine_utils
  ExecutionEngineUtils.cpp
  EXCLUDE_FROM_LIBMLIR

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/imex/ExecutionEngine/
)
