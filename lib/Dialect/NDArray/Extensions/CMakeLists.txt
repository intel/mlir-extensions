set(LLVM_OPTIONAL_SOURCES
  AllExtensions.cpp
  BufferizableOpInterfaceImpl.cpp
  MeshShardingExtensions.cpp
  )

add_imex_extension_library(IMEXNDArrayMeshShardingExtensions
  MeshShardingExtensions.cpp
  BufferizableOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mlir/Dialect/NDArray/Extensions

  LINK_LIBS PUBLIC
  IMEXNDArrayDialect
  MLIRIR
  MLIRShardingInterface
  )

add_imex_extension_library(IMEXNDArrayAllExtensions
  AllExtensions.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mlir/Dialect/NDArray/Extensions

  LINK_LIBS PUBLIC
  IMEXNDArrayMeshShardingExtensions
  )
