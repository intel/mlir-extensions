add_mlir_library(IMEXTransforms
  insert_gpu_allocs.cpp
  unstride_memref_pass.cpp
  set_spirv_capability.cpp
  spirv_abi_attribute_pass.cpp

  #ADDITIONAL_HEADER_DIRS
  #${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms

  DEPENDS
  IMEXTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  )