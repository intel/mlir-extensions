set(IMEX_ALL_LIBS
  IMEXTransforms
  IMEXNDArrayTransforms
  IMEXNDArrayDialect
  IMEXGPUXDialect
  IMEXDistRuntimeTransforms
  IMEXDistRuntimeDialect
  IMEXRegionTransforms
  IMEXRegionDialect
  IMEXDropRegions
  IMEXGPUToGPUX
  IMEXGPUToSPIRV
  IMEXGPUXToLLVM
  IMEXNDArrayToLinalg
  IMEXSCFToGPU
  MLIRXeGPUToXeVM
  IMEXUtil
  imex_runner_utils
)

# Add VC conversion libraries if enabled
if(IMEX_BUILD_VC_CONVERSIONS)
  list(APPEND IMEX_ALL_LIBS
    IMEXArithToVC
    IMEXMathToVC
    IMEXXeGPUToVC
  )
endif()

if(IMEX_ENABLE_L0_RUNTIME)
  list(APPEND IMEX_ALL_LIBS level-zero-runtime)
endif()

if(IMEX_ENABLE_SYCL_RUNTIME)
  list(APPEND IMEX_ALL_LIBS sycl-runtime)
endif()

add_mlir_public_c_api_library(ImexCAPI
  Dialects.cpp
  Passes.cpp
  LINK_LIBS PUBLIC
  IMEXRegionDialect
  ${IMEX_ALL_LIBS}
)
target_link_libraries(obj.ImexCAPI PUBLIC ImexInterface)
set_property(GLOBAL APPEND PROPERTY IMEX_MLIR_LIBS ImexCAPI)
