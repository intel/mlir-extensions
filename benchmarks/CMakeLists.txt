add_subdirectory(relu)
add_subdirectory(silu)
add_subdirectory(softmax)
add_subdirectory(transpose)
add_subdirectory(reduce)
add_subdirectory(kLoopFusion)
add_subdirectory(kInputFusion)

if(WIN32)
    set(MLIR_RUNNER_UTILS_DIR ${LLVM_BINARY_DIR}/bin)
else()
    set(MLIR_RUNNER_UTILS_DIR ${LLVM_LIBRARY_DIR})
endif()

configure_file(bench_imex.in ${IMEX_BINARY_DIR}/benchmarks/bench_imex @ONLY)

file(COPY pipelines/linalg-to-gpu.pp DESTINATION ${IMEX_BINARY_DIR}/benchmarks/pipelines)
file(COPY pipelines/linalg-to-cpu.pp DESTINATION ${IMEX_BINARY_DIR}/benchmarks/pipelines)
