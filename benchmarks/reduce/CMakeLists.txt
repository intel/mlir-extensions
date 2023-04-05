file(STRINGS reduce.dtypes.in test_dtypes)
list(APPEND test_shapes "1x16x512x512")

foreach(shape ${test_shapes})
    foreach(dtype ${test_dtypes})
        configure_file(reduce_gpu.mlir.in ${IMEX_BINARY_DIR}/benchmarks/reduce/gpu/reduce_${shape}_${dtype}.mlir @ONLY)
        configure_file(reduce_cpu.mlir.in ${IMEX_BINARY_DIR}/benchmarks/reduce/cpu/reduce_${shape}_${dtype}.mlir @ONLY)
    endforeach()
endforeach()
