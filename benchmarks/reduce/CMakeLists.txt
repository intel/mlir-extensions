file(STRINGS reduce.dtypes.in test_dtypes)
list(APPEND test_shapes "1x16x512x512")

foreach(shape ${test_shapes})
    foreach(dtype ${test_dtypes})
        configure_file(reduce.mlir.in ${IMEX_BINARY_DIR}/benchmarks/reduce/reduce_${shape}_${dtype}.mlir @ONLY)
    endforeach()
endforeach()
