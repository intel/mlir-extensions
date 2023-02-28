file(STRINGS kInputFusion.dtypes.in test_dtypes)
list(APPEND test_shapes "512x1024")

foreach(shape ${test_shapes})
    foreach(dtype ${test_dtypes})
        configure_file(kInputFusion.mlir.in ${IMEX_BINARY_DIR}/benchmarks/kInputFusion/kInputFusion_${shape}_${dtype}.mlir @ONLY)
    endforeach()
endforeach()
