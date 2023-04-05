
file(STRINGS softmax.shapes.in test_shapes)
file(STRINGS softmax.dtypes.in test_dtypes)

foreach(shape ${test_shapes})
    unset(sizes)
    string(REPLACE "x" ";" sizes ${shape} )
    list(LENGTH sizes dims)

    if(dims GREATER 2)
        message(NOTICE "Unsupported shape for softmax: ${shape}")
        continue()
    endif()

    list(GET sizes 0 batch_size)

    foreach(dtype ${test_dtypes})
        configure_file(softmax_cpu.mlir.in ${IMEX_BINARY_DIR}/benchmarks/softmax/cpu/softmax_${shape}_${dtype}.mlir @ONLY)
        configure_file(softmax_gpu.mlir.in ${IMEX_BINARY_DIR}/benchmarks/softmax/gpu/softmax_${shape}_${dtype}.mlir @ONLY)
    endforeach()
endforeach()
