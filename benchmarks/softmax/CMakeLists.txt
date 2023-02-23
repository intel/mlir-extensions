
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
        configure_file(softmax.mlir.in ${IMEX_BINARY_DIR}/benchmarks/softmax/softmax_${shape}_${dtype}.mlir @ONLY)
    endforeach()
endforeach()
