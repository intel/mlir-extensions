
file(STRINGS silu.shapes.in test_shapes)
file(STRINGS silu.dtypes.in test_dtypes)

foreach(shape ${test_shapes})
    unset(mathes)
    unset(iterators)
    unset(maps)
    unset(id)

    set(id, 0)
    string(REGEX MATCHALL x dims "${shape}" )
    list(APPEND iterators "\"parallel\"")
    list(APPEND maps "d0")

    # prepare the map and iterator
    foreach(i ${dims})
        math(EXPR id "${id}+1")
        list(APPEND maps "d${id}")
        list(APPEND iterators "\"parallel\"")
    endforeach()
    list(JOIN maps ", " affine_map)
    list(JOIN iterators ", " iterator_types)

    foreach(dtype ${test_dtypes})
        configure_file(silu.mlir.in ${IMEX_BINARY_DIR}/benchmarks/silu/silu_${shape}_${dtype}.mlir @ONLY)
    endforeach()
endforeach()
