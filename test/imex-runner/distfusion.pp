builtin.module(
    func.func(add-gpu-regions)
    canonicalize
    func.func(sharding-propagation)
    coalesce-shard-ops
    canonicalize
    func.func(mesh-spmdization)
    canonicalize
    convert-mesh-to-mpi
    canonicalize
    convert-ndarray-to-linalg
    linalg-generalize-named-ops
    linalg-fuse-elementwise-ops
    empty-tensor-to-alloc-tensor
    canonicalize
    one-shot-bufferize
    canonicalize
    imex-remove-temporaries
)
