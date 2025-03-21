builtin.module(
    convert-ndarray-to-linalg
    canonicalize
    func.func(tosa-make-broadcastable)
    func.func(tosa-to-linalg)
    func.func(tosa-to-tensor)
    canonicalize
    linalg-fuse-elementwise-ops
    arith-expand
    memref-expand
    func.func(empty-tensor-to-alloc-tensor)
    one-shot-bufferize{bufferize-function-boundaries}
    imex-remove-temporaries
    convert-bufferization-to-memref
    func.func(convert-linalg-to-parallel-loops)
    func.func(scf-parallel-loop-fusion)
    drop-regions
    canonicalize
    fold-memref-alias-ops
    expand-strided-metadata
    convert-math-to-funcs
    lower-affine
    convert-scf-to-cf
    finalize-memref-to-llvm
    convert-math-to-llvm
    convert-math-to-libm
    convert-func-to-llvm
    convert-arith-to-llvm
    convert-cf-to-llvm
    reconcile-unrealized-casts
)
