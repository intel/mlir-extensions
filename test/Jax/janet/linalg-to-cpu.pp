// linalg dialect to gpu dialect lowering pipeline
builtin.module(inline
    convert-tensor-to-linalg
    convert-elementwise-to-linalg
    arith-bufferize
    func.func(empty-tensor-to-alloc-tensor
          //eliminate-empty-tensors
          scf-bufferize
          shape-bufferize
          linalg-bufferize
          bufferization-bufferize
          tensor-bufferize)
    func-bufferize
    func.func(finalizing-bufferize
          convert-linalg-to-loops)
    convert-scf-to-cf
    convert-linalg-to-llvm
    convert-cf-to-llvm
    convert-arith-to-llvm
    convert-math-to-llvm
    convert-complex-to-llvm
    convert-index-to-llvm
    expand-strided-metadata
    lower-affine
    finalize-memref-to-llvm
    convert-func-to-llvm
    reconcile-unrealized-casts)
// End
