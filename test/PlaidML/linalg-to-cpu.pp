# linalg dialect to gpu dialect lowering pipeline
convert-tensor-to-linalg
arith-bufferize
func.func(empty-tensor-to-alloc-tensor
          eliminate-alloc-tensors
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
convert-memref-to-llvm
convert-func-to-llvm
reconcile-unrealized-casts
reconcile-unrealized-casts
# End
