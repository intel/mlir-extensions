// linalg dialect to gpu dialect lowering pipeline
// Ready for vulkan runner or narrow scope l0/sycl runner starting from GPU dialect.
builtin.module(
    imex-convert-gpu-to-spirv{enable-joint-matrix=true}
    canonicalize
    spirv.module(spirv-lower-abi-attrs
             spirv-update-vce)
    func.func(llvm-request-c-wrappers)
    serialize-spirv
    convert-vector-to-scf
    convert-gpu-to-gpux
    convert-scf-to-cf
    convert-cf-to-llvm
    convert-vector-to-llvm
    convert-index-to-llvm
    convert-arith-to-llvm
    convert-func-to-llvm
    convert-math-to-llvm
    convert-gpux-to-llvm
    convert-index-to-llvm
    expand-strided-metadata
    lower-affine
    finalize-memref-to-llvm
    reconcile-unrealized-casts)
// End
