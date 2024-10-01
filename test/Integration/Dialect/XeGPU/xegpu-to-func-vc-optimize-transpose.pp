// gpu dialect with intel intrinsic functions (func dialect) to
// llvm dialect (for host code) and
// spirv dialect (for device code) lowering pipeline.
// Ready for imex runner starting from GPU dialect.
builtin.module(
    imex-xegpu-optimize-transpose
    imex-vector-linearize
    gpu.module(convert-xegpu-to-vc)
    reconcile-unrealized-casts
    bf16-to-gpu
    imex-convert-gpu-to-spirv
    spirv.module(spirv-lower-abi-attrs
             spirv-update-vce)
    func.func(llvm-request-c-wrappers)
    serialize-spirv
    convert-vector-to-scf
    convert-gpu-to-gpux
    convert-scf-to-cf
    expand-strided-metadata
    finalize-memref-to-llvm
    convert-cf-to-llvm
    convert-vector-to-llvm
    convert-index-to-llvm
    convert-arith-to-llvm
    convert-func-to-llvm
    convert-math-to-llvm
    convert-gpux-to-llvm
    lower-affine
    reconcile-unrealized-casts)
// End
