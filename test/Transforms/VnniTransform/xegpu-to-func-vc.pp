// gpu dialect with intel intrinsic functions (func dialect) to
// llvm dialect (for host code) and
// spirv dialect (for device code) lowering pipeline.
// Ready for imex runner starting from GPU dialect.
builtin.module(
    imex-xegpu-apply-vnni-transformation
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
