// gpu dialect with intel intrinsic functions (func dialect) to
// llvm dialect (for host code) and
// spirv dialect (for device code) lowering pipeline.
// Ready for imex runner starting from GPU dialect.
builtin.module(
    gpu.module(imex-xegpu-hoist-transpose,
        imex-xegpu-apply-vnni-transformation,
        imex-xegpu-optimize-transpose)
    cse
    gpu.module(convert-math-to-vc{enable-high-precision-interim-calculation=true}
        convert-xegpu-to-vc)
    cse
    canonicalize
    xegpu-vector-linearize
    canonicalize
    cse
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
