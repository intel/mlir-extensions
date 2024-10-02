// linalg dialect to gpu dialect lowering pipeline
// Ready for vulkan runner or narrow scope l0/sycl runner starting from GPU dialect.
builtin.module(
    serialize-spirv
    convert-gpu-to-gpux
    convert-scf-to-cf
    expand-strided-metadata
    finalize-memref-to-llvm
    convert-cf-to-llvm
    convert-arith-to-llvm
    convert-func-to-llvm
    convert-math-to-llvm
    convert-gpux-to-llvm
    lower-affine
    reconcile-unrealized-casts)
// End
