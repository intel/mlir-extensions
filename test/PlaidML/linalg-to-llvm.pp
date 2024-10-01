// linalg dialect to gpu dialect lowering pipeline
// Ready for vulkan runner or narrow scope l0/sycl runner starting from GPU dialect.
builtin.module(convert-tensor-to-linalg
    func.func(empty-tensor-to-alloc-tensor)
          //eliminate-empty-tensors
    one-shot-bufferize{unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries}
    func.func(convert-linalg-to-parallel-loops
          imex-add-outer-parallel-loop
          gpu-map-parallel-loops
          convert-parallel-loops-to-gpu)
// insert-gpu-allocs pass can have client-api = opencl or vulkan args
    func.func(insert-gpu-allocs{client-api=opencl})
    canonicalize
    normalize-memrefs
// Unstride memrefs does not seem to be needed.
//  func.func(unstride-memrefs)
    func.func(lower-affine)
    gpu-kernel-outlining
    canonicalize
    cse
// The following set-spirv-* passes can have client-api = opencl or vulkan args
    set-spirv-capabilities{client-api=opencl}
    gpu.module(set-spirv-abi-attrs{client-api=opencl})
    canonicalize
    fold-memref-alias-ops
    imex-convert-gpu-to-spirv
    spirv.module(spirv-lower-abi-attrs
             spirv-update-vce)
    func.func(llvm-request-c-wrappers)
    serialize-spirv
    convert-gpu-to-gpux
    expand-strided-metadata
    finalize-memref-to-llvm
    convert-func-to-llvm
    convert-math-to-llvm
    convert-gpux-to-llvm
    lower-affine
    reconcile-unrealized-casts)
// End
