// linalg dialect to gpu dialect lowering pipeline
// Ready for vulkan runner or narrow scope l0/sycl runner starting from GPU dialect.
builtin.module(convert-tensor-to-linalg
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
          convert-linalg-to-parallel-loops
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
    convert-func-to-llvm
    convert-gpux-to-llvm
    finalize-memref-to-llvm
    reconcile-unrealized-casts)
// End
