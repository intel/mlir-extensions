# linalg dialect to gpu dialect lowering pipeline
# Ready for vulkan runner or narrow scope l0/sycl runner starting from GPU dialect.
convert-tensor-to-linalg
arith-bufferize
builtin.module(func.func(empty-tensor-to-alloc-tensor
          eliminate-alloc-tensors
          scf-bufferize
          shape-bufferize
          linalg-bufferize
          bufferization-bufferize
          tensor-bufferize))
func-bufferize
builtin.module(func.func(finalizing-bufferize
          convert-linalg-to-parallel-loops
          imex-add-outer-parallel-loop
          gpu-map-parallel-loops
          convert-parallel-loops-to-gpu))
# insert-gpu-allocs pass can have client-api = opencl or vulkan args
builtin.module(func.func(insert-gpu-allocs{client-api=opencl}))
canonicalize
normalize-memrefs
# Unstride memrefs does not seem to be needed.
#builtin.module(func.func(unstride-memrefs))
builtin.module(func.func(lower-affine))
gpu-kernel-outlining
canonicalize
cse
# The following set-spirv-* passes can have client-api = opencl or vulkan args
set-spirv-capabilities{client-api=opencl}
gpu.module(set-spirv-abi-attrs{client-api=opencl})
canonicalize
fold-memref-alias-ops
imex-convert-gpu-to-spirv
spirv.module(spirv-lower-abi-attrs
             spirv-update-vce)
builtin.module(func.func(llvm-request-c-wrappers))
serialize-spirv
convert-gpu-to-gpux
convert-func-to-llvm
convert-math-to-llvm
convert-gpux-to-llvm
convert-memref-to-llvm
reconcile-unrealized-casts
# End
