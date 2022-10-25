# linalg dialect to gpu dialect lowering pipeline
# Ready for vulkan runner or narrow scope l0/sycl runner starting from GPU dialect.
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
          convert-linalg-to-parallel-loops
          gpu-map-parallel-loops
          convert-parallel-loops-to-gpu)
# insert-gpu-allocs pass can have client-api = opencl or vulkan args
func.func(insert-gpu-allocs)
canonicalize
normalize-memrefs
# Unstride memrefs does not seem to be needed.
#func.func(unstride-memrefs)
func.func(lower-affine)
gpu-kernel-outlining
canonicalize
cse
# The following set-spirv-* passes can have client-api = opencl or vulkan args
set-spirv-capabilities
gpu.module(set-spirv-abi-attrs)
canonicalize
# End
