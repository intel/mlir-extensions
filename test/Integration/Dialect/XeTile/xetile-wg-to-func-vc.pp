// gpu dialect with workgroup level XeTile dialect to
// llvm dialect (for host code) and
// spirv dialect (for device code) lowering pipeline.
// Ready for imex runner starting from GPU dialect.

builtin.module(
    cse
    gpu.module(xetile-wg-to-sg,
        cse,
        xetile-init-duplicate,
        xetile-canonicalization,
        xetile-blockop-fallback,
        xetile-blocking,
	cse,
        convert-xetile-to-xegpu,
	cse,
        imex-xegpu-hoist-transpose,
        imex-xegpu-apply-vnni-transformation,
        imex-xegpu-optimize-transpose)
    cse
    gpu.module(convert-math-to-vc{enable-high-precision-interim-calculation=true},
        convert-xegpu-to-vc)
    cse
    xegpu-vector-linearize
    canonicalize
    cse
    reconcile-unrealized-casts
    gpu.module(math-extend-to-supported-types{target-type=f32})
    gpu.module(arith-emulate-unsupported-floats{source-types=bf16 target-type=f32})
    spirv-attach-target{ver=v1.0 caps=Addresses,BFloat16TypeKHR,Float16Buffer,Int64,Int16,Int8,Kernel,Linkage,Vector16,GenericPointer,Groups,Float16,Float64,AtomicFloat32AddEXT,ExpectAssumeKHR,SubgroupDispatch,VectorComputeINTEL,VectorAnyINTEL,Bfloat16ConversionINTEL exts=SPV_EXT_shader_atomic_float_add,SPV_KHR_bfloat16,SPV_KHR_expect_assume,SPV_INTEL_vector_compute,SPV_INTEL_bfloat16_conversion}
    imex-convert-to-spirv{use-64bit-index=true}
    gpu.module(spirv.module(spirv-lower-abi-attrs, spirv-update-vce))
    func.func(llvm-request-c-wrappers)
    convert-vector-to-scf
    convert-scf-to-cf
    func.func(gpu-async-region)
    expand-strided-metadata
    gpu-to-llvm{use-bare-pointers-for-kernels=true}
    finalize-memref-to-llvm
    convert-to-llvm
    gpu-module-to-binary
    lower-affine
    reconcile-unrealized-casts)
// End
