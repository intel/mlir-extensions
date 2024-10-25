// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/gpu-to-llvm.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
module attributes {
  gpu.container_module
}{

  func.func @main() {
     %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c100 = arith.constant 100: i32
    %cst_f32 = arith.constant 314.4: f32

    gpu.launch_func @kernel_module::@print_kernel
        blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
        args(%c100: i32, %cst_f32: f32)
    // CHECK: Hello
    // CHECK: Hello, world : 100 314.399994
    // CHECK: Thread id: 0
    return
  }

  gpu.module @kernel_module
  attributes {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, #spirv.resource_limits<>>
} {
      gpu.func @print_kernel(%arg0: i32, %arg1: f32) kernel
      attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
        %0 = gpu.block_id x
        %1 = gpu.block_id y
        %2 = gpu.thread_id x
        gpu.printf "\nHello\n"
        gpu.printf "\nHello, world : %d %f\n" %arg0, %arg1: i32, f32
        gpu.printf "\nThread id: %d\n" %2: index
        gpu.return
   }
  }
}
