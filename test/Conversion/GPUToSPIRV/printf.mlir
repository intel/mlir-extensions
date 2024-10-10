// RUN: imex-opt -allow-unregistered-dialect -split-input-file -imex-convert-gpu-to-spirv -verify-diagnostics %s -o - | FileCheck %s

module @test attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, #spirv.resource_limits<>>
} {
  func.func @print_test() {
    %c1 = arith.constant 1 : index
    %c100 = arith.constant 100: i32
    %cst_f32 = arith.constant 314.4: f32

    gpu.launch_func @kernel_module1::@test_printf_arg
        blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
        args(%c100: i32, %cst_f32: f32)
    return
  }

  // CHECK-LABEL: spirv.module @{{.*}} Physical64 OpenCL
  // CHECK-DAG: spirv.SpecConstant [[SPECCST:@.*]] = {{.*}} : i8
  // CHECK-DAG: spirv.SpecConstantComposite [[SPECCSTCOMPOSITE:@.*]] ([[SPECCST]], {{.*}}) : !spirv.array<[[ARRAYSIZE:.*]] x i8>
  // CHECK-DAG: spirv.GlobalVariable [[PRINTMSG:@.*]] initializer([[SPECCSTCOMPOSITE]]) {Constant} : !spirv.ptr<!spirv.array<[[ARRAYSIZE]] x i8>, UniformConstant>
    // spirv.SpecConstantComposite
  gpu.module @kernel_module0 {
      gpu.func @test_printf(%arg0: i32, %arg1: f32) kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
        %0 = gpu.block_id x
        %1 = gpu.block_id y
        %2 = gpu.thread_id x
        // CHECK: [[FMTSTR_ADDR:%.*]] = spirv.mlir.addressof [[PRINTMSG]] : !spirv.ptr<!spirv.array<[[ARRAYSIZE]] x i8>, UniformConstant>
        // CHECK-NEXT: [[FMTSTR_PTR:%.*]] = spirv.Bitcast [[FMTSTR_ADDR]] : !spirv.ptr<!spirv.array<[[ARRAYSIZE]] x i8>, UniformConstant> to !spirv.ptr<i8, UniformConstant>
        // CHECK-NEXT {{%.*}} = spirv.CL.printf [[FMTSTR_PTR]] : !spirv.ptr<i8, UniformConstant> -> i32
        gpu.printf "\nHello\n"
        gpu.return
    }
  }

  // CHECK-LABEL: spirv.module @{{.*}} Physical64 OpenCL
  // CHECK-DAG: spirv.SpecConstant [[SPECCST:@.*]] = {{.*}} : i8
  // CHECK-DAG: spirv.SpecConstantComposite [[SPECCSTCOMPOSITE:@.*]] ([[SPECCST]], {{.*}}) : !spirv.array<[[ARRAYSIZE:.*]] x i8>
  // CHECK-DAG: spirv.GlobalVariable [[PRINTMSG:@.*]] initializer([[SPECCSTCOMPOSITE]]) {Constant} : !spirv.ptr<!spirv.array<[[ARRAYSIZE]] x i8>, UniformConstant>
    // spirv.SpecConstantComposite
  gpu.module @kernel_module1 {
      gpu.func @test_printf_arg(%arg0: i32, %arg1: f32) kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
        %0 = gpu.block_id x
        %1 = gpu.block_id y
        %2 = gpu.thread_id x
        // CHECK: [[FMTSTR_ADDR:%.*]] = spirv.mlir.addressof [[PRINTMSG]] : !spirv.ptr<!spirv.array<[[ARRAYSIZE]] x i8>, UniformConstant>
        // CHECK-NEXT: [[FMTSTR_PTR:%.*]] = spirv.Bitcast [[FMTSTR_ADDR]] : !spirv.ptr<!spirv.array<[[ARRAYSIZE]] x i8>, UniformConstant> to !spirv.ptr<i8, UniformConstant>
        // CHECK-NEXT: {{%.*}} = spirv.CL.printf [[FMTSTR_PTR]] {{%.*}}, {{%.*}}, {{%.*}} : !spirv.ptr<i8, UniformConstant>, i32, f32, i64 -> i32
        gpu.printf "\nHello, world : %d %f \n Thread id: %d\n" %arg0, %arg1, %2: i32, f32, index
          gpu.return
    }
  }
}
