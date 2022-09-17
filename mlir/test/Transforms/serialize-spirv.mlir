// RUN: imex-opt -serialize-spirv %s | FileCheck %s
module attributes {gpu.container_module, spv.target_env = #spv.target_env<#spv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, #spv.resource_limits<>>} {
  // CHECK:         gpu.module @addt_kernel attributes {gpu.binary =
  spv.module @__spv__addt_kernel Physical64 OpenCL requires #spv.vce<v1.0, [Int64, Addresses, Kernel], []> {
    spv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spv.ptr<vector<3xi64>, Input>
    spv.func @addt_kernel(%arg0: !spv.ptr<f32, CrossWorkgroup>, %arg1: !spv.ptr<f32, CrossWorkgroup>, %arg2: !spv.ptr<f32, CrossWorkgroup>) "None" attributes {spv.entry_point_abi = #spv.entry_point_abi<>, workgroup_attributions = 0 : i64} {
      %cst5_i64 = spv.Constant 5 : i64
      %__builtin_var_WorkgroupId___addr = spv.mlir.addressof @__builtin_var_WorkgroupId__ : !spv.ptr<vector<3xi64>, Input>
      %0 = spv.Load "Input" %__builtin_var_WorkgroupId___addr : vector<3xi64>
      %1 = spv.CompositeExtract %0[0 : i32] : vector<3xi64>
      %__builtin_var_WorkgroupId___addr_0 = spv.mlir.addressof @__builtin_var_WorkgroupId__ : !spv.ptr<vector<3xi64>, Input>
      %2 = spv.Load "Input" %__builtin_var_WorkgroupId___addr_0 : vector<3xi64>
      %3 = spv.CompositeExtract %2[1 : i32] : vector<3xi64>
      spv.Branch ^bb1
    ^bb1:  // pred: ^bb0
      %4 = spv.IMul %1, %cst5_i64 : i64
      %5 = spv.IAdd %4, %3 : i64
      %6 = spv.InBoundsPtrAccessChain %arg0[%5] : !spv.ptr<f32, CrossWorkgroup>, i64
      %7 = spv.Load "CrossWorkgroup" %6 ["Aligned", 4] : f32
      %8 = spv.IMul %1, %cst5_i64 : i64
      %9 = spv.IAdd %8, %3 : i64
      %10 = spv.InBoundsPtrAccessChain %arg1[%9] : !spv.ptr<f32, CrossWorkgroup>, i64
      %11 = spv.Load "CrossWorkgroup" %10 ["Aligned", 4] : f32
      %12 = spv.FAdd %7, %11 : f32
      %13 = spv.IMul %1, %cst5_i64 : i64
      %14 = spv.IAdd %13, %3 : i64
      %15 = spv.InBoundsPtrAccessChain %arg2[%14] : !spv.ptr<f32, CrossWorkgroup>, i64
      spv.Store "CrossWorkgroup" %15, %12 ["Aligned", 4] : f32
      spv.Return
    }
    spv.EntryPoint "Kernel" @addt_kernel, @__builtin_var_WorkgroupId__
  }
  gpu.module @addt_kernel {
    gpu.func @addt_kernel(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) kernel attributes {spv.entry_point_abi = #spv.entry_point_abi<>} {
      %c5 = arith.constant 5 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %2 = arith.muli %0, %c5 : index
      %3 = arith.addi %2, %1 : index
      %4 = memref.load %arg0[%3] : memref<?xf32>
      %5 = arith.muli %0, %c5 : index
      %6 = arith.addi %5, %1 : index
      %7 = memref.load %arg1[%6] : memref<?xf32>
      %8 = arith.addf %4, %7 : f32
      %9 = arith.muli %0, %c5 : index
      %10 = arith.addi %9, %1 : index
      memref.store %8, %arg2[%10] : memref<?xf32>
      gpu.return
    }
  }
}
