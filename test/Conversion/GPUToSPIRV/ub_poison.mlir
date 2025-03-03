// RUN: imex-opt --split-input-file -imex-convert-gpu-to-spirv -verify-diagnostics %s -o - | FileCheck %s

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0,
      [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR],
      [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, #spirv.resource_limits<>>
} {
  gpu.module @kernels {
    gpu.func @ub_poison(%arg3: index) kernel attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = ub.poison : index
      %1 = ub.poison : i16
      %2 = ub.poison : f64
      %3 = ub.poison : vector<4xf32>
      gpu.return
    }
  }
}

// CHECK-LABEL: spirv.func @ub_poison
// CHECK: {{.*}} = spirv.Undef : i64
// CHECK: {{.*}} = spirv.Undef : i16
// CHECK: {{.*}} = spirv.Undef : f64
// CHECK: {{.*}} = spirv.Undef : vector<4xf32>
// CHECK: spirv.Return
