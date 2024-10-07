// RUN: imex-opt --split-input-file -imex-convert-gpu-to-spirv -verify-diagnostics %s -o - | FileCheck %s

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0,
      [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR],
      [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, #spirv.resource_limits<>>
} {
  gpu.module @kernels {
    gpu.func @create_mask(%arg3: index) kernel attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %697 = vector.create_mask %arg3 : vector<16xi1>
      gpu.return
    }
  }
}

// CHECK-LABEL: spirv.func @create_mask
// CHECK-SAME: %[[MASK_VAL:[[:alnum:]]+]]: i64
// CHECK-NEXT: %[[VECTOR_WIDTH:.*]] = spirv.Constant 16 : i32
// CHECK-NEXT: %[[MASK_VAL_I32:.*]] = spirv.SConvert %[[MASK_VAL]] : i64 to i32
// CHECK-NEXT: %[[CMP1:.*]] = spirv.SLessThan %[[MASK_VAL_I32]], %[[VECTOR_WIDTH]] : i32
// CHECK-NEXT: %[[ONE:.*]] = spirv.Constant 1 : i32
// CHECK-NEXT: %[[SHIFT:.*]] = spirv.ShiftLeftLogical %[[ONE]], %[[MASK_VAL_I32]] : i32, i32
// CHECK-NEXT: %[[MASK:.*]] = spirv.ISub %[[SHIFT]], %[[ONE]] : i32
// CHECK-NEXT: %[[MASK_ONES:.*]] = spirv.Constant -1 : i32
// CHECK-NEXT: %[[SELECT1:.*]] = spirv.Select %[[CMP1]], %[[MASK]], %[[MASK_ONES]] : i1, i32
// CHECK-NEXT: %[[ZERO:.*]] = spirv.Constant 0 : i32
// CHECK-NEXT: %[[CMP2:.*]] = spirv.SLessThan %[[MASK_VAL_I32]], %[[ZERO]] : i32
// CHECK-NEXT: %[[SELECT2:.*]] = spirv.Select %[[CMP2]], %[[ZERO]], %[[SELECT1]] : i1, i32
// CHECK-NEXT: %[[CAST:.*]] = spirv.SConvert %[[SELECT2]] : i32 to i16
// CHECK-NEXT: spirv.Bitcast %[[CAST]] : i16 to vector<16xi1>
// CHECK-NEXT: spirv.Return
