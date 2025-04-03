// RUN: imex-opt -convert-xevm-to-llvm -split-input-file %s | FileCheck %s

gpu.module @dpas_funcs {
llvm.func @xevm.dpas.i8(%c : vector<8xi32>, %a : vector<8xi16>, %b : vector<8xi32>) {
  // CHECK: llvm.func spir_funccc @_Z36intel_sub_group_i8_i8_matrix_mad_k32Dv8_sDv8_iS0_(vector<8xi16>, vector<8xi32>, vector<8xi32>) -> vector<8xi32> attributes {convergent, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return}
  // CHECK:     llvm.func @xevm.dpas.i8(%arg0: vector<8xi32>, %arg1: vector<8xi16>, %arg2: vector<8xi32>) {
  // CHECK-NEXT: llvm.call spir_funccc @_Z36intel_sub_group_i8_i8_matrix_mad_k32Dv8_sDv8_iS0_(%arg1, %arg2, %arg0) {{.*}} : (vector<8xi16>, vector<8xi32>, vector<8xi32>) -> vector<8xi32>
  %0 = xevm.dpas %c, %a, %b {pa = i8, pb = i8, rc = 8} : (vector<8xi32>, vector<8xi16>, vector<8xi32>) -> vector<8xi32>
  llvm.return
}
}

// -----

gpu.module @dpas_funcs {
  // CHECK: llvm.func spir_funccc @_Z36intel_sub_group_u8_u8_matrix_mad_k32Dv8_sDv8_iS0_(vector<8xi16>, vector<8xi32>, vector<8xi32>) -> vector<8xi32> attributes {convergent, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return}
llvm.func @xevm.dpas.u8(%c : vector<8xi32>, %a : vector<8xi16>, %b : vector<8xi32>) {
  // CHECK:     llvm.func @xevm.dpas.u8(%arg0: vector<8xi32>, %arg1: vector<8xi16>, %arg2: vector<8xi32>) {
  // CHECK-NEXT: llvm.call spir_funccc @_Z36intel_sub_group_u8_u8_matrix_mad_k32Dv8_sDv8_iS0_(%arg1, %arg2, %arg0) {{.*}} : (vector<8xi16>, vector<8xi32>, vector<8xi32>) -> vector<8xi32>
  %0 = xevm.dpas %c, %a, %b {pa = u8, pb = u8, rc = 8} : (vector<8xi32>, vector<8xi16>, vector<8xi32>) -> vector<8xi32>
  llvm.return
}
}
// -----

gpu.module @dpas_funcs {
  // CHECK: llvm.func spir_funccc @_Z40intel_sub_group_bf16_bf16_matrix_mad_k16Dv8_sDv8_iDv8_f(vector<8xi16>, vector<8xi32>, vector<8xf32>) -> vector<8xf32> attributes {convergent, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return}
llvm.func @xevm.dpas.bf16(%c : vector<8xf32>, %a : vector<8xi16>, %b : vector<8xi32>) {
  // CHECK:     llvm.func @xevm.dpas.bf16(%arg0: vector<8xf32>, %arg1: vector<8xi16>, %arg2: vector<8xi32>) {
  // CHECK-NEXT: llvm.call spir_funccc @_Z40intel_sub_group_bf16_bf16_matrix_mad_k16Dv8_sDv8_iDv8_f(%arg1, %arg2, %arg0) {{.*}} : (vector<8xi16>, vector<8xi32>, vector<8xf32>) -> vector<8xf32>
  %0 = xevm.dpas %c, %a, %b {pa = bf16, pb = bf16, rc = 8} : (vector<8xf32>, vector<8xi16>, vector<8xi32>) -> vector<8xf32>
  llvm.return
}
}

// -----

gpu.module @dpas_funcs {
  // CHECK: llvm.func spir_funccc @_Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_f(vector<8xi16>, vector<8xi32>, vector<8xf32>) -> vector<8xf32> attributes {convergent, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return}
llvm.func @xevm.dpas.f16(%c : vector<8xf32>, %a : vector<8xi16>, %b : vector<8xi32>) {
  // CHECK:     llvm.func @xevm.dpas.f16(%arg0: vector<8xf32>, %arg1: vector<8xi16>, %arg2: vector<8xi32>) {
  // CHECK-NEXT: llvm.call spir_funccc @_Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_f(%arg1, %arg2, %arg0) {{.*}} : (vector<8xi16>, vector<8xi32>, vector<8xf32>) -> vector<8xf32>
  %0 = xevm.dpas %c, %a, %b {pa = f16, pb = f16, rc = 8} : (vector<8xf32>, vector<8xi16>, vector<8xi32>) -> vector<8xf32>
  llvm.return
}
}

// -----

gpu.module @dpas_funcs {
  // CHECK: llvm.func spir_funccc @_Z39intel_sub_group_tf32_tf32_matrix_mad_k8Dv4_fDv8_fS0_(vector<4xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32> attributes {convergent, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return}
llvm.func @xevm.dpas.f32(%c : vector<8xf32>, %a : vector<4xf32>, %b : vector<8xf32>) {
  // CHECK:     llvm.func @xevm.dpas.f32(%arg0: vector<8xf32>, %arg1: vector<4xf32>, %arg2: vector<8xf32>) {
  // CHECK-NEXT: llvm.call spir_funccc @_Z39intel_sub_group_tf32_tf32_matrix_mad_k8Dv4_fDv8_fS0_
  // CHECK-SAME:    (%arg1, %arg2, %arg0)
  // CHECK-SAME:    convergent
  // CHECK-SAME:    : (vector<4xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  %0 = xevm.dpas %c, %a, %b {pa = tf32, pb = tf32, rc = 8} : (vector<8xf32>, vector<4xf32>, vector<8xf32>) -> vector<8xf32>
  llvm.return
}
}

// -----

gpu.module @dpas_funcs {
  // CHECK: llvm.func spir_funccc @_Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_Dh(vector<8xi16>, vector<8xi32>, vector<8xf16>) -> vector<8xf16> attributes {convergent, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return}
llvm.func @xevm.dpas.f16_accum(%c: vector<8xf16>, %a : vector<8xi16>, %b : vector<8xi32>) {
  // CHECK:     llvm.func @xevm.dpas.f16_accum(%arg0: vector<8xf16>, %arg1: vector<8xi16>, %arg2: vector<8xi32>) {
  // CHECK-NEXT: lvm.call spir_funccc @_Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_Dh
  // CHECK-SAME:    (%arg1, %arg2, %arg0)
  // CHECK-SAME:    convergent
  // CHECK-SAME:    : (vector<8xi16>, vector<8xi32>, vector<8xf16>) -> vector<8xf16>
  %0 = xevm.dpas %c, %a, %b {pa = f16, pb = f16, rc = 8} : (vector<8xf16>, vector<8xi16>, vector<8xi32>) -> vector<8xf16>
  llvm.return
}
}
// -----

gpu.module @dpas_funcs {
  // CHECK: llvm.func spir_funccc @_Z40intel_sub_group_bf16_bf16_matrix_mad_k16Dv8_sDv8_iS_(vector<8xi16>, vector<8xi32>, vector<8xi16>) -> vector<8xi16> attributes {convergent, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return}
llvm.func @xevm.dpas.bf16_accum(%c: vector<8xbf16>, %a : vector<8xi16>, %b : vector<8xi32>) {
  // CHECK:     llvm.func @xevm.dpas.bf16_accum(%arg0: vector<8xbf16>, %arg1: vector<8xi16>, %arg2: vector<8xi32>) {
  // CHECK-NEXT: [[CAST:%.*]] = llvm.bitcast %arg0 : vector<8xbf16> to vector<8xi16>
  // CHECK-NEXT: [[RES:%.*]] = llvm.call spir_funccc @_Z40intel_sub_group_bf16_bf16_matrix_mad_k16Dv8_sDv8_iS_
  // CHECK-SAME:    (%arg1, %arg2, [[CAST]])
  // CHECK-SAME:    convergent
  // CHECK-SAME:    : (vector<8xi16>, vector<8xi32>, vector<8xi16>) -> vector<8xi16>
  %0 = xevm.dpas %c, %a, %b {pa = bf16, pb = bf16, rc = 8} : (vector<8xbf16>, vector<8xi16>, vector<8xi32>) -> vector<8xbf16>
  // CHECK-NEXT: {{%.*}} = llvm.bitcast [[RES]] : vector<8xi16> to vector<8xbf16>
  llvm.return
}
}
