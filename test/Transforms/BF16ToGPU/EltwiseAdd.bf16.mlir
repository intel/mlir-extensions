// RUN: %python_executable %imex_runner -i %s --pass-pipeline-file=%p/bf16-to-gpu.pp \
// RUN:                                       --no-mlir-runner --filecheck

module @eltwise_add attributes {gpu.container_module} {
  memref.global "private" constant @__constant_10x20xbf16_0 : memref<10x20xbf16> = dense<5.000000e-01>
  memref.global "private" constant @__constant_10x20xbf16 : memref<10x20xbf16> = dense<"0x803F004040408040A040C040E040004110412041304140415041604170418041884190419841A041803F004040408040A040C040E040004110412041304140415041604170418041884190419841A041803F004040408040A040C040E040004110412041304140415041604170418041884190419841A041803F004040408040A040C040E040004110412041304140415041604170418041884190419841A041803F004040408040A040C040E040004110412041304140415041604170418041884190419841A041803F004040408040A040C040E040004110412041304140415041604170418041884190419841A041803F004040408040A040C040E040004110412041304140415041604170418041884190419841A041803F004040408040A040C040E040004110412041304140415041604170418041884190419841A041803F004040408040A040C040E040004110412041304140415041604170418041884190419841A041803F004040408040A040C040E040004110412041304140415041604170418041884190419841A041">
  func.func @test(%arg0: memref<10x20xbf16>, %arg1: memref<10x20xbf16>) -> memref<10x20xbf16> {
    %c20 = arith.constant 20 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    // CHECK: %[[MEMREF:.*]] = gpu.alloc  host_shared () : memref<400xi8>
    // CHECK: %[[VIEW:.*]] = memref.view %[[MEMREF]][%[[CONST0:.*]]][] : memref<400xi8> to memref<10x20xbf16>
    // CHECK: %[[VIEW_0:.*]] = memref.view %[[MEMREF]][%[[CONST0]]][] : memref<400xi8> to memref<10x20xi16>
    // CHECK: memref.copy %arg1, %[[VIEW]] : memref<10x20xbf16> to memref<10x20xbf16>
    %memref = gpu.alloc  host_shared () : memref<10x20xbf16>
    memref.copy %arg1, %memref : memref<10x20xbf16> to memref<10x20xbf16>
    // CHECK: %[[MEMREF_1:.*]] = gpu.alloc  host_shared () : memref<400xi8>
    // CHECK: %[[VIEW_2:.*]] = memref.view %[[MEMREF_1]][%[[CONST0]]][] : memref<400xi8> to memref<10x20xbf16>
    // CHECK: %[[VIEW_3:.*]] = memref.view %[[MEMREF_1]][%[[CONST0]]][] : memref<400xi8> to memref<10x20xi16>
    // CHECK: memref.copy %arg0, %[[VIEW_2]] : memref<10x20xbf16> to memref<10x20xbf16>
    %memref_0 = gpu.alloc  host_shared () : memref<10x20xbf16>
    memref.copy %arg0, %memref_0 : memref<10x20xbf16> to memref<10x20xbf16>
    // CHECK: %[[MEMREF_4:.*]] = gpu.alloc  host_shared () : memref<400xi8>
    // CHECK: %[[VIEW_5:.*]] = memref.view %[[MEMREF_4]][%[[CONST0]]][] : memref<400xi8> to memref<10x20xbf16>
    // CHECK: %[[VIEW_6:.*]] = memref.view %[[MEMREF_4]][%[[CONST0]]][] : memref<400xi8> to memref<10x20xi16>
    %memref_1 = gpu.alloc  host_shared () : memref<10x20xbf16>
    // CHECK: args(%[[VIEW_3]] : memref<10x20xi16>, %[[VIEW_0]] : memref<10x20xi16>, %[[VIEW_6]] : memref<10x20xi16>)
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c10, %c20, %c1) threads in (%c1, %c1, %c1) args(%memref_0 : memref<10x20xbf16>, %memref : memref<10x20xbf16>, %memref_1 : memref<10x20xbf16>)
    // CHECK: gpu.dealloc  %[[MEMREF_1]] : memref<400xi8>
    // CHECK: gpu.dealloc  %[[MEMREF]] : memref<400xi8>
    // CHECK: return %[[VIEW_5]] : memref<10x20xbf16>
    gpu.dealloc  %memref_0 : memref<10x20xbf16>
    gpu.dealloc  %memref : memref<10x20xbf16>
    return %memref_1 : memref<10x20xbf16>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    // CHECK: gpu.func @test_kernel(%arg0: memref<10x20xi16>, %arg1: memref<10x20xi16>, %arg2: memref<10x20xi16>)
    gpu.func @test_kernel(%arg0: memref<10x20xbf16>, %arg1: memref<10x20xbf16>, %arg2: memref<10x20xbf16>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 10, 20, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      // CHECK: %[[VAR2:.*]] = memref.load %arg0[%[[VAR0:.*]], %[[VAR1:.*]]] : memref<10x20xi16>
      %2 = memref.load %arg0[%0, %1] : memref<10x20xbf16>
      // CHECK: %[[VAR3:.*]] = memref.load %arg1[%[[VAR0]], %[[VAR1]]] : memref<10x20xi16>
      %3 = memref.load %arg1[%0, %1] : memref<10x20xbf16>
      // CHECK: %[[VAR4:.*]] = arith.bitcast %[[VAR2]] : i16 to bf16
      // CHECK: %[[VAR5:.*]] = arith.extf %[[VAR4]] : bf16 to f32
      // CHECK: %[[VAR6:.*]] = arith.bitcast %[[VAR3]] : i16 to bf16
      // CHECK: %[[VAR7:.*]] = arith.extf %[[VAR6]] : bf16 to f32
      // CHECK: %[[VAR8:.*]] = arith.addf %[[VAR5]], %[[VAR7]] : f32
      // CHECK: %[[VAR9:.*]] = arith.truncf %[[VAR8]] : f32 to bf16
      // CHECK: %[[VAR10:.*]] = arith.bitcast %[[VAR9]] : bf16 to i16
      %4 = arith.addf %2, %3 : bf16
      // CHECK: memref.store %[[VAR10]], %arg2[%[[VAR0]], %[[VAR1]]] : memref<10x20xi16>
      memref.store %4, %arg2[%0, %1] : memref<10x20xbf16>
      gpu.return
    }
  }
  func.func @main() {
    %0 = memref.get_global @__constant_10x20xbf16 : memref<10x20xbf16>
    %1 = memref.get_global @__constant_10x20xbf16_0 : memref<10x20xbf16>
    %2 = call @test(%0, %1) : (memref<10x20xbf16>, memref<10x20xbf16>) -> memref<10x20xbf16>
    %cast = memref.cast %2 : memref<10x20xbf16> to memref<*xbf16>
    call @printMemrefBF16(%cast) : (memref<*xbf16>) -> ()
    return
  }
  func.func private @printMemrefBF16(memref<*xbf16>)
}
