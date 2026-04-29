// RUN: imex-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup zebin-chip=pvc" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_levelzero_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --shared-libs=%irunner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @reduction {
    gpu.func @cross_sg_intra_lane_2D_scalar(%arg0: memref<4x1x1024xf32>, %arg1: memref<4x1x1xf32>) kernel attributes {known_block_size = array<i32: 1024, 1, 1>} {
      %true = arith.constant true
      %cst = arith.constant 1.024000e+03 : f32
      %cst_0 = arith.constant 0.000000e+00 : f32
      %cst_1 = arith.constant dense<true> : vector<32x32xi1>
      %cst_2 = arith.constant  dense<32> : vector<32xindex>
      %c1024 = arith.constant 1024 : index
      %block_id_y = gpu.block_id y
      // TODO: Remove explicit layout once #1641 is resolved.
      %0 = vector.step {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [32, 1], sg_data = [1, 1]>, dims = [1]>} : vector<32xindex>
      %1 = arith.muli %0, %cst_2 : vector<32xindex>
      %2 = vector.step : vector<32xindex>
      %3 = vector.shape_cast %1 : vector<32xindex> to vector<32x1xindex>
      %4 = vector.broadcast %3 : vector<32x1xindex> to vector<32x32xindex>
      %5 = vector.broadcast %2 : vector<32xindex> to vector<32x32xindex>
      %6 = arith.addi %4, %5 : vector<32x32xindex>
      %7 = arith.muli %block_id_y, %c1024 : index
      %8 = vector.broadcast %7 : index to vector<32x32xindex>
      %9 = arith.addi %8, %6 : vector<32x32xindex>
      %intptr = memref.extract_aligned_pointer_as_index %arg0 : memref<4x1x1024xf32> -> index
      %10 = arith.index_cast %intptr : index to i64
      %11 = xegpu.load %10[%9], %cst_1 <{layout = #xegpu.layout<sg_layout = [32, 1], sg_data = [1, 32]>}> : i64, vector<32x32xindex>, vector<32x32xi1> -> vector<32x32xf32>
      %12 = vector.multi_reduction <add>, %11, %cst_0 [0, 1] : vector<32x32xf32> to f32
      %13 = arith.divf %12, %cst : f32
      %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [4], strides: [1] : memref<4x1x1xf32> to memref<4xf32>
      scf.if %true {
        memref.store %13, %reinterpret_cast[%block_id_y] : memref<4xf32>
      }
      gpu.return
    }
  }

  func.func @test(%src : memref<4x1x1024xf32>, %dst : memref<4x1x1xf32>) attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c1024 = arith.constant 1024 : index

    %stream0_0 = gpu.wait async

    %gpu_memref_src, %stream0_1 = gpu.alloc async [%stream0_0] () : memref<4x1x1024xf32>
    %stream0_2 = gpu.memcpy async [%stream0_1] %gpu_memref_src, %src  : memref<4x1x1024xf32>, memref<4x1x1024xf32>

    %gpu_memref_dst, %stream0_3 = gpu.alloc async [%stream0_2] () : memref<4x1x1xf32>
    %stream0_4 = gpu.memcpy async [%stream0_3] %gpu_memref_dst, %dst  : memref<4x1x1xf32>, memref<4x1x1xf32>

    %stream0_5 = gpu.launch_func async[%stream0_4] @reduction::@cross_sg_intra_lane_2D_scalar blocks in (%c1, %c4, %c1) threads in (%c1024, %c1, %c1) args(%gpu_memref_src : memref<4x1x1024xf32>, %gpu_memref_dst : memref<4x1x1xf32>)

    %stream0_6 = gpu.memcpy async [%stream0_5]  %dst, %gpu_memref_dst : memref<4x1x1xf32>, memref<4x1x1xf32>
    %stream0_7 = gpu.dealloc async [%stream0_6] %gpu_memref_src : memref<4x1x1024xf32>
    %stream0_8 = gpu.dealloc async [%stream0_7] %gpu_memref_dst : memref<4x1x1xf32>
    gpu.wait [%stream0_8]
    return
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %src = memref.alloc() : memref<4x1x1024xf32>
    %dst = memref.alloc() : memref<4x1x1xf32>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c1024 = arith.constant 1024 : index

    %c0_f32 = arith.constant 0. : f32
    %c1_f32 = arith.constant 1. : f32

    // Initialize source with value 1.0 for all elements
    scf.for %i = %c0 to %c4 step %c1 {
      scf.for %j = %c0 to %c1 step %c1 {
        scf.for %k = %c0 to %c1024 step %c1 {
          memref.store %c1_f32, %src[%i, %j, %k] : memref<4x1x1024xf32>
        }
      }
    }

    // Initialize destination with 0
    scf.for %i = %c0 to %c4 step %c1 {
      scf.for %j = %c0 to %c1 step %c1 {
        memref.store %c0_f32, %dst[%i, %j, %c0] : memref<4x1x1xf32>
      }
    }

    call @test(%src, %dst) : (memref<4x1x1024xf32>, memref<4x1x1xf32>) -> ()

    %dst_cast = memref.cast %dst : memref<4x1x1xf32> to memref<*xf32>

    // CHECK: Unranked Memref base@ = 0x{{[0-9a-f]+}} rank = 3 offset = 0 sizes = [4, 1, 1] strides = [1, 1, 1] data =
    // CHECK-NEXT: {{\[}}{{\[}}{{\[}}1]],
    // CHECK-NEXT: {{\[}}{{\[}}1]],
    // CHECK-NEXT: {{\[}}{{\[}}1]],
    // CHECK-NEXT: {{\[}}{{\[}}1]]]
    call @printMemrefF32(%dst_cast) : (memref<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF32(%ptr : memref<*xf32>) attributes { llvm.emit_c_interface }
}