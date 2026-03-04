// RUN: imex-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup zebin-chip=pvc" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_levelzero_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --shared-libs=%irunner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

#data_layout = #xegpu.layout<sg_layout = [2, 1], sg_data = [16, 32]>
module attributes {gpu.container_module} {
  gpu.module @reduction {
    gpu.func @cross_sg_intra_lane_1D(%dst: memref<128xf32, 1>, %src: memref<128x32xf32, 1>) kernel {
      %dst1 = memref.memory_space_cast %dst : memref<128xf32, 1> to memref<128xf32>
      %dst_ptr_idx = memref.extract_aligned_pointer_as_index %dst1 : memref<128xf32> -> index
      %dst_ptr_i64 = arith.index_cast %dst_ptr_idx : index to i64
      %src1 = memref.memory_space_cast %src : memref<128x32xf32, 1> to memref<128x32xf32>
      %src_ptr_idx = memref.extract_aligned_pointer_as_index %src1 : memref<128x32xf32> -> index
      %src_ptr_i64 = arith.index_cast %src_ptr_idx : index to i64

      %offset_ldd = vector.step : vector<1024xindex>
      %offset_ld = vector.shape_cast %offset_ldd : vector<1024xindex> to vector<32x32xindex>
      %mask_ld = arith.constant dense<1> : vector<32x32xi1>
      %val = xegpu.load %src_ptr_i64[%offset_ld], %mask_ld : i64, vector<32x32xindex>, vector<32x32xi1> -> vector<32x32xf32>
      %acc = arith.constant dense<0.0> : vector<32xf32>
      %res = vector.multi_reduction <add>, %val, %acc [0] : vector<32x32xf32> to vector<32xf32>

      %offset = vector.step : vector<32xindex>
      %mask = arith.constant dense<1> : vector<32xi1>
      xegpu.store %res, %dst_ptr_i64[%offset], %mask { layout = #xegpu.slice<#data_layout, dims = [0]> } : vector<32xf32>, i64, vector<32xindex>, vector<32xi1>
      gpu.return
    }
  }

func.func @test(%dst : memref<128xf32>, %src : memref<128x32xf32>) attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
        %c16 = arith.constant 16 : index

    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index

    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index

    %stream0_0 = gpu.wait async

    %gpu_memref_dst, %stream0_1 = gpu.alloc async [%stream0_0] () : memref<128xf32>
    %stream0_2 = gpu.memcpy async [%stream0_1] %gpu_memref_dst, %dst  : memref<128xf32>, memref<128xf32>

    %gpu_memref_src, %stream0_3 = gpu.alloc async [%stream0_2] () : memref<128x32xf32>
    %stream0_4 = gpu.memcpy async [%stream0_3] %gpu_memref_src, %src  : memref<128x32xf32>, memref<128x32xf32>


    %dst_ptr_idx = memref.extract_aligned_pointer_as_index %gpu_memref_dst : memref<128xf32> -> index
    %dst_ptr_i64 = arith.index_cast %dst_ptr_idx : index to i64

    %src_ptr_idx = memref.extract_aligned_pointer_as_index %gpu_memref_src : memref<128x32xf32> -> index
    %src_ptr_i64 = arith.index_cast %dst_ptr_idx : index to i64

    %gpu_memref_dst_casted = memref.memory_space_cast %gpu_memref_dst : memref<128xf32> to memref<128xf32, 1>
    %gpu_memref_src_casted = memref.memory_space_cast %gpu_memref_src : memref<128x32xf32> to memref<128x32xf32, 1>

    %stream0_5 = gpu.launch_func async[%stream0_4] @reduction::@cross_sg_intra_lane_1D blocks in (%c1, %c1, %c1) threads in (%c32, %c1, %c1) args(%gpu_memref_dst_casted : memref<128xf32, 1>, %gpu_memref_src_casted : memref<128x32xf32, 1>)

    %stream0_6 = gpu.memcpy async [%stream0_5]  %dst, %gpu_memref_dst : memref<128xf32>, memref<128xf32>
    %stream0_8 = gpu.dealloc async [%stream0_6] %gpu_memref_dst : memref<128xf32>
    gpu.wait [%stream0_8]
    return
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %dst = memref.alloc() : memref<128xf32>
    %src = memref.alloc() : memref<128x32xf32>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index

    %c0_f32 = arith.constant 0. : f32
    scf.for %i = %c0 to %c128 step %c1 {
      scf.for %j = %c0 to %c32 step %c1 {
        %i_f32 = arith.index_cast %i : index to i32
        %j_f32 = arith.index_cast %j : index to i32
        %i_float = arith.sitofp %i_f32 : i32 to f32
        %j_float = arith.sitofp %j_f32 : i32 to f32
        %c1000_f32 = arith.constant 1000.0 : f32
        %j_scaled = arith.divf %j_float, %c1000_f32 : f32
        %val = arith.addf %i_float, %j_scaled : f32
        // Input is in format (#row_idx).(#col_idx/1000.0)
        memref.store %i_float, %src[%i, %j] : memref<128x32xf32>
      }
    }
        %c0_i64 = arith.constant 0 : i64

    scf.for %i = %c0 to %c128 step %c1 {
      memref.store %c0_f32, %dst[%i] : memref<128xf32>
    }
    call @test(%dst, %src) : (memref<128xf32>, memref<128x32xf32>) -> ()
    %dst_cast = memref.cast %dst : memref<128xf32> to memref<*xf32>
    %src_cast = memref.cast %src : memref<128x32xf32> to memref<*xf32>

    // CHECK: Unranked Memref base@ = 0x{{[0-9a-f]+}}
    // CHECK-COUNT-32: 496
    call @printMemrefF32(%dst_cast) : (memref<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF32(%ptr : memref<*xf32>) attributes { llvm.emit_c_interface }
}
