// RUN: imex-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup zebin-chip=pvc" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_levelzero_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --shared-libs=%irunner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

#data_layout_leading_unit_dim = #xegpu.layout<sg_layout = [1, 16, 1], sg_data = [1, 16, 256]>
module attributes {gpu.container_module} {
  gpu.module @reduction {
    gpu.func @cross_sg_intra_lane_3D_1D(%src: memref<?xf32, 1>, %dst: memref<256xf32, 1>) kernel {
      %src1 = memref.memory_space_cast %src : memref<?xf32, 1> to memref<?xf32>
      %dst1 = memref.memory_space_cast %dst : memref<256xf32, 1> to memref<256xf32>

      // reduce_fullwg_32x16xf32
      %src_ptr_idx = memref.extract_aligned_pointer_as_index %src1 : memref<?xf32> -> index
      %src_ptr_i64 = arith.index_cast %src_ptr_idx : index to i64

      %dst_ptr_idx = memref.extract_aligned_pointer_as_index %dst1 : memref<256xf32> -> index
      %dst_ptr_i64 = arith.index_cast %dst_ptr_idx : index to i64

      %offset_ld = arith.constant dense<0> : vector<1x256x256xindex>
      %mask_ld = arith.constant dense<1> : vector<1x256x256xi1>
      %val = xegpu.load %src_ptr_i64[%offset_ld], %mask_ld {layout = #data_layout_leading_unit_dim} : i64, vector<1x256x256xindex>, vector<1x256x256xi1> -> vector<1x256x256xf32>

      %acc = arith.constant dense<0.0> : vector<1xf32>
      %res = vector.multi_reduction <add>, %val, %acc [1, 2] : vector<1x256x256xf32> to vector<1xf32>

      %offset = arith.constant dense<0> : vector<1xindex>
      %mask = arith.constant dense<1> : vector<1xi1>

      xegpu.store %res, %dst_ptr_i64[%offset], %mask { layout = #xegpu.slice<#data_layout_leading_unit_dim, dims = [1, 2]>} : vector<1xf32>, i64, vector<1xindex>, vector<1xi1>
      gpu.return
    }
  }

  func.func @test(%src : memref<65536xf32>, %dst : memref<256xf32>) attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %c65536 = arith.constant 65536 : index

    %stream0_0 = gpu.wait async

    %gpu_memref_src, %stream0_1 = gpu.alloc async [%stream0_0] (%c65536) : memref<?xf32>
    %stream0_2 = gpu.memcpy async [%stream0_1] %gpu_memref_src, %src  : memref<?xf32>, memref<65536xf32>

    %gpu_memref_dst, %stream0_3 = gpu.alloc async [%stream0_2] () : memref<256xf32>
    %stream0_4 = gpu.memcpy async [%stream0_3] %gpu_memref_dst, %dst  : memref<256xf32>, memref<256xf32>

    %gpu_memref_src_casted = memref.memory_space_cast %gpu_memref_src : memref<?xf32> to memref<?xf32, 1>
    %gpu_memref_dst_casted = memref.memory_space_cast %gpu_memref_dst : memref<256xf32> to memref<256xf32, 1>

    %stream0_5 = gpu.launch_func async[%stream0_4] @reduction::@cross_sg_intra_lane_3D_1D blocks in (%c1, %c1, %c1) threads in (%c1024, %c1, %c1) args(%gpu_memref_src_casted : memref<?xf32, 1>, %gpu_memref_dst_casted : memref<256xf32, 1>)

    %stream0_6 = gpu.memcpy async [%stream0_5]  %dst, %gpu_memref_dst : memref<256xf32>, memref<256xf32>
    %stream0_7 = gpu.dealloc async [%stream0_6] %gpu_memref_src : memref<?xf32>
    %stream0_8 = gpu.dealloc async [%stream0_7] %gpu_memref_dst : memref<256xf32>
    gpu.wait [%stream0_8]
    return
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %c65536 = arith.constant 65536 : index
    %src = memref.alloc() : memref<65536xf32>
    %dst = memref.alloc() : memref<256xf32>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index

    %c0_f32 = arith.constant 0. : f32
    %c1_f32 = arith.constant 1. : f32

    // Initialize source with value 1.0 for all elements (1*256*256 = 65536 elements)
    scf.for %i = %c0 to %c65536 step %c1 {
      memref.store %c1_f32, %src[%i] : memref<65536xf32>
    }

    // Initialize destination with 0
    scf.for %i = %c0 to %c256 step %c1 {
      memref.store %c0_f32, %dst[%i] : memref<256xf32>
    }

    call @test(%src, %dst) : (memref<65536xf32>, memref<256xf32>) -> ()

    %dst_cast = memref.cast %dst : memref<256xf32> to memref<*xf32>

    // CHECK: Unranked Memref base@ = 0x{{[0-9a-f]+}}
    // CHECK-NEXT: [65536
    call @printMemrefF32(%dst_cast) : (memref<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF32(%ptr : memref<*xf32>) attributes { llvm.emit_c_interface }
}