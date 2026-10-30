// RUN: mlir-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=lane zebin-chip=cri" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_levelzero_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

// Round trip through the fp4 conversions: xevm.truncf packs f16 down to e2m1,
// xevm.extf unpacks it back. The inputs are all exactly representable in e2m1
// (it holds 0, +-0.5, +-1, +-1.5, +-2, +-3, +-4, +-6), so the round trip has to
// reproduce them bit for bit and any error in either direction shows up as a
// changed value rather than as rounding noise.
//
// xevm.extf is otherwise only covered by the conversion lit tests, so this is
// the execution coverage for the up-conversion.

module @fp4_roundtrip attributes {gpu.container_module} {
  gpu.module @kernel {
    gpu.func @e2m1_roundtrip(%out: !llvm.ptr<1>) kernel {
      // 16 lanes x 8 values, one f32 per row per lane, matching the store tile.
      // Every lane converts the same 8 values.
      %vals = arith.constant dense<[1.0, -1.0, 0.5, -0.5, 2.0, -3.0, 4.0, 6.0]>
          : vector<8xf16>

      // f16 -> e2m1: 8 values pack into 4 bytes.
      %packed = xevm.truncf %vals { src_etype = f16, dst_etype = e2m1 }
          : (vector<8xf16>) -> vector<4xi8>
      // e2m1 -> f16: back to 8 values.
      %unpacked = xevm.extf %packed { src_etype = e2m1, dst_etype = f16 }
          : (vector<4xi8>) -> vector<8xf16>

      %as_f32 = arith.extf %unpacked : vector<8xf16> to vector<8xf32>
      %bits = vector.bitcast %as_f32 : vector<8xf32> to vector<8xi32>

      %x = arith.constant 0 : i32
      %y = arith.constant 0 : i32
      %base_width = arith.constant 64 : i32
      %base_height = arith.constant 8 : i32
      %base_pitch = arith.constant 64 : i32
      xevm.blockstore2d %out, %base_width, %base_height, %base_pitch, %x, %y, %bits
          <{elem_size_in_bits=32 : i32, tile_width=16 : i32, tile_height=8 : i32}>
          : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi32>)
      gpu.return
    }
  }

  func.func @test(%out : memref<8x16xf32>) -> memref<8x16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index

    %memref_out = gpu.alloc() : memref<8x16xf32>
    gpu.memcpy %memref_out, %out : memref<8x16xf32>, memref<8x16xf32>
    %out_ptr_as_idx = memref.extract_aligned_pointer_as_index %memref_out : memref<8x16xf32> -> index
    %out_ptr_as_i64 = arith.index_cast %out_ptr_as_idx : index to i64
    %out_ptr = llvm.inttoptr %out_ptr_as_i64 : i64 to !llvm.ptr
    %out_ptr_casted = llvm.addrspacecast %out_ptr : !llvm.ptr to !llvm.ptr<1>

    gpu.launch_func @kernel::@e2m1_roundtrip blocks in (%c1, %c1, %c1) threads in (%c16, %c1, %c1)
        args(%out_ptr_casted : !llvm.ptr<1>)

    %res = memref.alloc() : memref<8x16xf32>
    gpu.memcpy %res, %memref_out : memref<8x16xf32>, memref<8x16xf32>
    gpu.dealloc %memref_out : memref<8x16xf32>
    return %res : memref<8x16xf32>
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c0f32 = arith.constant 0.0 : f32

    %O = memref.alloc() : memref<8x16xf32>
    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %j = %c0 to %c16 step %c1 {
        memref.store %c0f32, %O[%i, %j] : memref<8x16xf32>
      }
    }

    %O_res = call @test(%O) : (memref<8x16xf32>) -> memref<8x16xf32>
    %O_cast = memref.cast %O_res : memref<8x16xf32> to memref<*xf32>
    call @printMemrefF32(%O_cast) : (memref<*xf32>) -> ()

    // The store writes one value per row, so row r holds the r-th round-tripped
    // value replicated across the 16 lanes.
    // CHECK: Unranked Memref base@ = 0x{{[0-9a-f]+}}
    // CHECK: [1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1]
    // CHECK-NEXT: [-1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1]
    // CHECK-NEXT: [0.5,   0.5,   0.5,   0.5,   0.5,   0.5,   0.5,   0.5,   0.5,   0.5,   0.5,   0.5,   0.5,   0.5,   0.5,   0.5]
    // CHECK-NEXT: [-0.5,   -0.5,   -0.5,   -0.5,   -0.5,   -0.5,   -0.5,   -0.5,   -0.5,   -0.5,   -0.5,   -0.5,   -0.5,   -0.5,   -0.5,   -0.5]
    // CHECK-NEXT: [2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2]
    // CHECK-NEXT: [-3,   -3,   -3,   -3,   -3,   -3,   -3,   -3,   -3,   -3,   -3,   -3,   -3,   -3,   -3,   -3]
    // CHECK-NEXT: [4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4]
    // CHECK-NEXT: [6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6]

    memref.dealloc %O : memref<8x16xf32>
    memref.dealloc %O_res : memref<8x16xf32>
    return
  }
  func.func private @printMemrefF32(%ptr : memref<*xf32>) attributes { llvm.emit_c_interface }
}
