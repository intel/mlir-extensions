// RUN: imex-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=lane" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_levelzero_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --shared-libs=%irunner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

// Example of pass pipeline usage:
// %python_executable %imex_runner --requires=mlir-levelzero-runtime,spirv-backend -i %s --pass-pipeline-file=%p/xegpu-to-llvm.pp \
//                                        --runner mlir-runner -e main \
//                                        --entry-point-result=void \
//                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

#sg_map_a_f16 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>
#sg_map_b_f16 = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>
#sg_map_c_f32 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>

module @gemm attributes {gpu.container_module} {
  gpu.module @kernel {
    gpu.func @load_store_2d_dpas(%a: memref<8x16xf16, 1>, %b: memref<16x16xf16, 1>, %c: memref<8x16xf32, 1>) kernel {
      // Casts to satisfy tensor creation
      %a_tdesc_memref = memref.memory_space_cast %a : memref<8x16xf16, 1> to memref<8x16xf16>
      %b_tdesc_memref = memref.memory_space_cast %b : memref<16x16xf16, 1> to memref<16x16xf16>
      %c_tdesc_memref = memref.memory_space_cast %c : memref<8x16xf32, 1> to memref<8x16xf32>

      %a_tdesc = xegpu.create_nd_tdesc %a_tdesc_memref : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<memory_space = global>>
      %b_tdesc = xegpu.create_nd_tdesc %b_tdesc_memref : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space = global>>
      %c_tdesc = xegpu.create_nd_tdesc %c_tdesc_memref : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space = global>>

      %a_loaded = xegpu.load_nd %a_tdesc[0, 0] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<memory_space = global>> -> vector<8xf16>
      %b_loaded = xegpu.load_nd %b_tdesc[0, 0] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<memory_space = global>> -> vector<16xf16>
      %c_loaded = xegpu.load_nd %c_tdesc[0, 0] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space = global>> -> vector<8xf32>

      %d = xegpu.dpas %a_loaded, %b_loaded, %c_loaded {a_layout = #sg_map_a_f16, b_layout = #sg_map_b_f16, c_layout = #sg_map_c_f32} : vector<8xf16>, vector<16xf16>, vector<8xf32> -> vector<8xf32>

      xegpu.store_nd %d, %c_tdesc[0, 0] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}>: vector<8xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space = global>>
      gpu.return
    }
  }

  func.func @test(%a : memref<8x16xf16>, %b : memref<16x16xf16>, %c : memref<8x16xf32>) -> memref<8x16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %memref_a = gpu.alloc  () : memref<8x16xf16>
    gpu.memcpy %memref_a, %a : memref<8x16xf16>, memref<8x16xf16>
    %memref_b = gpu.alloc  () : memref<16x16xf16>
    gpu.memcpy %memref_b, %b : memref<16x16xf16>, memref<16x16xf16>
    %memref_c = gpu.alloc  () : memref<8x16xf32>
    gpu.memcpy %memref_c, %c : memref<8x16xf32>, memref<8x16xf32>

    %a_gpu = memref.memory_space_cast %memref_a : memref<8x16xf16> to memref<8x16xf16, 1>
    %b_gpu = memref.memory_space_cast %memref_b : memref<16x16xf16> to memref<16x16xf16, 1>
    %c_gpu = memref.memory_space_cast %memref_c : memref<8x16xf32> to memref<8x16xf32, 1>

    gpu.launch_func @kernel::@load_store_2d_dpas blocks in (%c1, %c1, %c1) threads in (%c16, %c1, %c1) args(%a_gpu : memref<8x16xf16, 1>, %b_gpu : memref<16x16xf16, 1>, %c_gpu : memref<8x16xf32, 1>)
    gpu.wait
    gpu.memcpy %c, %memref_c : memref<8x16xf32>, memref<8x16xf32>
    gpu.dealloc %memref_a : memref<8x16xf16>
    gpu.dealloc %memref_b : memref<16x16xf16>
    gpu.dealloc %memref_c : memref<8x16xf32>
    return %c : memref<8x16xf32>
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c11_f32 = arith.constant 11.11 : f16

    %A = memref.alloc() : memref<8x16xf16>
    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %j = %c0 to %c16 step %c1 {
        %row_idx = arith.index_cast %i : index to i32
        %row = arith.sitofp %row_idx : i32 to f16
        memref.store %row, %A[%i, %j] : memref<8x16xf16>
      }
    }
    %B = memref.alloc() : memref<16x16xf16>
    scf.for %i = %c0 to %c16 step %c1 {
      scf.for %j = %c0 to %c16 step %c1 {
        %col_idx = arith.index_cast %j : index to i32
        %col = arith.sitofp %col_idx : i32 to f16
        memref.store %col, %B[%i, %j] : memref<16x16xf16>
      }
    }

    %C = memref.alloc() : memref<8x16xf32>
    %c0_f16 = arith.constant 0.0 : f32
    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %j = %c0 to %c16 step %c1 {
        memref.store %c0_f16, %C[%i, %j] : memref<8x16xf32>
      }
    }

    %C_res = call @test(%A, %B, %C) : (memref<8x16xf16>, memref<16x16xf16>, memref<8x16xf32>) -> memref<8x16xf32>
    %C_res_cast = memref.cast %C_res : memref<8x16xf32> to memref<*xf32>
    // %A_cast = memref.cast %A : memref<8x16xf16> to memref<*xf16>
    // call @printMemrefF16(%A_cast) : (memref<*xf16>) -> ()
    // %B_cast = memref.cast %B : memref<16x16xf16> to memref<*xf16>
    // call @printMemrefF16(%B_cast) : (memref<*xf16>) -> ()

    call @printMemrefF32(%C_res_cast) : (memref<*xf32>) -> ()
    // CHECK: Unranked Memref base@ = 0x{{[0-9a-f]+}}
    // CHECK-NEXT: [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]
    // CHECK-NEXT: [0,   16,   32,   48,   64,   80,   96,   112,   128,   144,   160,   176,   192,   208,   224,   240]
    // CHECK-NEXT: [0,   32,   64,   96,   128,   160,   192,   224,   256,   288,   320,   352,   384,   416,   448,   480]
    // CHECK-NEXT: [0,   48,   96,   144,   192,   240,   288,   336,   384,   432,   480,   528,   576,   624,   672,   720]
    // CHECK-NEXT: [0,   64,   128,   192,   256,   320,   384,   448,   512,   576,   640,   704,   768,   832,   896,   960]
    // CHECK-NEXT: [0,   80,   160,   240,   320,   400,   480,   560,   640,   720,   800,   880,   960,   1040,   1120,   1200]
    // CHECK-NEXT: [0,   96,   192,   288,   384,   480,   576,   672,   768,   864,   960,   1056,   1152,   1248,   1344,   1440]
    // CHECK-NEXT: [0,   112,   224,   336,   448,   560,   672,   784,   896,   1008,   1120,   1232,   1344,   1456,   1568,   1680]


    memref.dealloc %A : memref<8x16xf16>
    return
  }
  func.func private @printMemrefF32(%ptr : memref<*xf32>)
  func.func private @printMemrefF16(%ptr : memref<*xf16>) attributes {llvm.emit_c_interface}
}
