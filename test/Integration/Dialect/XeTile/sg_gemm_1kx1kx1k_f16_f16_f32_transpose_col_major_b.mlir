// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xetile-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xetile-to-func-vc.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck


module @gemm attributes {gpu.container_module} {
  func.func @test(%A: memref<1024x1024xf16>, %B: memref<1024x1024xf16, strided<[1, 1024]>>, %C: memref<1024x1024xf32>) -> memref<1024x1024xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c512 = arith.constant 512 : index
    %A_gpu = gpu.alloc  host_shared () : memref<1024x1024xf16>
    memref.copy %A, %A_gpu : memref<1024x1024xf16> to memref<1024x1024xf16>
    %B_gpu = gpu.alloc  host_shared () : memref<1024x1024xf16>
    %B_gpu_cm = memref.reinterpret_cast %B_gpu to offset : [0], sizes : [1024,1024], strides : [1, 1024] : memref<1024x1024xf16> to memref<1024x1024xf16, strided<[1, 1024]>>
    memref.copy %B, %B_gpu_cm : memref<1024x1024xf16, strided<[1, 1024]>> to memref<1024x1024xf16, strided<[1, 1024]>>
    %C_gpu = gpu.alloc  host_shared () : memref<1024x1024xf32>
    memref.copy %C, %C_gpu : memref<1024x1024xf32> to memref<1024x1024xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c64, %c32, %c1) threads in (%c1, %c1, %c1) args(%A_gpu : memref<1024x1024xf16>, %B_gpu_cm : memref<1024x1024xf16, strided<[1, 1024]>>, %C_gpu : memref<1024x1024xf32>)
    gpu.dealloc  %A_gpu : memref<1024x1024xf16>
    gpu.dealloc  %B_gpu : memref<1024x1024xf16>
    return %C_gpu : memref<1024x1024xf32>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%A: memref<1024x1024xf16>, %B_cm: memref<1024x1024xf16, strided<[1, 1024]>>, %C: memref<1024x1024xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c1024 = arith.constant 1024 : index
      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y
      %m = arith.muli %block_id_x, %c16 : index
      %n = arith.muli %block_id_y, %c32 : index
      // intialize C tile and load it
      %c_init_tile = xetile.init_tile %C[%m, %n] : memref<1024x1024xf32> -> !xetile.tile<16x32xf32>
      %c_init_value = xetile.load_tile %c_init_tile  : !xetile.tile<16x32xf32> -> vector<16x32xf32>
      // initalize A and B tiles
      %a_init_tile = xetile.init_tile %A[%m, %c0] : memref<1024x1024xf16> -> !xetile.tile<16x32xf16>
      %b_init_tile = xetile.init_tile %B_cm[%n, %c0] : memref<1024x1024xf16, strided<[1, 1024]>> -> !xetile.tile<32x32xf16, #xetile.tile_attr<order=[0, 1]>>
      // compute the value of C tile by iterating over tiles in k-dimension and doing dpas
      %out:3 = scf.for %k = %c0 to %c1024 step %c32
        iter_args(%a_tile = %a_init_tile, %b_tile = %b_init_tile, %c_value = %c_init_value)
        -> (!xetile.tile<16x32xf16>, !xetile.tile<32x32xf16, #xetile.tile_attr<order=[0, 1]>>, vector<16x32xf32>) {

        // load A and B tiles
        %a_value = xetile.load_tile %a_tile  : !xetile.tile<16x32xf16> -> vector<16x32xf16>
        %b_value = xetile.load_tile %b_tile  : !xetile.tile<32x32xf16, #xetile.tile_attr<order=[0, 1]>> -> vector<32x32xf16>
        %b_trans = vector.transpose %b_value, [1, 0] : vector<32x32xf16> to vector<32x32xf16>
        // perform dpas and accumulate
        %c_new_value = xetile.tile_mma %a_value, %b_trans, %c_value
          : vector<16x32xf16>, vector<32x32xf16>, vector<16x32xf32> -> vector<16x32xf32>
        // update the offsets for A and B tiles
        %a_next_tile = xetile.update_tile_offset %a_tile, [%c0, %c32] : !xetile.tile<16x32xf16>
        %b_next_tile = xetile.update_tile_offset %b_tile, [%c0, %c32] : !xetile.tile<32x32xf16, #xetile.tile_attr<order=[0, 1]>>
        // partial C tile result
        scf.yield %a_next_tile, %b_next_tile, %c_new_value
          : !xetile.tile<16x32xf16>, !xetile.tile<32x32xf16, #xetile.tile_attr<order=[0, 1]>>, vector<16x32xf32>
      }
      // store the final accumulated C tile result back to memory
      xetile.store_tile %out#2, %c_init_tile: vector<16x32xf32>, !xetile.tile<16x32xf32>
      gpu.return
    }
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %cf_0 = arith.constant 0.0 : f16
    %cf_1 = arith.constant 1.0 : f16
    %A = memref.alloc() : memref<1024x1024xf16>
    %B = memref.alloc() : memref<1024x1024xf16>
    // convert B to col-major
    %B_cm = memref.reinterpret_cast %B to offset : [0], sizes : [1024,1024], strides : [1, 1024] : memref<1024x1024xf16> to memref<1024x1024xf16, strided<[1, 1024], offset:0>>
    %C = memref.alloc() : memref<1024x1024xf32>
    %C_ref = memref.alloc() : memref<1024x1024xf32>
    // intialize matrix B ; B[i, j] = j
    scf.for %i = %c0 to %c1024 step %c1 {
      scf.for %j = %c0 to %c1024 step %c1 {
        %t = index.castu %j : index to i16
        %val = arith.uitofp %t : i16 to f16
        memref.store %val, %B_cm[%i, %j] : memref<1024x1024xf16, strided<[1, 1024]>>
      }
    }
    // make matrix A an identity matrix
    scf.for %i = %c0 to %c1024 step %c1 {
      scf.for %j = %c0 to %c1024 step %c1 {
        %i_i32 = index.castu %i : index to i32
        %j_i32 = index.castu %j : index to i32
        %i_j_same = arith.cmpi eq, %i_i32, %j_i32 : i32

        scf.if %i_j_same {
          memref.store %cf_1, %A[%i, %j] : memref<1024x1024xf16>
        } else {
          memref.store %cf_0, %A[%i, %j] : memref<1024x1024xf16>
        }
      }
    }
    // intialize matrix C and C_ref ; C[i, j] = 0
    %c0_f32 = arith.constant 0.0 : f32
    scf.for %i = %c0 to %c1024 step %c1 {
      scf.for %j = %c0 to %c1024 step %c1 {
        memref.store %c0_f32, %C[%i, %j] : memref<1024x1024xf32>
        memref.store %c0_f32, %C_ref[%i, %j] : memref<1024x1024xf32>
      }
    }
    // compute C for reference
    scf.for %i = %c0 to %c1024 step %c1 {
      scf.for %j = %c0 to %c1024 step %c1 {
        %c_curr = memref.load %C_ref[%i, %j] : memref<1024x1024xf32>
        %c_val = scf.for %k = %c0 to %c1024 step %c1 iter_args(%c_partial = %c_curr) -> f32 {
          %a_val = memref.load %A[%i, %k] : memref<1024x1024xf16>
          %b_val = memref.load %B_cm[%j, %k] : memref<1024x1024xf16, strided<[1, 1024]>>
          %t = arith.mulf %a_val, %b_val : f16
          %t_cast = arith.extf %t : f16 to f32
          %c_sum = arith.addf %t_cast, %c_partial : f32
          scf.yield %c_sum : f32
        }
        memref.store %c_val , %C_ref[%i, %j] : memref<1024x1024xf32>
      }
    }
    %2 = call @test(%A, %B_cm, %C) : (memref<1024x1024xf16>, memref<1024x1024xf16, strided<[1, 1024]>>, memref<1024x1024xf32>) -> memref<1024x1024xf32>
    // %cast = memref.cast %B : memref<1024x1024xf16> to memref<*xf16>
    // call @printMemrefF16(%cast) : (memref<*xf16>) -> ()
    %cast_C = memref.cast %2 : memref<1024x1024xf32> to memref<*xf32>
    %cast_C_ref = memref.cast %C_ref : memref<1024x1024xf32> to memref<*xf32>
    // call @printMemrefF32(%cast_C) : (memref<*xf32>) -> ()
    // call @printMemrefF32(%cast_C_ref) : (memref<*xf32>) -> ()
    // %C_row_0 = memref.subview %2[0, 0][1, 1024][1, 1] : memref<1024x1024xf32> to memref<1x1024xf32>
    // %C_row_0_cast = memref.cast %C_row_0 : memref<1x1024xf32> to memref<*xf32>
    // call @printMemrefF32(%C_row_0_cast) : (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF32(%cast_C, %cast_C_ref) : (memref<*xf32>, memref<*xf32>) -> ()
    // memref.dealloc %A : memref<1024x1024xf16>
    // memref.dealloc %B : memref<1024x1024xf16>
    // memref.dealloc %C : memref<1024x1024xf32>
    // memref.dealloc %C_ref : memref<1024x1024xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
