// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-llvm-joint-matrix.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-llvm-joint-matrix.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck

// NOTE: This test case provides an end-to-end example of XeGPU SIMT mode ops to SPIR-V JointMatrix ops lowering

#sg_map_fp16_a = #xegpu.sg_map<{mma_block_size = [8, 16], wi_layout = [1, 16], wi_data = [1, 1]}>
#sg_map_fp16_b = #xegpu.sg_map<{mma_block_size = [16, 16], wi_layout = [1, 16], wi_data = [1, 1]}>
#sg_map_fp16_c = #xegpu.sg_map<{mma_block_size = [8, 16], wi_layout = [1, 16], wi_data = [1, 1]}>
module @gemm attributes {gpu.container_module} {
  func.func @test(%A: memref<1024x1024xf16>, %B: memref<1024x1024xf16>, %C: memref<1024x1024xf32>) -> memref<1024x1024xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %A_gpu = gpu.alloc  host_shared () : memref<1024x1024xf16>
    memref.copy %A, %A_gpu : memref<1024x1024xf16> to memref<1024x1024xf16>
    %B_gpu = gpu.alloc  host_shared () : memref<1024x1024xf16>
    memref.copy %B, %B_gpu : memref<1024x1024xf16> to memref<1024x1024xf16>
    %C_gpu = gpu.alloc  host_shared () : memref<1024x1024xf32>
    memref.copy %C, %C_gpu : memref<1024x1024xf32> to memref<1024x1024xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c128, %c64, %c1) threads in (%c1, %c16, %c1) args(%A_gpu : memref<1024x1024xf16>, %B_gpu : memref<1024x1024xf16>, %C_gpu : memref<1024x1024xf32>)
    gpu.dealloc  %A_gpu : memref<1024x1024xf16>
    gpu.dealloc  %B_gpu : memref<1024x1024xf16>
    return %C_gpu : memref<1024x1024xf32>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorAnyINTEL, JointMatrixINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_joint_matrix]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%a: memref<1024x1024xf16>, %b: memref<1024x1024xf16>, %c: memref<1024x1024xf32>) kernel attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c64 = arith.constant 64 : index
      %c1024 = arith.constant 1024 : index

      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y
      %m = arith.muli %block_id_x, %c8 : index
      %n = arith.muli %block_id_y, %c16 : index

      %1 = xegpu.create_nd_tdesc %a[%m, %c0] {boundary_check = false} : memref<1024x1024xf16> -> !xegpu.tensor_desc<8x16xf16, #sg_map_fp16_a>
      %2 = xegpu.create_nd_tdesc %b[%c0, %n] {boundary_check = false} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16, #sg_map_fp16_b>
      %tmpC = arith.constant dense<0.0> : vector<8xf32>
      %3 = vector.shape_cast %tmpC : vector<8xf32> to vector<8x1xf32>
      %tmp0, %tmp1, %result = scf.for %k= %c0 to %c1024 step %c16 iter_args(%subA = %1, %subB = %2, %subC = %3)
              -> (!xegpu.tensor_desc<8x16xf16, #sg_map_fp16_a>, !xegpu.tensor_desc<16x16xf16, #sg_map_fp16_b>, vector<8x1xf32>) {
        %4 = xegpu.load_nd %subA : !xegpu.tensor_desc<8x16xf16, #sg_map_fp16_a> -> vector<8x1xf16>
        %5 = xegpu.load_nd %subB {packed} : !xegpu.tensor_desc<16x16xf16, #sg_map_fp16_b> -> vector<8x1x2xf16>
        %6 = xegpu.dpas %4, %5, %subC  : vector<8x1xf16>, vector<8x1x2xf16>, vector<8x1xf32> -> vector<8x1xf32>
        %7 = xegpu.update_nd_offset %subA, [%c0, %c16] : !xegpu.tensor_desc<8x16xf16, #sg_map_fp16_a>
            -> !xegpu.tensor_desc<8x16xf16, #sg_map_fp16_a>
        %8 = xegpu.update_nd_offset %subB, [%c16, %c0] : !xegpu.tensor_desc<16x16xf16, #sg_map_fp16_b>
            -> !xegpu.tensor_desc<16x16xf16, #sg_map_fp16_b>
        scf.yield %7, %8, %6: !xegpu.tensor_desc<8x16xf16, #sg_map_fp16_a>, !xegpu.tensor_desc<16x16xf16, #sg_map_fp16_b>, vector<8x1xf32>
      }
      %9 = xegpu.create_nd_tdesc %c[%m, %n] {boundary_check = false} : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #sg_map_fp16_c>
      xegpu.store_nd %result, %9 : vector<8x1xf32>, !xegpu.tensor_desc<8x16xf32, #sg_map_fp16_c>
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
    %C = memref.alloc() : memref<1024x1024xf32>
    %C_ref = memref.alloc() : memref<1024x1024xf32>
    // intialize matrix A ; A[i, j] = j
    scf.for %i = %c0 to %c1024 step %c1 {
      scf.for %j = %c0 to %c1024 step %c1 {
        %t = index.castu %j : index to i16
        %val = arith.uitofp %t : i16 to f16
        memref.store %val, %A[%i, %j] : memref<1024x1024xf16>
      }
    }
    // make matrix B an identity matrix
    scf.for %i = %c0 to %c1024 step %c1 {
      scf.for %j = %c0 to %c1024 step %c1 {
        %i_i32 = index.castu %i : index to i32
        %j_i32 = index.castu %j : index to i32
        %i_j_same = arith.cmpi eq, %i_i32, %j_i32 : i32

        scf.if %i_j_same {
          memref.store %cf_1, %B[%i, %j] : memref<1024x1024xf16>
        } else {
          memref.store %cf_0, %B[%i, %j] : memref<1024x1024xf16>
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
          %b_val = memref.load %B[%k, %j] : memref<1024x1024xf16>
          %t = arith.mulf %a_val, %b_val : f16
          %t_cast = arith.extf %t : f16 to f32
          %c_sum = arith.addf %t_cast, %c_partial : f32
          scf.yield %c_sum : f32
        }
        memref.store %c_val , %C_ref[%i, %j] : memref<1024x1024xf32>
      }
    }
    %2 = call @test(%A, %B, %C) : (memref<1024x1024xf16>, memref<1024x1024xf16>, memref<1024x1024xf32>) -> memref<1024x1024xf32>
    %cast_C = memref.cast %2 : memref<1024x1024xf32> to memref<*xf32>
    %cast_C_ref = memref.cast %C_ref : memref<1024x1024xf32> to memref<*xf32>
    // call @printMemrefF32(%cast_C) : (memref<*xf32>) -> ()
    // call @printMemrefF32(%cast_C_ref) : (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF32(%cast_C, %cast_C_ref) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %A : memref<1024x1024xf16>
    memref.dealloc %B : memref<1024x1024xf16>
    memref.dealloc %C : memref<1024x1024xf32>
    memref.dealloc %C_ref : memref<1024x1024xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
