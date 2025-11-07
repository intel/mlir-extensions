// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime,spirv-backend -i %s --pass-pipeline-file=%p/xegpu-to-llvm.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck


#map = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], inst_data = [8, 16]>
module @gemm attributes {gpu.container_module} {
  func.func @test_fast_math(%input: memref<256x256xf32>) -> (memref<256x256xf32>, memref<256x256xf32>) attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c512 = arith.constant 512 : index
    %input_gpu = gpu.alloc () : memref<256x256xf32>
    gpu.memcpy %input_gpu, %input : memref<256x256xf32>, memref<256x256xf32>
    %result_gpu = gpu.alloc () : memref<256x256xf32>
    %result_gpu_with_fastmath = gpu.alloc () : memref<256x256xf32>
    // NOTE: Here we can't use [8, 64] wi threads following
    // the SG thread layout of [8, 4]. Because runtime will linearize
    // the x dimension first (we need y dimension to be linearized first).
    // So just use linearized thread layout of [512, 1] wi threads.
    gpu.launch_func  @math_exp_module::@gpu_exp blocks in (%c1, %c1, %c1) threads in (%c512, %c1, %c1) args(%input_gpu : memref<256x256xf32>, %result_gpu : memref<256x256xf32>)
    gpu.launch_func  @math_exp_fastmath_module::@gpu_exp_with_fastmath blocks in (%c1, %c1, %c1) threads in (%c512, %c1, %c1) args(%input_gpu : memref<256x256xf32>, %result_gpu_with_fastmath : memref<256x256xf32>)

    %result_host = memref.alloc() : memref<256x256xf32>
    %result_host_with_fastmath = memref.alloc() : memref<256x256xf32>
    gpu.memcpy %result_host, %result_gpu : memref<256x256xf32>, memref<256x256xf32>
    gpu.memcpy %result_host_with_fastmath, %result_gpu_with_fastmath : memref<256x256xf32>, memref<256x256xf32>
    gpu.dealloc %input_gpu : memref<256x256xf32>
    gpu.dealloc %result_gpu : memref<256x256xf32>
    gpu.dealloc %result_gpu_with_fastmath : memref<256x256xf32>
    return %result_host, %result_host_with_fastmath : memref<256x256xf32>, memref<256x256xf32>
  }

  gpu.module @math_exp_module   {
    gpu.func @gpu_exp(%input_gpu : memref<256x256xf32>, %result_gpu : memref<256x256xf32>) kernel  {
      %c256 = arith.constant 256 : index
      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y
      %m = arith.muli %block_id_x, %c256 : index
      %n = arith.muli %block_id_y, %c256 : index
      %input_tdesc = xegpu.create_nd_tdesc %input_gpu : memref<256x256xf32> -> !xegpu.tensor_desc<256x256xf32, #map>
      %input_val = xegpu.load_nd %input_tdesc[%m, %n] : !xegpu.tensor_desc<256x256xf32, #map> -> vector<256x256xf32>
      %result_val = math.exp %input_val {layout_result_0 = #map} : vector<256x256xf32>
      %result_tdesc = xegpu.create_nd_tdesc %result_gpu : memref<256x256xf32> -> !xegpu.tensor_desc<256x256xf32, #map>
      xegpu.store_nd %result_val, %result_tdesc[%m, %n] : vector<256x256xf32>, !xegpu.tensor_desc<256x256xf32, #map>
      gpu.return
    }


  }

  gpu.module @math_exp_fastmath_module {
    // Kernel with fastmath attribute
    gpu.func @gpu_exp_with_fastmath(%input_gpu_with_fast_math : memref<256x256xf32>, %result_gpu_with_fastmath : memref<256x256xf32>) kernel  {
      %c256 = arith.constant 256 : index
      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y
      %m = arith.muli %block_id_x, %c256 : index
      %n = arith.muli %block_id_y, %c256 : index
      %input_tdesc = xegpu.create_nd_tdesc %input_gpu_with_fast_math : memref<256x256xf32> -> !xegpu.tensor_desc<256x256xf32, #map>
      %input_val = xegpu.load_nd %input_tdesc[%m, %n] : !xegpu.tensor_desc<256x256xf32, #map> -> vector<256x256xf32>
      %result_val = math.exp %input_val fastmath<fast> {layout_result_0 = #map} : vector<256x256xf32>
      %result_tdesc = xegpu.create_nd_tdesc %result_gpu_with_fastmath : memref<256x256xf32> -> !xegpu.tensor_desc<256x256xf32, #map>
      xegpu.store_nd %result_val, %result_tdesc[%m, %n] : vector<256x256xf32>, !xegpu.tensor_desc<256x256xf32, #map>
      gpu.return
    }
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2_f32 = arith.constant 2.2 : f32
    %c256 = arith.constant 256 : index

    %init_val_min_range = arith.constant -0.5 : f32
    %init_val_max_range = arith.constant 0.5 : f32
    %false = arith.constant false


    %input = memref.alloc() : memref<256x256xf32>
    %cpu_ref_result = memref.alloc() : memref<256x256xf32>

    // Initialize

    // Initialize to constant values
    // scf.for %arg0 = %c0 to %c256 step %c1 {
    //   scf.for %arg1 = %c0 to %c256 step %c1 {
    //     memref.store %c2_f32, %input[%arg0, %arg1] : memref<256x256xf32>
    //     memref.store %c2_f32, %input_ref[%arg0, %arg1] : memref<256x256xf32>
    //   }
    // }

    // Initialize to random values
    %input_cast = memref.cast %input : memref<256x256xf32> to memref<*xf32>
    call @fillResource1DRandomF32(%input_cast, %init_val_min_range, %init_val_max_range, %false) : (memref<*xf32>, f32, f32, i1) -> ()


    // Run CPU version
    scf.for %arg0 = %c0 to %c256 step %c1 {
      scf.for %arg1 = %c0 to %c256 step %c1 {
        %val = memref.load %input[%arg0, %arg1] : memref<256x256xf32>
        %res_val = math.exp %val : f32
        memref.store %res_val, %cpu_ref_result[%arg0, %arg1] : memref<256x256xf32>
      }
    }

    // Run GPU version.
    %gpu_result, %gpu_result_fastmath = call @test_fast_math(%input) : (memref<256x256xf32>) -> (memref<256x256xf32>, memref<256x256xf32>)
    %gpu_result_cast = memref.cast %gpu_result : memref<256x256xf32> to memref<*xf32>
    %gpu_result_fastmath_cast = memref.cast %gpu_result_fastmath : memref<256x256xf32> to memref<*xf32>
    %cpu_ref_result_cast = memref.cast %cpu_ref_result : memref<256x256xf32> to memref<*xf32>

    // call @printMemrefF32(%gpu_result_cast) : (memref<*xf32>) -> ()

    call @printMaxErrorF32(%cpu_ref_result_cast, %gpu_result_cast) : (memref<*xf32>, memref<*xf32>) -> ()
    call @printMaxErrorF32(%cpu_ref_result_cast, %gpu_result_fastmath_cast) : (memref<*xf32>, memref<*xf32>) -> ()

    // CHECK: [ALLCLOSE: TRUE]
    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF32(%cpu_ref_result_cast, %gpu_result_cast) : (memref<*xf32>, memref<*xf32>) -> ()
    call @printAllcloseF32(%cpu_ref_result_cast, %gpu_result_fastmath_cast) : (memref<*xf32>, memref<*xf32>) -> ()

    memref.dealloc %input : memref<256x256xf32>
    memref.dealloc %cpu_ref_result : memref<256x256xf32>
    memref.dealloc %gpu_result : memref<256x256xf32>
    memref.dealloc %gpu_result_fastmath : memref<256x256xf32>
    return
  }
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMaxErrorF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF32(memref<*xf32>, f32, f32, i1) attributes {llvm.emit_c_interface}

}
