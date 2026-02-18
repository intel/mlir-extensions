// RUN: imex-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup" \
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

#map = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], inst_data = [8, 16]>
module @gemm attributes {gpu.container_module} {
  func.func @test_fast_math(%input1: memref<256x256xf32>, %input2: memref<256x256xf32>) -> (memref<256x256xf32>, memref<256x256xf32>) attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c512 = arith.constant 512 : index
    %input1_gpu = gpu.alloc () : memref<256x256xf32>
    gpu.memcpy %input1_gpu, %input1 : memref<256x256xf32>, memref<256x256xf32>
    %input2_gpu = gpu.alloc () : memref<256x256xf32>
    gpu.memcpy %input2_gpu, %input2 : memref<256x256xf32>, memref<256x256xf32>
    %result_gpu = gpu.alloc () : memref<256x256xf32>
    %result_gpu_with_fastmath = gpu.alloc () : memref<256x256xf32>
    // NOTE: Here we can't use [8, 64] wi threads following
    // the SG thread layout of [8, 4]. Because runtime will linearize
    // the x dimension first (we need y dimension to be linearized first).
    // So just use linearized thread layout of [512, 1] wi threads.
    gpu.launch_func  @arith_maximumf_module::@gpu_maximumf blocks in (%c1, %c1, %c1) threads in (%c512, %c1, %c1) args(%input1_gpu : memref<256x256xf32>, %input2_gpu : memref<256x256xf32>, %result_gpu : memref<256x256xf32>)
    gpu.launch_func  @arith_maximumf_fastmath_module::@gpu_maximumf_with_fastmath blocks in (%c1, %c1, %c1) threads in (%c512, %c1, %c1) args(%input1_gpu : memref<256x256xf32>, %input2_gpu : memref<256x256xf32>, %result_gpu_with_fastmath : memref<256x256xf32>)

    %result_host = memref.alloc() : memref<256x256xf32>
    %result_host_with_fastmath = memref.alloc() : memref<256x256xf32>
    gpu.memcpy %result_host, %result_gpu : memref<256x256xf32>, memref<256x256xf32>
    gpu.memcpy %result_host_with_fastmath, %result_gpu_with_fastmath : memref<256x256xf32>, memref<256x256xf32>
    gpu.dealloc %input1_gpu : memref<256x256xf32>
    gpu.dealloc %input2_gpu : memref<256x256xf32>
    gpu.dealloc %result_gpu : memref<256x256xf32>
    return %result_host, %result_host_with_fastmath : memref<256x256xf32>, memref<256x256xf32>
  }

  gpu.module @arith_maximumf_module {
    gpu.func @gpu_maximumf(%input1_gpu : memref<256x256xf32>, %input2_gpu : memref<256x256xf32>, %result_gpu : memref<256x256xf32>) kernel  {
      %c256 = arith.constant 256 : index
      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y
      %m = arith.muli %block_id_x, %c256 : index
      %n = arith.muli %block_id_y, %c256 : index
      %input_tdesc_1 = xegpu.create_nd_tdesc %input1_gpu : memref<256x256xf32> -> !xegpu.tensor_desc<256x256xf32, #map>
      %input_val_1 = xegpu.load_nd %input_tdesc_1[%m, %n] {layout = #map}: !xegpu.tensor_desc<256x256xf32, #map> -> vector<256x256xf32>
      %input_tdesc_2 = xegpu.create_nd_tdesc %input2_gpu : memref<256x256xf32> -> !xegpu.tensor_desc<256x256xf32, #map>
      %input_val_2 = xegpu.load_nd %input_tdesc_2[%m, %n] {layout = #map}: !xegpu.tensor_desc<256x256xf32, #map> -> vector<256x256xf32>
      %result_val = arith.maximumf %input_val_1, %input_val_2 {layout_result_0 = #map} : vector<256x256xf32>
      %result_tdesc = xegpu.create_nd_tdesc %result_gpu : memref<256x256xf32> -> !xegpu.tensor_desc<256x256xf32, #map>
      xegpu.store_nd %result_val, %result_tdesc[%m, %n] {layout = #map}: vector<256x256xf32>, !xegpu.tensor_desc<256x256xf32, #map>
      gpu.return
    }
  }

  gpu.module @arith_maximumf_fastmath_module {
    // Kernel with fastmath attribute
    gpu.func @gpu_maximumf_with_fastmath(%input1_gpu : memref<256x256xf32>, %input2_gpu : memref<256x256xf32>, %result_gpu : memref<256x256xf32>) kernel  {
      %c256 = arith.constant 256 : index
      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y
      %m = arith.muli %block_id_x, %c256 : index
      %n = arith.muli %block_id_y, %c256 : index
      %input_tdesc_1 = xegpu.create_nd_tdesc %input1_gpu : memref<256x256xf32> -> !xegpu.tensor_desc<256x256xf32, #map>
      %input_val_1 = xegpu.load_nd %input_tdesc_1[%m, %n] {layout = #map}: !xegpu.tensor_desc<256x256xf32, #map> -> vector<256x256xf32>
      %input_tdesc_2 = xegpu.create_nd_tdesc %input2_gpu : memref<256x256xf32> -> !xegpu.tensor_desc<256x256xf32, #map>
      %input_val_2 = xegpu.load_nd %input_tdesc_2[%m, %n] {layout = #map}: !xegpu.tensor_desc<256x256xf32, #map> -> vector<256x256xf32>
      %result_val = arith.maximumf %input_val_1, %input_val_2 fastmath<fast> {layout_result_0 = #map} : vector<256x256xf32>
      %result_tdesc = xegpu.create_nd_tdesc %result_gpu : memref<256x256xf32> -> !xegpu.tensor_desc<256x256xf32, #map>
      xegpu.store_nd %result_val, %result_tdesc[%m, %n] {layout = #map}: vector<256x256xf32>, !xegpu.tensor_desc<256x256xf32, #map>
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

    %input_1 = memref.alloc() : memref<256x256xf32>
    %input_2 = memref.alloc() : memref<256x256xf32>
    %cpu_ref_result = memref.alloc() : memref<256x256xf32>

    // Initialize

    // Initialize to constant values
    // scf.for %arg0 = %c0 to %c256 step %c1 {
    //   scf.for %arg1 = %c0 to %c256 step %c1 {
    //     memref.store %c2_f32, %input_1[%arg0, %arg1] : memref<256x256xf32>
    //     memref.store %c2_f32, %input_2[%arg0, %arg1] : memref<256x256xf32>
    //   }
    // }

    // Initialize to random values
    %input_1_cast = memref.cast %input_1 : memref<256x256xf32> to memref<*xf32>
    %input_2_cast = memref.cast %input_2 : memref<256x256xf32> to memref<*xf32>
    call @fillResource1DRandomF32(%input_1_cast, %init_val_min_range, %init_val_max_range, %false) : (memref<*xf32>, f32, f32, i1) -> ()
    call @fillResource1DRandomF32(%input_2_cast, %init_val_min_range, %init_val_max_range, %false) : (memref<*xf32>, f32, f32, i1) -> ()

    // Run CPU version
    scf.for %arg0 = %c0 to %c256 step %c1 {
      scf.for %arg1 = %c0 to %c256 step %c1 {
        %val_1 = memref.load %input_1[%arg0, %arg1] : memref<256x256xf32>
        %val_2 = memref.load %input_2[%arg0, %arg1] : memref<256x256xf32>
        %res_val = arith.maximumf %val_1, %val_2 : f32
        memref.store %res_val, %cpu_ref_result[%arg0, %arg1] : memref<256x256xf32>
      }
    }

    // Run GPU version.
    %gpu_result, %gpu_result_fastmath = call @test_fast_math(%input_1, %input_2) : (memref<256x256xf32>, memref<256x256xf32>) -> (memref<256x256xf32>, memref<256x256xf32>)
    %gpu_result_cast = memref.cast %gpu_result : memref<256x256xf32> to memref<*xf32>
    %gpu_result_fastmath_cast = memref.cast %gpu_result_fastmath : memref<256x256xf32> to memref<*xf32>
    %cpu_ref_result_cast = memref.cast %cpu_ref_result : memref<256x256xf32> to memref<*xf32>

    call @printMaxErrorF32(%cpu_ref_result_cast, %gpu_result_cast) : (memref<*xf32>, memref<*xf32>) -> ()
    call @printMaxErrorF32(%cpu_ref_result_cast, %gpu_result_fastmath_cast) : (memref<*xf32>, memref<*xf32>) -> ()

    // call @printMemrefF32(%gpu_result_cast) : (memref<*xf32>) -> ()

    // CHECK: [ALLCLOSE: TRUE]
    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF32(%cpu_ref_result_cast, %gpu_result_cast) : (memref<*xf32>, memref<*xf32>) -> ()
    call @printAllcloseF32(%cpu_ref_result_cast, %gpu_result_fastmath_cast) : (memref<*xf32>, memref<*xf32>) -> ()

    memref.dealloc %input_1 : memref<256x256xf32>
    memref.dealloc %input_2 : memref<256x256xf32>
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
