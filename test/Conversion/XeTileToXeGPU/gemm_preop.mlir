// RUN: imex-opt --split-input-file --xetile-init-duplicate --xetile-blocking="enable-2d-transform=true" --cse \
// RUN: --convert-xetile-to-xegpu --cse %s -verify-diagnostics -o -| FileCheck %s
#map = affine_map<() -> (0)>
#map1 = affine_map<() -> (64)>
module attributes {gpu.container_module} {
  func.func @preop_m512_n256_k64_entry(%arg0: memref<512x64xf16>, %arg1: memref<256x64xf16>, %arg2: memref<512x256xf32>, %arg3: memref<512x256xf32>) attributes {gemm_tiles_x = dense<[2, 1, 2, 4]> : vector<4xi64>, gemm_tiles_y = dense<[1, 1, 1, 8]> : vector<4xi64>, habana_runner.num_inputs = 3 : i64, habana_runner.tests = [{inputs = [dense<1.000000e+00> : tensor<512x64xf16>, dense<1.000000e+00> : tensor<256x64xf16>, dense<1.900000e+01> : tensor<512x256xf32>], outputs = [dense<2.750000e+02> : tensor<512x256xf32>]}], physical_nd_range = dense<2> : vector<2xi64>, syn.fusion_successful, syn.tensor_signature = (tensor<512x64xf16>, tensor<256x64xf16>, tensor<512x256xf32>) -> tensor<512x256xf32>, synFusionGenOps = 6 : i64, synFusionRequiredBeamSize = 1 : i64, synFusionTotalCost = 1000015321.24 : f64} {
    %c2 = arith.constant 2 : index
    %c2_0 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    gpu.launch_func  @preop_m512_n256_k64::@preop_m512_n256_k64 blocks in (%c2, %c2_0, %c1) threads in (%c4, %c8, %c1)  args(%arg0 : memref<512x64xf16>, %arg1 : memref<256x64xf16>, %arg2 : memref<512x256xf32>, %arg3 : memref<512x256xf32>)
    return
  }
  gpu.module @preop_m512_n256_k64 attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Bfloat16ConversionINTEL, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, VectorAnyINTEL], [SPV_INTEL_bfloat16_conversion, SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @preop_m512_n256_k64(%arg0: memref<512x64xf16>, %arg1: memref<256x64xf16>, %arg2: memref<512x256xf32>, %arg3: memref<512x256xf32>) kernel attributes {gpu.known_block_size = array<i32: 4, 8, 1>, gpu.known_grid_size = array<i32: 2, 2, 1>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %thread_id_z = gpu.thread_id  z
      %grid_dim_x = gpu.grid_dim  x
      %grid_dim_y = gpu.grid_dim  y
      %grid_dim_z = gpu.grid_dim  z
      %block_dim_x = gpu.block_dim  x
      %block_dim_y = gpu.block_dim  y
      %block_dim_z = gpu.block_dim  z
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %cst = arith.constant dense<0.000000e+00> : vector<32x32xf32>
      %c256 = arith.constant 256 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %thread_id_x_0 = gpu.thread_id  x
      %thread_id_y_1 = gpu.thread_id  y
      %block_dim_y_2 = gpu.block_dim  y
      %0 = arith.muli %thread_id_x_0, %block_dim_y_2 : index
      %1 = arith.addi %0, %thread_id_y_1 : index
      %block_id_x_3 = gpu.block_id  x
      %block_id_y_4 = gpu.block_id  y
      %2 = arith.divsi %1, %c8 : index
      %3 = arith.remsi %1, %c8 : index
      %4 = arith.muli %2, %c32 : index
      %5 = arith.remsi %4, %c128 : index
      %6 = arith.muli %3, %c64 : index
      %7 = arith.remsi %6, %c64 : index
      %8 = arith.muli %block_id_x_3, %c256 : index
      %9 = arith.muli %block_id_y_4, %c128 : index
      %10 = arith.addi %8, %9 : index
      %11 = arith.addi %10, %5 : index
      %12 = arith.muli %3, %c32 : index
      %13 = arith.remsi %12, %c256 : index
      %14 = arith.muli %2, %c64 : index
      %15 = arith.remsi %14, %c64 : index
      %16 = xetile.init_tile %arg0[%11, %7] : memref<512x64xf16> -> !xetile.tile<32x32xf16>
      %17 = xetile.init_tile %arg1[%13, %15] : memref<256x64xf16> -> !xetile.tile<32x32xf16>
      %18:3 = scf.for %arg4 = %c0 to %c64 step %c32 iter_args(%arg5 = %16, %arg6 = %17, %arg7 = %cst) -> (!xetile.tile<32x32xf16>, !xetile.tile<32x32xf16>, vector<32x32xf32>) {
        %27 = xetile.load_tile %arg5 {padding = 0.000000e+00 : f32}  : !xetile.tile<32x32xf16> -> vector<32x32xf16>
        %28 = xetile.load_tile %arg6 {padding = 0.000000e+00 : f32}  : !xetile.tile<32x32xf16> -> vector<32x32xf16>
        xegpu.compile_hint
        //CHECK-COUNT-4: {{.*}} = vector.extract_strided_slice {{.*}} {offsets = [{{.*}}], sizes = [16, 16], strides = [1, 1]} : vector<32x16xf16> to vector<16x16xf16>
        //CHECK-COUNT-8: {{.*}} = arith.addf {{.*}}, {{.*}} : vector<8x16xf16>
        %29 = arith.addf %27, %27 : vector<32x32xf16>
        xegpu.compile_hint
        %30 = xetile.update_tile_offset %arg5, [%c0,  %c32] : !xetile.tile<32x32xf16>
        %31 = xetile.update_tile_offset %arg6, [%c0,  %c32] : !xetile.tile<32x32xf16>
        xegpu.compile_hint

        // CHECK-COUNT-16: {{.*}} = xegpu.dpas {{.*}}, {{.*}}, {{.*}} : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %32 = xetile.tile_mma %29, %28, %arg7 : vector<32x32xf16>, vector<32x32xf16>, vector<32x32xf32> -> vector<32x32xf32>
        xegpu.compile_hint
        scf.yield %30, %31, %32 : !xetile.tile<32x32xf16>, !xetile.tile<32x32xf16>, vector<32x32xf32>
      } {lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 3>, step = 32 : index, upperBoundMap = #map1}
      %19 = xetile.init_tile %arg2[%11, %13] : memref<512x256xf32> -> !xetile.tile<32x32xf32>
      %20 = xetile.load_tile %19 {padding = 0.000000e+00 : f32}  : !xetile.tile<32x32xf32> -> vector<32x32xf32>
      %21 = arith.addf %18#2, %20 : vector<32x32xf32>
      %22 = arith.muli %block_id_x_3, %c256 : index
      %23 = arith.muli %block_id_y_4, %c128 : index
      %24 = arith.addi %22, %23 : index
      %25 = arith.addi %24, %5 : index
      %26 = xetile.init_tile %arg3[%25, %13] : memref<512x256xf32> -> !xetile.tile<32x32xf32>
      xetile.store_tile %21,  %26 : vector<32x32xf32>, !xetile.tile<32x32xf32>
      gpu.return
    }
  }
}
