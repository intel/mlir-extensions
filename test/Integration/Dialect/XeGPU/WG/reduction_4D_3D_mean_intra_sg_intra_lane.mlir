// RUN: imex-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup zebin-chip=pvc" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_levelzero_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --shared-libs=%irunner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

module attributes {gpu.container_module} {

  gpu.module @reduce [#xevm.target<O = 3, chip = "pvc">] {
    gpu.func @mean(%arg0: memref<1x24x1024x64xf32>, %arg1: memref<1x24x1024x1xf32>) kernel attributes {known_block_size = array<i32: 128, 1, 1>} {
      %cst = arith.constant dense<true> : vector<1x1x16x1xi1>
      %cst_0 = arith.constant dense<1024> : vector<1xindex>
      %cst_1 = arith.constant dense<24576> : vector<1xindex>
      %c1024 = arith.constant 1024 : index
      %cst_2 = arith.constant dense<true> : vector<1x1x16x64xi1>
      %cst_3 = arith.constant dense<64> : vector<16xindex>
      %cst_4 = arith.constant dense<65536> : vector<1xindex>
      %cst_5 = arith.constant dense<1572864> : vector<1xindex>
      %c64 = arith.constant 64 : index
      %c65536 = arith.constant 65536 : index
      %cst_6 = arith.constant dense<6.400000e+01> : vector<1x1x16xf32>
      %cst_7 = arith.constant dense<0.000000e+00> : vector<1x1x16xf32>
      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y
      %0 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%block_id_y]
      %1 = vector.step : vector<1xindex>
      %2 = vector.step : vector<16xindex>
      %3 = vector.step : vector<64xindex>
      %4 = arith.muli %1, %cst_5 : vector<1xindex>
      %5 = arith.muli %1, %cst_4 : vector<1xindex>
      %6 = arith.muli %2, %cst_3 : vector<16xindex>
      %7 = vector.shape_cast %6 : vector<16xindex> to vector<1x1x16x1xindex>
      %8 = vector.broadcast %4 : vector<1xindex> to vector<1x1x16x64xindex>
      %9 = vector.broadcast %5 : vector<1xindex> to vector<1x1x16x64xindex>
      %10 = vector.broadcast %7 : vector<1x1x16x1xindex> to vector<1x1x16x64xindex>
      %11 = vector.broadcast %3 : vector<64xindex> to vector<1x1x16x64xindex>
      %12 = arith.addi %8, %9 : vector<1x1x16x64xindex>
      %13 = arith.addi %12, %10 : vector<1x1x16x64xindex>
      %14 = arith.addi %13, %11 : vector<1x1x16x64xindex>
      %15 = arith.muli %block_id_x, %c65536 : index
      %16 = arith.muli %0, %c64 : index
      %17 = arith.addi %15, %16 : index
      %18 = vector.broadcast %17 : index to vector<1x1x16x64xindex>
      %19 = arith.addi %18, %14 : vector<1x1x16x64xindex>
      %intptr = memref.extract_aligned_pointer_as_index %arg0 : memref<1x24x1024x64xf32> -> index
      %20 = arith.index_cast %intptr : index to i64
      %21 = xegpu.load %20[%19], %cst_2  : i64, vector<1x1x16x64xindex>, vector<1x1x16x64xi1> -> vector<1x1x16x64xf32>
      %22 = vector.multi_reduction <add>, %21, %cst_7 [3] : vector<1x1x16x64xf32> to vector<1x1x16xf32>
      %23 = arith.divf %22, %cst_6 : vector<1x1x16xf32>
      %24 = vector.shape_cast %23 : vector<1x1x16xf32> to vector<1x1x16x1xf32>
      %25 = arith.muli %1, %cst_1 : vector<1xindex>
      %26 = arith.muli %1, %cst_0 : vector<1xindex>
      %27 = vector.shape_cast %2 : vector<16xindex> to vector<1x1x16x1xindex>
      %28 = vector.broadcast %25 : vector<1xindex> to vector<1x1x16x1xindex>
      %29 = vector.broadcast %26 : vector<1xindex> to vector<1x1x16x1xindex>
      %30 = vector.broadcast %1 : vector<1xindex> to vector<1x1x16x1xindex>
      %31 = arith.addi %28, %29 : vector<1x1x16x1xindex>
      %32 = arith.addi %31, %27 : vector<1x1x16x1xindex>
      %33 = arith.addi %32, %30 : vector<1x1x16x1xindex>
      %34 = arith.muli %block_id_x, %c1024 : index
      %35 = arith.addi %34, %0 : index
      %36 = vector.broadcast %35 : index to vector<1x1x16x1xindex>
      %37 = arith.addi %36, %33 : vector<1x1x16x1xindex>
      %intptr_8 = memref.extract_aligned_pointer_as_index %arg1 : memref<1x24x1024x1xf32> -> index
      %38 = arith.index_cast %intptr_8 : index to i64
      xegpu.store %24, %38[%37], %cst {layout = #xegpu.layout<sg_layout = [1, 1, 1, 1], sg_data = [1, 1, 16, 1]>} : vector<1x1x16x1xf32>, i64, vector<1x1x16x1xindex>, vector<1x1x16x1xi1>
      gpu.return
    }
  }

  func.func @test(%src: memref<1x24x1024x64xf32>, %dst: memref<1x24x1024x1xf32>) attributes {llvm.emit_c_interface} {
    %s0 = gpu.wait async
    %gpu_src, %s1 = gpu.alloc async [%s0] () : memref<1x24x1024x64xf32>
    %gpu_dst, %s2 = gpu.alloc async [%s1] () : memref<1x24x1024x1xf32>
    %s3 = gpu.memcpy async [%s2] %gpu_src, %src  : memref<1x24x1024x64xf32>, memref<1x24x1024x64xf32>


    %c1 = arith.constant 1 : index
    %c24 = arith.constant 24 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %s4 = gpu.launch_func async[%s3]  @reduce::@mean blocks in (%c24, %c64, %c1) threads in (%c128, %c1, %c1)  args(%gpu_src : memref<1x24x1024x64xf32>, %gpu_dst : memref<1x24x1024x1xf32>)

    %s5 = gpu.dealloc async [%s4] %gpu_src : memref<1x24x1024x64xf32>
    %s6 = gpu.memcpy async [%s5] %dst, %gpu_dst  : memref<1x24x1024x1xf32>, memref<1x24x1024x1xf32>
    %s7 = gpu.dealloc async [%s6] %gpu_dst : memref<1x24x1024x1xf32>
    gpu.wait [%s7]
    return
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c24 = arith.constant 24 : index
    %c64 = arith.constant 64 : index
    %c1024 = arith.constant 1024 : index
    %f0 = arith.constant 0.0 : f32
    %f64 = arith.constant 64.0 : f32
    %f_min = arith.constant -1.0 : f32
    %f_max = arith.constant 1.0 : f32
    %false = arith.constant false

    %src = memref.alloc() : memref<1x24x1024x64xf32>
    %dst = memref.alloc() : memref<1x24x1024x1xf32>

    %src_cast = memref.cast %src : memref<1x24x1024x64xf32> to memref<*xf32>
    call @fillResource1DRandomF32(%src_cast, %f_min, %f_max, %false) : (memref<*xf32>, f32, f32, i1) -> ()

    call @test(%src, %dst) : (memref<1x24x1024x64xf32>, memref<1x24x1024x1xf32>) -> ()

    // Compute reference: mean along last dim (size 64)
    %ref = memref.alloc() : memref<1x24x1024x1xf32>
    scf.for %b = %c0 to %c1 step %c1 {
      scf.for %h = %c0 to %c24 step %c1 {
        scf.for %s = %c0 to %c1024 step %c1 {
          %sum = scf.for %k = %c0 to %c64 step %c1 iter_args(%acc = %f0) -> f32 {
            %v = memref.load %src[%b, %h, %s, %k] : memref<1x24x1024x64xf32>
            %new_acc = arith.addf %acc, %v : f32
            scf.yield %new_acc : f32
          }
          %mean = arith.divf %sum, %f64 : f32
          memref.store %mean, %ref[%b, %h, %s, %c0] : memref<1x24x1024x1xf32>
        }
      }
    }
    memref.dealloc %src : memref<1x24x1024x64xf32>

    %dst_cast = memref.cast %dst : memref<1x24x1024x1xf32> to memref<*xf32>
    %ref_cast = memref.cast %ref : memref<1x24x1024x1xf32> to memref<*xf32>

    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF32(%ref_cast, %dst_cast) : (memref<*xf32>, memref<*xf32>) -> ()

    memref.dealloc %dst : memref<1x24x1024x1xf32>
    memref.dealloc %ref : memref<1x24x1024x1xf32>
    return
  }

  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMaxErrorF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF32(memref<*xf32>, f32, f32, i1) attributes {llvm.emit_c_interface}
}
