// RUN: imex-opt %s --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup zebin-chip=pvc" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_levelzero_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --shared-libs=%irunner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @reduction {
    gpu.func @cross_sg_intra_lane_post_op_3D_1D(%arg0: memref<4x1024x1024xf32>, %arg1: memref<4x1024x1xf32>) kernel {
        %c28 = arith.constant 28 : index
        %block_id_x = gpu.block_id x
        %0 = arith.remsi %block_id_x, %c28 : index
        %1 = arith.divsi %block_id_x, %c28 : index
        %cst = arith.constant dense<1.024000e+03> : vector<2xf32>
        %cst_0 = arith.constant dense<1024> : vector<2xindex>
        %cst_1 = arith.constant dense<32> : vector<16xindex>
        %c1024_2 = arith.constant 1024 : index
        %c1048576 = arith.constant 1048576 : index
        %c512 = arith.constant 512 : index
        %c28_3 = arith.constant 28 : index
        %c1_4 = arith.constant 1 : index
        %c37 = arith.constant 37 : index
        %cst_5 = arith.constant dense<0.000000e+00> : vector<2x16x32xf32>
        %c0 = arith.constant 0 : index
        %cst_6 = arith.constant dense<0.000000e+00> : vector<2xf32>
        %c2 = arith.constant 2 : index
        %c14 = arith.constant 14 : index
        %block_id_x_7 = gpu.block_id x
        %block_id_y = gpu.block_id y
        %2 = arith.remsi %0, %c14 : index
        %3 = arith.divsi %0, %c14 : index
        %4 = arith.remsi %3, %c2 : index
        scf.for %arg14 = %c0 to %c37 step %c1_4 {
            %5 = scf.for %arg15 = %c0 to %c2 step %c1_4 iter_args(%arg16 = %cst_5) -> (vector<2x16x32xf32>) {
                %21 = arith.muli %1, %c2 overflow<nsw> : index
                %22 = arith.addi %21, %4 : index
                %23 = arith.muli %2, %c2 overflow<nsw> : index
                %24 = arith.muli %arg14, %c28_3 overflow<nsw> : index
                %25 = arith.addi %23, %24 : index
                %26 = vector.step : vector<2xindex>
                %27 = vector.broadcast %25 : index to vector<2xindex>
                %28 = arith.addi %27, %26 : vector<2xindex>
                %29 = arith.muli %arg15, %c512 overflow<nsw> : index
                %30 = vector.step : vector<16xindex>
                %31 = arith.muli %30, %cst_1 : vector<16xindex>
                %32 = vector.step : vector<32xindex>
                %33 = vector.broadcast %29 : index to vector<16xindex>
                %34 = arith.addi %33, %31 : vector<16xindex>
                %35 = vector.shape_cast %34 : vector<16xindex> to vector<1x16x1xindex>
                %36 = vector.broadcast %35 : vector<1x16x1xindex> to vector<2x16x32xindex>
                %37 = vector.broadcast %32 : vector<32xindex> to vector<2x16x32xindex>
                %38 = arith.addi %36, %37 : vector<2x16x32xindex>
                %39 = arith.muli %22, %c1048576 : index
                %40 = vector.broadcast %39 : index to vector<2xindex>
                %41 = arith.muli %28, %cst_0 : vector<2xindex>
                %42 = arith.addi %40, %41 : vector<2xindex>
                %43 = vector.shape_cast %42 : vector<2xindex> to vector<2x1x1xindex>
                %44 = vector.broadcast %43 : vector<2x1x1xindex> to vector<2x16x32xindex>
                %45 = arith.addi %44, %38 : vector<2x16x32xindex>
                %46 = arith.subi %c1024_2, %25 : index
                %47 = vector.create_mask %46 : vector<2xi1>
                %48 = vector.shape_cast %47 : vector<2xi1> to vector<2x1x1xi1>
                %49 = vector.broadcast %48 : vector<2x1x1xi1> to vector<2x16x32xi1>
                %intptr_8 = memref.extract_aligned_pointer_as_index %arg0 : memref<4x1024x1024xf32> -> index
                %50 = arith.index_cast %intptr_8 : index to i64
                %51 = xegpu.load %50[%45], %49 <{layout = #xegpu.layout<sg_layout = [2, 16, 1], sg_data = [1, 1, 32]>}> : i64, vector<2x16x32xindex>, vector<2x16x32xi1> -> vector<2x16x32xf32>
                %52 = arith.select %49, %51, %cst_5 : vector<2x16x32xi1>, vector<2x16x32xf32>
                %53 = arith.addf %arg16, %52 : vector<2x16x32xf32>
                scf.yield %53 : vector<2x16x32xf32>
            }
            %6 = vector.multi_reduction <add>, %5, %cst_6 [1, 2] : vector<2x16x32xf32> to vector<2xf32>
            %7 = arith.divf %6, %cst : vector<2xf32>
            %8 = arith.muli %1, %c2 overflow<nsw> : index
            %9 = arith.addi %8, %4 : index
            %10 = arith.muli %2, %c2 overflow<nsw> : index
            %11 = arith.muli %arg14, %c28_3 overflow<nsw> : index
            %12 = arith.addi %10, %11 : index
            %13 = vector.step : vector<2xindex>
            %14 = arith.muli %9, %c1024_2 : index
            %15 = arith.addi %14, %12 : index
            %16 = vector.broadcast %15 : index to vector<2xindex>
            %17 = arith.addi %16, %13 : vector<2xindex>
            %18 = arith.subi %c1024_2, %12 : index
            %19 = vector.create_mask %18 : vector<2xi1>
            %intptr = memref.extract_aligned_pointer_as_index %arg1 : memref<4x1024x1xf32> -> index
            %20 = arith.index_cast %intptr : index to i64
            xegpu.store %7, %20[%17], %19 <{layout = #xegpu.slice<#xegpu.layout<sg_layout = [2, 16, 1], sg_data = [1, 1, 32]>, dims = [1, 2]>}> {layout_operand_0 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 16, 1], sg_data = [1, 1, 32]>, dims = [1, 2]>, layout_operand_2 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 16, 1], sg_data = [1, 1, 32]>, dims = [1, 2]>, layout_operand_3 = #xegpu.slice<#xegpu.layout<sg_layout = [2, 16, 1], sg_data = [1, 1, 32]>, dims = [1, 2]>} : vector<2xf32>, i64, vector<2xindex>, vector<2xi1>
        }
        gpu.return
    }
  }

  func.func @test(%src : memref<4x1024x1024xf32>, %dst : memref<4x1024x1xf32>) attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index

    %stream0_0 = gpu.wait async

    %gpu_memref_src, %stream0_1 = gpu.alloc async [%stream0_0] () : memref<4x1024x1024xf32>
    %stream0_2 = gpu.memcpy async [%stream0_1] %gpu_memref_src, %src  : memref<4x1024x1024xf32>, memref<4x1024x1024xf32>

    %gpu_memref_dst, %stream0_3 = gpu.alloc async [%stream0_2] () : memref<4x1024x1xf32>
    %stream0_4 = gpu.memcpy async [%stream0_3] %gpu_memref_dst, %dst  : memref<4x1024x1xf32>, memref<4x1024x1xf32>

    %stream0_5 = gpu.launch_func async[%stream0_4] @reduction::@cross_sg_intra_lane_post_op_3D_1D blocks in (%c1, %c1, %c1) threads in (%c1024, %c1, %c1) args(%gpu_memref_src : memref<4x1024x1024xf32>, %gpu_memref_dst : memref<4x1024x1xf32>)

    %stream0_6 = gpu.memcpy async [%stream0_5]  %dst, %gpu_memref_dst : memref<4x1024x1xf32>, memref<4x1024x1xf32>
    %stream0_7 = gpu.dealloc async [%stream0_6] %gpu_memref_src : memref<4x1024x1024xf32>
    %stream0_8 = gpu.dealloc async [%stream0_7] %gpu_memref_dst : memref<4x1024x1xf32>
    gpu.wait [%stream0_8]
    return
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %src = memref.alloc() : memref<4x1024x1024xf32>
    %dst = memref.alloc() : memref<4x1024x1xf32>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c1024 = arith.constant 1024 : index

    %c0_f32 = arith.constant 0. : f32
    %c1_f32 = arith.constant 1. : f32

    // Initialize source with value 1.0 for all elements (4*1024*1024 = 4,194,304 elements)
    scf.for %i = %c0 to %c4 step %c1 {
      scf.for %j = %c0 to %c1024 step %c1 {
        scf.for %k = %c0 to %c1024 step %c1 {
          memref.store %c1_f32, %src[%i, %j, %k] : memref<4x1024x1024xf32>
        }
      }
    }

    // Initialize destination with 0
    scf.for %i = %c0 to %c4 step %c1 {
      scf.for %j = %c0 to %c1024 step %c1 {
        memref.store %c0_f32, %dst[%i, %j, %c0] : memref<4x1024x1xf32>
      }
    }

    call @test(%src, %dst) : (memref<4x1024x1024xf32>, memref<4x1024x1xf32>) -> ()

    %dst_cast = memref.cast %dst : memref<4x1024x1xf32> to memref<*xf32>

    // CHECK: Unranked Memref base@ = 0x{{[0-9a-f]+}}
    // CHECK: [1024
    call @printMemrefF32(%dst_cast) : (memref<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF32(%ptr : memref<*xf32>) attributes { llvm.emit_c_interface }
}
