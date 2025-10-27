// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xetile-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

// NOTES :
// This example assumes one subgroup per one workgroup and the kernel specifies the computation
// done by a single subgroup.
module @transpose attributes {gpu.container_module} {
  func.func @transpose_test(%arg0: memref<1024x1024xf16>) -> memref<1024x1024xf16> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %memref = gpu.alloc  () : memref<1024x1024xf16>
    gpu.memcpy  %memref, %arg0 : memref<1024x1024xf16>, memref<1024x1024xf16>
    %memref_0 = gpu.alloc  () : memref<1024x1024xf16>
    gpu.launch_func  @transpose_kernel::@transpose_kernel blocks in (%c64, %c32, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<1024x1024xf16>, %memref_0 : memref<1024x1024xf16>)
    gpu.dealloc  %memref : memref<1024x1024xf16>
    %alloc = memref.alloc() : memref<1024x1024xf16>
    gpu.memcpy  %alloc, %memref_0 : memref<1024x1024xf16>, memref<1024x1024xf16>
    gpu.dealloc  %memref_0 : memref<1024x1024xf16>
    return %alloc : memref<1024x1024xf16>
  }
  gpu.module @transpose_kernel  {
    gpu.func @transpose_kernel(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c1024 = arith.constant 1024 : index
      // initalize A and B tiles
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = arith.muli %block_id_x, %c16 : index
      %1 = arith.muli %block_id_y, %c32 : index
      %2 = xetile.init_tile %arg0[%0, %1] : memref<1024x1024xf16> -> !xetile.tile<16x32xf16>
      %3 = xetile.load_tile %2 : !xetile.tile<16x32xf16> -> vector<16x32xf16>
      %4 = vector.transpose %3, [1, 0] : vector<16x32xf16> to vector<32x16xf16>
      %5 = xetile.init_tile %arg1[%1, %0] : memref<1024x1024xf16> -> !xetile.tile<32x16xf16>
      xetile.store_tile %4,  %5 : vector<32x16xf16>, !xetile.tile<32x16xf16>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    // intialize matrix A; A[i, j] = j; B[i, j] = i
    %alloc = memref.alloc() : memref<1024x1024xf16>
    %alloc_0 = memref.alloc() : memref<1024x1024xf32>
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %1 = index.castu %arg1 : index to i16
        %2 = arith.uitofp %1 : i16 to f16
        memref.store %2, %alloc[%arg0, %arg1] : memref<1024x1024xf16>
        %3 = arith.extf %2 : f16 to f32
        memref.store %3, %alloc_0[%arg1, %arg0] : memref<1024x1024xf32>
      }
    }
    // CHECK: [ALLCLOSE: TRUE]
    %0 = call @transpose_test(%alloc) : (memref<1024x1024xf16>) -> memref<1024x1024xf16>
    %cast = memref.cast %0 : memref<1024x1024xf16> to memref<*xf16>
    %cast_1 = memref.cast %alloc_0 : memref<1024x1024xf32> to memref<*xf32>
    call @printAllcloseF16(%cast, %cast_1) : (memref<*xf16>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<1024x1024xf16>
    memref.dealloc %alloc_0 : memref<1024x1024xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF16(memref<*xf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
}

