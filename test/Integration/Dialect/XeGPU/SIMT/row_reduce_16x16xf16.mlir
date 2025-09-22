// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime,spirv-backend -i %s --pass-pipeline-file=%p/xegpu-to-llvm.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck
module @gemm attributes {gpu.container_module} {
  gpu.module @kernel {
    gpu.func @row_reduce(%in: memref<16x16xf16>, %c: memref<16x16xf16>) kernel attributes {intel_reqd_sub_group_size = 16 : i32} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c2_i32 = arith.constant 2 : i32
      %c4_i32 = arith.constant 4 : i32
      %c8_i32 = arith.constant 8 : i32
      %c16_i32 = arith.constant 16 : i32
      %c16 = arith.constant 16 : index
      %cst = arith.constant dense<1.0> : vector<16xf16>
      %in_tdesc = xegpu.create_nd_tdesc %in : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
      %c_tdesc = xegpu.create_nd_tdesc %c : memref<16x16xf16> -> !xegpu.tensor_desc<8x16xf16>
      %in_val = xegpu.load_nd %in_tdesc[%c0, %c0]  : !xegpu.tensor_desc<16x16xf16> -> vector<16xf16>
      %out0 = arith.constant dense<0.0> : vector<16xf16>
      // Do a cross-lane reduction for each row.
      %r = scf.for %i = %c0 to %c16 step %c1 iter_args(%arg0 = %out0) -> vector<16xf16> {
        %in_val0 = vector.extract %in_val [%i] : f16 from vector<16xf16>
        %shuffle1, %t0 = gpu.shuffle xor %in_val0, %c1_i32, %c16_i32 : f16
        %add1 = arith.addf %in_val0, %shuffle1 : f16
        %shuffle2, %t1 = gpu.shuffle xor %add1, %c2_i32, %c16_i32 : f16
        %add2 = arith.addf %add1, %shuffle2 : f16
        %shuffle4, %t2 = gpu.shuffle xor %add2, %c4_i32, %c16_i32 : f16
        %add4 = arith.addf %add2, %shuffle4 : f16
        %shuffle8, %t3 = gpu.shuffle xor %add4, %c8_i32, %c16_i32 : f16
        %add8 = arith.addf %add4, %shuffle8 : f16
        %r = vector.insert %add8, %arg0 [%i] : f16 into vector<16xf16>
        scf.yield %r : vector<16xf16>
      }
      // Store the result.
      %out_extract0 = vector.extract_strided_slice %r {offsets = [0], sizes = [8], strides = [1]} : vector<16xf16> to vector<8xf16>
      %out_extract1 = vector.extract_strided_slice %r {offsets = [8], sizes = [8], strides = [1]} : vector<16xf16> to vector<8xf16>
      xegpu.store_nd %out_extract0, %c_tdesc[%c0, %c0] : vector<8xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %out_extract1, %c_tdesc[%c8, %c0] : vector<8xf16>, !xegpu.tensor_desc<8x16xf16>
      gpu.return
    }
  }

  func.func @test(%in: memref<16x16xf16>, %c: memref<16x16xf16>) -> memref<16x16xf16> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %memref_in = gpu.alloc () : memref<16x16xf16>
    gpu.memcpy %memref_in, %in : memref<16x16xf16>, memref<16x16xf16>
    %memref_out = gpu.alloc () : memref<16x16xf16>
    gpu.memcpy %memref_out, %c : memref<16x16xf16>, memref<16x16xf16>
    gpu.launch_func  @kernel::@row_reduce blocks in (%c1, %c1, %c1) threads in (%c16, %c1, %c1) args(%memref_in : memref<16x16xf16>, %memref_out : memref<16x16xf16>)
    gpu.wait
    gpu.memcpy %c, %memref_out : memref<16x16xf16>, memref<16x16xf16>
    gpu.dealloc %memref_in : memref<16x16xf16>
    gpu.dealloc %memref_out : memref<16x16xf16>
    return %c : memref<16x16xf16>
  }


  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %in = memref.alloc() : memref<16x16xf16>
    %out = memref.alloc() : memref<16x16xf16>
    %out_host = memref.alloc() : memref<16x16xf32>
    // Fill input with random values
    %in_cast = memref.cast %in : memref<16x16xf16> to memref<*xf16>
    %lower = arith.constant 0.0 : f32
    %upper = arith.constant 5.0 : f32
    %gen_int = arith.constant 1 : i1
    call @fillResource1DRandomF16(%in_cast, %lower, %upper, %gen_int) : (memref<*xf16>, f32, f32, i1) -> ()

    // CPU version.
    %c0_f16 = arith.constant 0.0 : f16
    %cst = arith.constant dense<1.0> : vector<16xf16>
    scf.for %i = %c0 to %c16 step %c1  {
      %row = vector.transfer_read %in[%i, %c0], %c0_f16 : memref<16x16xf16>, vector<16xf16>
      %reduce = vector.reduction <add>, %row : vector<16xf16> into f16
      %broadcast = vector.broadcast %reduce : f16 to vector<16xf16>
      %broadcast_f32 = arith.extf %broadcast : vector<16xf16> to vector<16xf32>
      vector.transfer_write %broadcast_f32, %out_host[%i, %c0] : vector<16xf32>, memref<16x16xf32>
    }
    %out_host_cast = memref.cast %out_host : memref<16x16xf32> to memref<*xf32>
    // GPU version.
    %gpu_out = call @test(%in, %out) : (memref<16x16xf16>, memref<16x16xf16>) -> memref<16x16xf16>
    %gpu_out_cast = memref.cast %gpu_out : memref<16x16xf16> to memref<*xf16>

    // call @printMemrefF16(%gpu_out_cast) : (memref<*xf16>) -> ()
    // call @printMemrefF32(%out_host_cast) : (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF16(%gpu_out_cast, %out_host_cast) : (memref<*xf16>, memref<*xf32>) -> ()

    memref.dealloc %in : memref<16x16xf16>
    memref.dealloc %out : memref<16x16xf16>
    memref.dealloc %out_host : memref<16x16xf32>
    return
  }
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF16(memref<*xf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF16(memref<*xf16>, f32, f32, i1) attributes {llvm.emit_c_interface}
}
