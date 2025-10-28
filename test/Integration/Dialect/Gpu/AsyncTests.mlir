// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/gpu-to-llvm.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @eltwise_add attributes {gpu.container_module} {
  func.func @fillRandom(%arg0: memref<4194304xf32>, %arg1: memref<4194304xf32>) {
    %cst = arith.constant 1.000000e+01 : f32
    %cst_0 = arith.constant 5.000000e+01 : f32
    %false = arith.constant false
    %cast = memref.cast %arg0 : memref<4194304xf32> to memref<*xf32>
    call @fillResource1DRandomF32(%cast, %cst, %cst_0, %false) : (memref<*xf32>, f32, f32, i1) -> ()
    memref.copy %arg0, %arg1 : memref<4194304xf32> to memref<4194304xf32>
    return
  }
  func.func @fillZeros(%arg0: memref<4194304xf32>, %arg1: memref<4194304xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %cast = memref.cast %arg0 : memref<4194304xf32> to memref<*xf32>
    call @fillResource1DF32(%cast, %cst) : (memref<*xf32>, f32) -> ()
    memref.copy %arg0, %arg1 : memref<4194304xf32> to memref<4194304xf32>
    return
  }
  gpu.module @eltwiseAdd_kernel {
    gpu.func @eltwiseAdd_kernel(%arg0: memref<4194304xf32>, %arg1: memref<4194304xf32>, %arg2: memref<4194304xf32>) kernel attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %global_id_x = gpu.global_id  x
      %cst = arith.constant 5.000000e-01 : f32
      %0 = memref.load %arg0[%global_id_x] : memref<4194304xf32>
      %1 = memref.load %arg1[%global_id_x] : memref<4194304xf32>
      %2 = arith.addf %0, %1 : f32
      %3 = arith.addf %2, %cst : f32
      memref.store %3, %arg2[%global_id_x] : memref<4194304xf32>
      gpu.return
    }
  }
  // compute CPU reference (takes minutes)
  func.func @cpu_reference(%arg0: memref<4194304xf32>, %arg1: memref<4194304xf32>, %arg2: memref<4194304xf32>) {
    %c4194304 = arith.constant 4194304 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 5.000000e-01 : f32
    scf.for %arg3 = %c0 to %c4194304 step %c1 {
      %0 = memref.load %arg0[%arg3] : memref<4194304xf32>
      %1 = memref.load %arg1[%arg3] : memref<4194304xf32>
      %2 = arith.addf %0, %1 : f32
      %3 = arith.addf %2, %cst : f32
      memref.store %3, %arg2[%arg3] : memref<4194304xf32>
    }
    return
  }
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c8192 = arith.constant 8192 : index
    // Test1: Two async launches followed by sync launch that
    //        waits for events returned by the two async launches
    // CHECK: [ALLCLOSE: TRUE]
    // Test2: An async launch followed by another async launch and
    //        finally a sync launch. Each launch waits on the event
    //        from the preceeding launch.
    // CHECK: [ALLCLOSE: TRUE]
    // Test3: An async launch followed by two async launches and
    //        finally a sync launch. The event from the first async launch
    //        is passed to the subsequent two async launches which wait on
    //        the same event. The last sync launch waits from two events
    //        from the preceeding two async launches.
    // CHECK: [ALLCLOSE: TRUE]
    %alloc = memref.alloc() : memref<4194304xf32>
    %memref = gpu.alloc  () : memref<4194304xf32>
    call @fillRandom(%alloc, %memref) : (memref<4194304xf32>, memref<4194304xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<4194304xf32>
    %memref_1 = gpu.alloc  () : memref<4194304xf32>
    call @fillRandom(%alloc_0, %memref_1) : (memref<4194304xf32>, memref<4194304xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<4194304xf32>
    %memref_3 = gpu.alloc  () : memref<4194304xf32>
    call @fillRandom(%alloc_2, %memref_3) : (memref<4194304xf32>, memref<4194304xf32>) -> ()
    %alloc_4 = memref.alloc() : memref<4194304xf32>
    %memref_5 = gpu.alloc  () : memref<4194304xf32>
    call @fillRandom(%alloc_4, %memref_5) : (memref<4194304xf32>, memref<4194304xf32>) -> ()
    %alloc_6 = memref.alloc() : memref<4194304xf32>
    %memref_7 = gpu.alloc  () : memref<4194304xf32>
    call @fillZeros(%alloc_6, %memref_7) : (memref<4194304xf32>, memref<4194304xf32>) -> ()
    %alloc_8 = memref.alloc() : memref<4194304xf32>
    %memref_9 = gpu.alloc  () : memref<4194304xf32>
    call @fillZeros(%alloc_8, %memref_9) : (memref<4194304xf32>, memref<4194304xf32>) -> ()
    %alloc_10 = memref.alloc() : memref<4194304xf32>
    %memref_11 = gpu.alloc  () : memref<4194304xf32>
    call @fillZeros(%alloc_10, %memref_11) : (memref<4194304xf32>, memref<4194304xf32>) -> ()
    %alloc_12 = memref.alloc() : memref<4194304xf32>
    %memref_13 = gpu.alloc  () : memref<4194304xf32>
    call @fillZeros(%alloc_12, %memref_13) : (memref<4194304xf32>, memref<4194304xf32>) -> ()
    %0 = gpu.launch_func async @eltwiseAdd_kernel::@eltwiseAdd_kernel blocks in (%c8192, %c1, %c1) threads in (%c512, %c1, %c1)  args(%memref : memref<4194304xf32>, %memref_1 : memref<4194304xf32>, %memref_7 : memref<4194304xf32>)
    %1 = gpu.launch_func async @eltwiseAdd_kernel::@eltwiseAdd_kernel blocks in (%c8192, %c1, %c1) threads in (%c512, %c1, %c1)  args(%memref_3 : memref<4194304xf32>, %memref_5 : memref<4194304xf32>, %memref_9 : memref<4194304xf32>)
    gpu.launch_func [%0, %1] @eltwiseAdd_kernel::@eltwiseAdd_kernel blocks in (%c8192, %c1, %c1) threads in (%c512, %c1, %c1)  args(%memref_7 : memref<4194304xf32>, %memref_9 : memref<4194304xf32>, %memref_13 : memref<4194304xf32>)
    call @cpu_reference(%alloc, %alloc_0, %alloc_6) : (memref<4194304xf32>, memref<4194304xf32>, memref<4194304xf32>) -> ()
    call @cpu_reference(%alloc_2, %alloc_4, %alloc_8) : (memref<4194304xf32>, memref<4194304xf32>, memref<4194304xf32>) -> ()
    call @cpu_reference(%alloc_6, %alloc_8, %alloc_12) : (memref<4194304xf32>, memref<4194304xf32>, memref<4194304xf32>) -> ()
    %cast = memref.cast %alloc_12 : memref<4194304xf32> to memref<*xf32>
    %cast_14 = memref.cast %memref_13 : memref<4194304xf32> to memref<*xf32>
    call @printAllcloseF32(%cast, %cast_14) : (memref<*xf32>, memref<*xf32>) -> ()
    call @fillZeros(%alloc_6, %memref_7) : (memref<4194304xf32>, memref<4194304xf32>) -> ()
    call @fillZeros(%alloc_8, %memref_9) : (memref<4194304xf32>, memref<4194304xf32>) -> ()
    call @fillZeros(%alloc_12, %memref_13) : (memref<4194304xf32>, memref<4194304xf32>) -> ()
    %2 = gpu.launch_func async @eltwiseAdd_kernel::@eltwiseAdd_kernel blocks in (%c8192, %c1, %c1) threads in (%c512, %c1, %c1)  args(%memref : memref<4194304xf32>, %memref_1 : memref<4194304xf32>, %memref_7 : memref<4194304xf32>)
    %3 = gpu.launch_func async [%2] @eltwiseAdd_kernel::@eltwiseAdd_kernel blocks in (%c8192, %c1, %c1) threads in (%c512, %c1, %c1)  args(%memref_3 : memref<4194304xf32>, %memref_7 : memref<4194304xf32>, %memref_9 : memref<4194304xf32>)
    gpu.launch_func [%3] @eltwiseAdd_kernel::@eltwiseAdd_kernel blocks in (%c8192, %c1, %c1) threads in (%c512, %c1, %c1)  args(%memref_5 : memref<4194304xf32>, %memref_9 : memref<4194304xf32>, %memref_13 : memref<4194304xf32>)
    call @cpu_reference(%alloc, %alloc_0, %alloc_6) : (memref<4194304xf32>, memref<4194304xf32>, memref<4194304xf32>) -> ()
    call @cpu_reference(%alloc_2, %alloc_6, %alloc_8) : (memref<4194304xf32>, memref<4194304xf32>, memref<4194304xf32>) -> ()
    call @cpu_reference(%alloc_4, %alloc_8, %alloc_12) : (memref<4194304xf32>, memref<4194304xf32>, memref<4194304xf32>) -> ()
    %cast_15 = memref.cast %alloc_12 : memref<4194304xf32> to memref<*xf32>
    %cast_16 = memref.cast %memref_13 : memref<4194304xf32> to memref<*xf32>
    call @printAllcloseF32(%cast_15, %cast_16) : (memref<*xf32>, memref<*xf32>) -> ()
    call @fillZeros(%alloc_6, %memref_7) : (memref<4194304xf32>, memref<4194304xf32>) -> ()
    call @fillZeros(%alloc_8, %memref_9) : (memref<4194304xf32>, memref<4194304xf32>) -> ()
    call @fillZeros(%alloc_12, %memref_13) : (memref<4194304xf32>, memref<4194304xf32>) -> ()
    %4 = gpu.launch_func async @eltwiseAdd_kernel::@eltwiseAdd_kernel blocks in (%c8192, %c1, %c1) threads in (%c512, %c1, %c1)  args(%memref : memref<4194304xf32>, %memref_1 : memref<4194304xf32>, %memref_7 : memref<4194304xf32>)
    %5 = gpu.launch_func async [%4] @eltwiseAdd_kernel::@eltwiseAdd_kernel blocks in (%c8192, %c1, %c1) threads in (%c512, %c1, %c1)  args(%memref_3 : memref<4194304xf32>, %memref_7 : memref<4194304xf32>, %memref_9 : memref<4194304xf32>)
    %6 = gpu.launch_func async [%4] @eltwiseAdd_kernel::@eltwiseAdd_kernel blocks in (%c8192, %c1, %c1) threads in (%c512, %c1, %c1)  args(%memref_5 : memref<4194304xf32>, %memref_7 : memref<4194304xf32>, %memref_11 : memref<4194304xf32>)
    gpu.launch_func [%5, %6] @eltwiseAdd_kernel::@eltwiseAdd_kernel blocks in (%c8192, %c1, %c1) threads in (%c512, %c1, %c1)  args(%memref_9 : memref<4194304xf32>, %memref_11 : memref<4194304xf32>, %memref_13 : memref<4194304xf32>)
    call @cpu_reference(%alloc, %alloc_0, %alloc_6) : (memref<4194304xf32>, memref<4194304xf32>, memref<4194304xf32>) -> ()
    call @cpu_reference(%alloc_2, %alloc_6, %alloc_8) : (memref<4194304xf32>, memref<4194304xf32>, memref<4194304xf32>) -> ()
    call @cpu_reference(%alloc_4, %alloc_6, %alloc_10) : (memref<4194304xf32>, memref<4194304xf32>, memref<4194304xf32>) -> ()
    call @cpu_reference(%alloc_8, %alloc_10, %alloc_12) : (memref<4194304xf32>, memref<4194304xf32>, memref<4194304xf32>) -> ()
    %cast_17 = memref.cast %alloc_12 : memref<4194304xf32> to memref<*xf32>
    %cast_18 = memref.cast %memref_13 : memref<4194304xf32> to memref<*xf32>
    call @printAllcloseF32(%cast_17, %cast_18) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<4194304xf32>
    memref.dealloc %alloc_0 : memref<4194304xf32>
    memref.dealloc %alloc_2 : memref<4194304xf32>
    memref.dealloc %alloc_4 : memref<4194304xf32>
    memref.dealloc %alloc_6 : memref<4194304xf32>
    memref.dealloc %alloc_8 : memref<4194304xf32>
    memref.dealloc %alloc_12 : memref<4194304xf32>
    gpu.dealloc  %memref : memref<4194304xf32>
    gpu.dealloc  %memref_1 : memref<4194304xf32>
    gpu.dealloc  %memref_3 : memref<4194304xf32>
    gpu.dealloc  %memref_5 : memref<4194304xf32>
    gpu.dealloc  %memref_7 : memref<4194304xf32>
    gpu.dealloc  %memref_9 : memref<4194304xf32>
    gpu.dealloc  %memref_13 : memref<4194304xf32>
    return
  }
  func.func private @fillResource1DRandomF32(memref<*xf32>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DF32(memref<*xf32>, f32) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
