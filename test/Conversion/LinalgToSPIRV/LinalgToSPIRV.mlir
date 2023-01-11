// RUN: mlir-opt %s -convert-linalg-to-loops -convert-scf-to-cf -convert-linalg-to-llvm  \
// RUN:             -convert-arith-to-llvm -convert-memref-to-llvm -convert-func-to-llvm \
// RUN:             -reconcile-unrealized-casts |                                        \
// RUN: mlir-cpu-runner -O3 -e main -entry-point-result=void                             \
// RUN:   -shared-libs=%mlir_lib_dir/libmlir_runner_utils%shlibext                       \
// RUN: | FileCheck %s

func.func private @printMemrefF32(memref<*xf32>)
// func.func private @verifyMemRefF32(index, index, index) -> i64

!arr_t = memref<?x?xf32>

func.func @matmul(%A: !arr_t, %B: !arr_t) -> (!arr_t) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f0 = arith.constant 0.0 : f32
  %x = memref.dim %A, %c0 : !arr_t
  %y = memref.dim %B, %c1 : !arr_t
  %C = memref.alloc(%x, %y) : !arr_t
  linalg.fill ins(%f0 : f32) outs(%C : !arr_t)
  linalg.matmul ins(%A, %B: !arr_t, !arr_t)
                outs(%C: !arr_t)
  return %C : !arr_t
}


// ARG0[M, K] * ARG1[K, N] -> [M, N]
// TODO need "out=" parameter to allow user allocated memory
func.func @reference(%arg0: !arr_t, %arg1: !arr_t) -> !arr_t
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init_val_f32 = arith.constant 0.0 : f32

  // get size values from input array dimensions
  %dim_M = memref.dim %arg0, %c0 : !arr_t
  %dim_K = memref.dim %arg0, %c1 : !arr_t
  %dim_N = memref.dim %arg1, %c1 : !arr_t

  // check shape correctness
  %dim_K_ref = memref.dim %arg1, %c0 : !arr_t
  %check1 = arith.cmpi eq, %dim_K, %dim_K_ref : index
  cf.assert %check1, "Matrix shape indexes LHS[?, K] should equal to RHS[K, ?]"

  // alloc output array
  %result_ptr = memref.alloc(%dim_M, %dim_N) : !arr_t

  // compute matrix multiplication
  // TODO use scf.parallel
  scf.for %it_M = %c0 to %dim_M step %c1
  {
    scf.for %it_N = %c0 to %dim_N step %c1
    {

      // do reduction into accumulator
      %accum = scf.for %it_K = %c0 to %dim_K step %c1
                 iter_args(%accum_iter = %init_val_f32) -> f32
      {
        %0 = memref.load %arg0[%it_M, %it_K] : !arr_t
        %1 = memref.load %arg1[%it_K, %it_N] : !arr_t
        %3 = arith.mulf %0, %1 : f32
        %4 = arith.addf %accum_iter, %3 : f32
        scf.yield %4 : f32
      }

      memref.store %accum, %result_ptr[%it_M, %it_N] : !arr_t
    }
  }

  return %result_ptr : !arr_t
}

func.func @main()
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %m = arith.constant 5 : index
  %n = arith.constant 2 : index
  %k = arith.constant 3 : index

  %val1 = arith.constant 42.0 : f32
  %val2 = arith.constant 17.0 : f32

  %A = memref.alloc(%m, %k) : !arr_t
  %B = memref.alloc(%k, %n) : !arr_t

  linalg.fill ins(%val1 : f32) outs(%A : !arr_t)
  linalg.fill ins(%val2 : f32) outs(%B : !arr_t)
  memref.store %val1, %B[%c0, %c0] : !arr_t

  %C1 = call @matmul(%A, %B) : (!arr_t, !arr_t) -> !arr_t
  %C2 = call @reference(%A, %B) : (!arr_t, !arr_t) -> !arr_t

  scf.for %i = %c0 to %m step %c1 {
    scf.for %j = %c0 to %n step %c1 {
      %e1 = memref.load %C1[%i, %j] : !arr_t
      %e2 = memref.load %C2[%i, %j] : !arr_t
      %c = arith.cmpf oeq, %e1, %e2 : f32
      //cf.assert %c, "Matmul does not produce same output as matvec"
    }
  }
  %C1_ = memref.cast %C1 : !arr_t to memref<*xf32>
  %C2_ = memref.cast %C2 : !arr_t to memref<*xf32>
  call @printMemrefF32(%C1_) : (memref<*xf32>) -> ()
  call @printMemrefF32(%C2_) : (memref<*xf32>) -> ()

  // %C1_1 = memref.extract_aligned_pointer_as_index %C1 : !arr_t -> index
  // %C2_1 = memref.extract_aligned_pointer_as_index %C2 : !arr_t -> index
  // %cmp_result = call @verifyMemRefF32(%n, %C1_1, %C2_1)  : (index, index, index) -> i64

  memref.dealloc %C1 : !arr_t
  memref.dealloc %C2 : !arr_t

  return
}

// CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [5, 2] strides = [2, 1] data =
// CHECK-NEXT:      [
// CHECK-SAME:  [3192,  2142],
// CHECK-NEXT:  [3192,   2142],
// CHECK-NEXT:  [3192,   2142],
// CHECK-NEXT:  [3192,   2142],
// CHECK-NEXT:  [3192,   2142]
// CHECK-SAME: ]
