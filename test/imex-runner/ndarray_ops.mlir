// RUN: %python_executable %imex_runner -i %s -f %p/ndarray.pp -e main -entry-point-result=void --shared-libs=%mlir_c_runner_utils --shared-libs=%mlir_runner_utils | FileCheck %s

module {
  func.func private @printMemrefI32(tensor<*xi32>)
  func.func private @printMemrefF32(tensor<*xf32>)
  func.func @main() {
    call @test_linspace_i32() : () -> ()
    call @test_linspace_f32() : () -> ()
    call @test_subview() : () -> ()
    call @test_subview_2d() : () -> ()
    call @test_insert_slice() : () -> ()
    call @test_insert_slice_scalar() : () -> ()
    call @test_insert_slice_2d() : () -> ()
    // call @test_reshape() : () -> ()
    return
  }

  func.func @test_linspace_i32() {
    %c0 = arith.constant 0 : i32
    %c5 = arith.constant 5 : i32
    %c10 = arith.constant 10 : i32
    %0 = ndarray.linspace %c0 %c10 %c5 false   : (i32, i32, i32) -> tensor<?xi32>
    %cast = tensor.cast %0 : tensor<?xi32> to tensor<*xi32>
    call @printMemrefI32(%cast) : (tensor<*xi32>) -> ()
    return
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 1 offset = 0 sizes = [5] strides = [1] data =
    // CHECK-NEXT: [0,  2,  4,  6,  8]
  }

  func.func @test_linspace_f32() {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 4.000000e+00 : f32
    %c9 = arith.constant 9 : i32
    %0 = ndarray.linspace %cst %cst_0 %c9 true   : (f32, f32, i32) -> tensor<?xf32>
    %cast = tensor.cast %0 : tensor<?xf32> to tensor<*xf32>
    call @printMemrefF32(%cast) : (tensor<*xf32>) -> ()
    return
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 1 offset = 0 sizes = [9] strides = [1] data =
    // CHECK-NEXT: [0,  0.5,  1,  1.5,  2,  2.5,  3,  3.5,  4]
  }

  func.func @test_subview() {
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
    %c0 = arith.constant 0 : i32
    %c5 = arith.constant 5 : i32
    %c10 = arith.constant 10 : i32
    %0 = ndarray.linspace %c0 %c10 %c5 false   : (i32, i32, i32) -> tensor<?xi32>
    %1 = ndarray.subview %0[%i1] [%i2] [%i2] : tensor<?xi32> to tensor<?xi32>
    %cast = tensor.cast %1 : tensor<?xi32> to tensor<*xi32>
    call @printMemrefI32(%cast) : (tensor<*xi32>) -> ()
    return
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 1 offset = 1 sizes = [2] strides = [2] data =
    // CHECK-NEXT: [2, 6]
  }

  func.func @test_subview_2d() {
    %i4 = arith.constant 4 : index
    %c1 = arith.constant 1 : i32
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
    %empty = tensor.empty(%i4, %i4) : tensor<?x?xi32>
    %0 = linalg.fill ins(%c1 : i32) outs(%empty: tensor<?x?xi32>) -> tensor<?x?xi32>
    %1 = ndarray.subview %0[%i1, %i0] [%i2, %i2] [%i1, %i1] : tensor<?x?xi32> to tensor<?x?xi32>
    %cast = tensor.cast %1 : tensor<?x?xi32> to tensor<*xi32>
    call @printMemrefI32(%cast) : (tensor<*xi32>) -> ()
    return
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 2 offset = 4 sizes = [2, 2] strides = [4, 1] data =
    // CHECK-NEXT{LITERAL}: [[1,   1],
    // CHECK-NEXT{LITERAL}:  [1,   1]]
  }

  func.func @test_insert_slice() {
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
    %c0 = arith.constant 0 : i32
    %c2 = arith.constant 2 : i32
    %c5 = arith.constant 5 : i32
    %c10 = arith.constant 10 : i32
    %0 = ndarray.linspace %c0 %c10 %c5 false   : (i32, i32, i32) -> tensor<?xi32>
    %1 = ndarray.linspace %c0 %c2 %c2 false   : (i32, i32, i32) -> tensor<?xi32>
    %3 = ndarray.insert_slice %1 into %0[%i1] [%i2] [%i2] : tensor<?xi32> into tensor<?xi32>
    %cast = tensor.cast %3 : tensor<?xi32> to tensor<*xi32>
    call @printMemrefI32(%cast) : (tensor<*xi32>) -> ()
    return
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 1 offset = 0 sizes = [5] strides = [1] data =
    // CHECK-NEXT: [0,  0,  4,  1,  8]
  }

  func.func @test_insert_slice_2d() {
    %c1 = arith.constant 1 : i32
    %c5 = arith.constant 5 : i32
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
    %i4 = arith.constant 4 : index
    %empty = tensor.empty(%i4, %i4) : tensor<?x?xi32>
    %0 = linalg.fill ins(%c1 : i32) outs(%empty: tensor<?x?xi32>) -> tensor<?x?xi32>
    %empty1 = tensor.empty(%i2, %i2) : tensor<?x?xi32>
    %1 = linalg.fill ins(%c5 : i32) outs(%empty1: tensor<?x?xi32>) -> tensor<?x?xi32>
    %2 = ndarray.insert_slice %1 into %0[%i1, %i2] [%i2, %i2] [%i1, %i1] : tensor<?x?xi32> into tensor<?x?xi32>
    %cast = tensor.cast %2 : tensor<?x?xi32> to tensor<*xi32>
    call @printMemrefI32(%cast) : (tensor<*xi32>) -> ()
    return
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 2 offset = 0 sizes = [4, 4] strides = [4, 1] data =
    // CHECK-NEXT{LITERAL}: [[1,   1,   1,   1],
    // CHECK-NEXT{LITERAL}:  [1,   1,   5,   5],
    // CHECK-NEXT{LITERAL}:  [1,   1,   5,   5],
    // CHECK-NEXT{LITERAL}:  [1,   1,   1,   1]]
  }

  func.func @test_insert_slice_scalar() {
    %c1 = arith.constant 1 : i32
    %c5 = arith.constant 5 : i32
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
    %i4 = arith.constant 4 : index
    %empty = tensor.empty(%i4, %i4) : tensor<?x?xi32>
    %0 = linalg.fill ins(%c1 : i32) outs(%empty: tensor<?x?xi32>) -> tensor<?x?xi32>
    %empty1 = tensor.empty() : tensor<i32>
    %1 = linalg.fill ins(%c5 : i32) outs(%empty1: tensor<i32>) -> tensor<i32>
    %2 = ndarray.insert_slice %1 into %0[%i1, %i2] [%i2, %i2] [%i1, %i1] : tensor<i32> into tensor<?x?xi32>
    %cast = tensor.cast %2 : tensor<?x?xi32> to tensor<*xi32>
    call @printMemrefI32(%cast) : (tensor<*xi32>) -> ()
    return
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 2 offset = 0 sizes = [4, 4] strides = [4, 1] data =
    // CHECK-NEXT{LITERAL}: [[1,   1,   1,   1],
    // CHECK-NEXT{LITERAL}:  [1,   1,   5,   5],
    // CHECK-NEXT{LITERAL}:  [1,   1,   5,   5],
    // CHECK-NEXT{LITERAL}:  [1,   1,   1,   1]]
  }

  // func.func @test_reshape() {
  //   %i1 = arith.constant 1 : index
  //   %i5 = arith.constant 5 : index
  //   %c0 = arith.constant 0 : i32
  //   %c5 = arith.constant 5 : i32
  //   %c10 = arith.constant 10 : i32
  //   %3 = ndarray.linspace %c0 %c10 %c5 false   : (i32, i32, i32) -> tensor<?xi32>
  //   %60 = "ndarray.reshape"(%3, %i5, %i1) : (tensor<?xi32>, index, index) -> tensor<?x?xi32>
  //   %32 = builtin.unrealized_conversion_cast %60 : (tensor<?x?xi32>) -> tensor<?x?xi32>
  //   %65 = tensor.cast %32 : tensor<?x?xi32> to tensor<*xi32>
  //   call @printMemrefI32(%65) : (tensor<*xi32>) -> ()
  //   // FIXME_CHECK: rank = 2 offset = 0 sizes = [5, 1] strides = [1, 1] data =
  //   // FIXME_CHECK-NEXT{LITERAL}: [[0],
  //   // FIXME_CHECK-NEXT{LITERAL}: [2],
  //   // FIXME_CHECK-NEXT{LITERAL}: [4],
  //   // FIXME_CHECK-NEXT{LITERAL}: [6],
  //   // FIXME_CHECK-NEXT{LITERAL}: [8]]
  //   return

    // %s = ndarray.create %i1, %i5 value %c5 {dtype = 4 : i8} : (index, index, i32) -> tensor<?x?xi32>
    // %70 = ndarray.linspace %i0 %i36 %i36 false : (index, index, index) -> tensor<?xi32>
    // %71 = "ndarray.reshape"(%70, %i6, %i6) : (tensor<?xi32>, index, index) -> tensor<?x?xi32>
    // %75 = "ndarray.reshape"(%71, %i4, %i3, %i3) {copy = 1 : i1} : (tensor<?x?xi32>, index, index, index) -> tensor<?x?x?xi32>
    // %76 = "ndarray.reshape"(%71, %i36) {copy = 0 : i1} : (tensor<?x?xi32>, index) -> tensor<?xi32>
    // // we modify the first reshaped, the second should not change, the third should
    // ndarray.insert_slice %s into %71[%i1, %i1] [%i1, %i5] [%i5, %i1] : tensor<?x?xi32> into tensor<?x?xi32>
    // %72 = builtin.unrealized_conversion_cast %71 : (tensor<?x?xi32>) -> tensor<?x?xi32>
    // %73 = tensor.cast %72 : tensor<?x?xi32> to tensor<*xi32>
    // call @printMemrefI32(%73) : (tensor<*xi32>) -> ()
    // // FIXME_CHECK-NEXT{LITERAL}: rank = 2 offset = 0 sizes = [6, 6] strides = [6, 1] data =
    // // FIXME_CHECK-NEXT{LITERAL}: [[0,   1,   2,   3,   4,   5],
    // // FIXME_CHECK-NEXT{LITERAL}:  [6,   5,   5,   5,   5,   5],
    // // FIXME_CHECK-NEXT{LITERAL}:  [12,   13,   14,   15,   16,   17],
    // // FIXME_CHECK-NEXT{LITERAL}:  [18,   19,   20,   21,   22,   23],
    // // FIXME_CHECK-NEXT{LITERAL}:  [24,   25,   26,   27,   28,   29],
    // // FIXME_CHECK-NEXT{LITERAL}:  [30,   31,   32,   33,   34,   35]]

    // %78 = builtin.unrealized_conversion_cast %75 : (tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
    // %79 = tensor.cast %78 : tensor<?x?x?xi32> to tensor<*xi32>
    // call @printMemrefI32(%79) : (tensor<*xi32>) -> ()
    // // FIXME_CHECK-NEXT{LITERAL}: rank = 3 offset = 0 sizes = [4, 3, 3] strides = [9, 3, 1] data =
    // // FIXME_CHECK-NEXT{LITERAL}: [[[0,    1,    2],
    // // FIXME_CHECK-NEXT{LITERAL}:   [3,    4,    5],
    // // FIXME_CHECK-NEXT{LITERAL}:   [6,    7,    8]],
    // // FIXME_CHECK-NEXT{LITERAL}:  [[9,    10,    11],
    // // FIXME_CHECK-NEXT{LITERAL}:   [12,    13,    14],
    // // FIXME_CHECK-NEXT{LITERAL}:   [15,    16,    17]],
    // // FIXME_CHECK-NEXT{LITERAL}:  [[18,    19,    20],
    // // FIXME_CHECK-NEXT{LITERAL}:   [21,    22,    23],
    // // FIXME_CHECK-NEXT{LITERAL}:   [24,    25,    26]],
    // // FIXME_CHECK-NEXT{LITERAL}:  [[27,    28,    29],
    // // FIXME_CHECK-NEXT{LITERAL}:   [30,    31,    32],
    // // FIXME_CHECK-NEXT{LITERAL}:   [33,    34,    35]]]

    // %80 = builtin.unrealized_conversion_cast %76 : (tensor<?xi32>) -> tensor<?xi32>
    // %81 = tensor.cast %80 : tensor<?xi32> to tensor<*xi32>
    // call @printMemrefI32(%81) : (tensor<*xi32>) -> ()
    // // FIXME_CHECK{LITERAL}: rank = 1 offset = 0 sizes = [36] strides = [1] data =
    // // FIXME_CHECK-NEXT{LITERAL}: [0,  1,  2,  3,  4,  5,  6,  5,  5,  5,  5,  5,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35]
    // }
}
