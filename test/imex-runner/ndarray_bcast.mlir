// RUN: %python_executable %imex_runner -i %s -f %p/ndarray.pp -e main -entry-point-result=void --shared-libs=%mlir_c_runner_utils --shared-libs=%mlir_runner_utils | FileCheck %s
// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/ndarray-gpu.pp --runner imex-cpu-runner -e main --entry-point-result=void --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck

func.func private @printMemrefI32(%ptr : tensor<*xi32>)

func.func @main() {
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
    %i3 = arith.constant 3 : index
    %i5 = arith.constant 5 : index
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c5 = arith.constant 5 : i32

    %x0 = ndarray.linspace %c0 %c5 %c5 false : (i32, i32, i32) -> !ndarray.ndarray<5xi32>
    %0 = "ndarray.reshape"(%x0, %i5, %i1) : (!ndarray.ndarray<5xi32>, index, index) -> !ndarray.ndarray<5x1xi32>
    %1 = ndarray.create value %c2 {dtype = 4 : i8} : (i32) -> !ndarray.ndarray<i32>
    %2 = ndarray.create %i1 value %c2 {dtype = 4 : i8} : (index, i32) -> !ndarray.ndarray<1xi32>
    %3 = ndarray.create %i1, %i3 value %c2 {dtype = 4 : i8} : (index, index, i32) -> !ndarray.ndarray<1x3xi32>
    %4 = ndarray.create %i5, %i3 value %c5 {dtype = 4 : i8} : (index, index, i32) -> !ndarray.ndarray<5x3xi32>
    %5 = ndarray.create %i2, %i1, %i2, %i1 value %c5 {dtype = 4 : i8} : (index, index, index, index, i32) -> !ndarray.ndarray<2x1x2x1xi32>
    %6 = ndarray.create %i2, %i2, %i2, %i2 value %c2 {dtype = 4 : i8} : (index, index, index, index, i32) -> !ndarray.ndarray<2x2x2x2xi32>

    call @test_arith_1D_A(%0, %1) : (!ndarray.ndarray<5x1xi32>, !ndarray.ndarray<i32>) -> ()
    call @test_arith_1D_B(%0, %2) : (!ndarray.ndarray<5x1xi32>, !ndarray.ndarray<1xi32>) -> ()
    call @test_arith_2D_A(%0, %3) : (!ndarray.ndarray<5x1xi32>, !ndarray.ndarray<1x3xi32>) -> ()
    call @test_arith_2D_B(%1, %4) : (!ndarray.ndarray<i32>, !ndarray.ndarray<5x3xi32>) -> ()
    call @test_arith_4D_A(%5, %6) : (!ndarray.ndarray<2x1x2x1xi32>, !ndarray.ndarray<2x2x2x2xi32>) -> ()

    // FIXME tosa-make-broadcastable can insert reshapes which breaks GPU lowering
    // call @test_tosa_1D_B(%0, %2) : (!ndarray.ndarray<5x1xi32>, !ndarray.ndarray<1xi32>) -> ()
    // // NOTE tosa 2d broadcasting is broken
    // // call @test_tosa_2D_A(%0, %3) : (!ndarray.ndarray<?xi32>, !ndarray.ndarray<1x?xi32>) -> ()
    // call @test_tosa_2D_B(%1, %4) : (!ndarray.ndarray<i32>, !ndarray.ndarray<?x?xi32>) -> ()

    return
}

func.func @test_arith_1D_A(%a : !ndarray.ndarray<5x1xi32>, %b : !ndarray.ndarray<i32>) {

    %0 = ndarray.ewbin %a, %b {op = 0 : i32} : (!ndarray.ndarray<5x1xi32>, !ndarray.ndarray<i32>) -> !ndarray.ndarray<?x?xi32>

    %1 = ndarray.to_tensor %0 : !ndarray.ndarray<?x?xi32> -> tensor<?x?xi32>
    %2 = tensor.cast %1 : tensor<?x?xi32> to tensor<*xi32>
    call @printMemrefI32(%2) : (tensor<*xi32>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 2 offset = 0 sizes = [5, 1] strides = [1, 1] data =
    // CHECK-NEXT{LITERAL}: [[2],
    // CHECK-NEXT{LITERAL}:  [3],
    // CHECK-NEXT{LITERAL}:  [4],
    // CHECK-NEXT{LITERAL}:  [5],
    // CHECK-NEXT{LITERAL}:  [6]]

    %5 = ndarray.ewbin %b, %a {op = 0 : i32} : (!ndarray.ndarray<i32>, !ndarray.ndarray<5x1xi32>) -> !ndarray.ndarray<?x?xi32>

    %6 = ndarray.to_tensor %5 : !ndarray.ndarray<?x?xi32> -> tensor<?x?xi32>
    %7 = tensor.cast %6 : tensor<?x?xi32> to tensor<*xi32>
    call @printMemrefI32(%7) : (tensor<*xi32>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 2 offset = 0 sizes = [5, 1] strides = [1, 1] data =
    // CHECK-NEXT{LITERAL}: [[2],
    // CHECK-NEXT{LITERAL}:  [3],
    // CHECK-NEXT{LITERAL}:  [4],
    // CHECK-NEXT{LITERAL}:  [5],
    // CHECK-NEXT{LITERAL}:  [6]]

    return
}

func.func @test_arith_1D_B(%a : !ndarray.ndarray<5x1xi32>, %b : !ndarray.ndarray<1xi32>) {

    %0 = ndarray.ewbin %a, %b {op = 0 : i32} : (!ndarray.ndarray<5x1xi32>, !ndarray.ndarray<1xi32>) -> !ndarray.ndarray<?x?xi32>

    %1 = ndarray.to_tensor %0 : !ndarray.ndarray<?x?xi32> -> tensor<?x?xi32>
    %2 = tensor.cast %1 : tensor<?x?xi32> to tensor<*xi32>
    call @printMemrefI32(%2) : (tensor<*xi32>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 2 offset = 0 sizes = [5, 1] strides = [1, 1] data =
    // CHECK-NEXT{LITERAL}: [[2],
    // CHECK-NEXT{LITERAL}:  [3],
    // CHECK-NEXT{LITERAL}:  [4],
    // CHECK-NEXT{LITERAL}:  [5],
    // CHECK-NEXT{LITERAL}:  [6]]

    %5 = ndarray.ewbin %b, %a {op = 0 : i32} : (!ndarray.ndarray<1xi32>, !ndarray.ndarray<5x1xi32>) -> !ndarray.ndarray<?x?xi32>

    %6 = ndarray.to_tensor %5 : !ndarray.ndarray<?x?xi32> -> tensor<?x?xi32>
    %7 = tensor.cast %6 : tensor<?x?xi32> to tensor<*xi32>
    call @printMemrefI32(%7) : (tensor<*xi32>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 2 offset = 0 sizes = [5, 1] strides = [1, 1] data =
    // CHECK-NEXT{LITERAL}: [[2],
    // CHECK-NEXT{LITERAL}:  [3],
    // CHECK-NEXT{LITERAL}:  [4],
    // CHECK-NEXT{LITERAL}:  [5],
    // CHECK-NEXT{LITERAL}:  [6]]

    return
}

func.func @test_arith_2D_A(%a : !ndarray.ndarray<5x1xi32>, %b : !ndarray.ndarray<1x3xi32>) {

    %0 = ndarray.ewbin %a, %b {op = 0 : i32} : (!ndarray.ndarray<5x1xi32>, !ndarray.ndarray<1x3xi32>) -> !ndarray.ndarray<?x?xi32>

    %1 = ndarray.to_tensor %0 : !ndarray.ndarray<?x?xi32> -> tensor<?x?xi32>
    %2 = tensor.cast %1 : tensor<?x?xi32> to tensor<*xi32>
    call @printMemrefI32(%2) : (tensor<*xi32>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 2 offset = 0 sizes = [5, 3] strides = [3, 1] data =
    // CHECK-NEXT{LITERAL}: [[2,   2,   2],
    // CHECK-NEXT{LITERAL}:  [3,   3,   3],
    // CHECK-NEXT{LITERAL}:  [4,   4,   4],
    // CHECK-NEXT{LITERAL}:  [5,   5,   5],
    // CHECK-NEXT{LITERAL}:  [6,   6,   6]]

    %5 = ndarray.ewbin %b, %a {op = 0 : i32} : (!ndarray.ndarray<1x3xi32>, !ndarray.ndarray<5x1xi32>) -> !ndarray.ndarray<?x?xi32>

    %6 = ndarray.to_tensor %5 : !ndarray.ndarray<?x?xi32> -> tensor<?x?xi32>
    %7 = tensor.cast %6 : tensor<?x?xi32> to tensor<*xi32>
    call @printMemrefI32(%7) : (tensor<*xi32>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 2 offset = 0 sizes = [5, 3] strides = [3, 1] data =
    // CHECK-NEXT{LITERAL}: [[2,   2,   2],
    // CHECK-NEXT{LITERAL}:  [3,   3,   3],
    // CHECK-NEXT{LITERAL}:  [4,   4,   4],
    // CHECK-NEXT{LITERAL}:  [5,   5,   5],
    // CHECK-NEXT{LITERAL}:  [6,   6,   6]]

    return
}

func.func @test_arith_2D_B(%a : !ndarray.ndarray<i32>, %b : !ndarray.ndarray<5x3xi32>) {

    %0 = ndarray.ewbin %a, %b {op = 0 : i32} : (!ndarray.ndarray<i32>, !ndarray.ndarray<5x3xi32>) -> !ndarray.ndarray<?x?xi32>

    %1 = ndarray.to_tensor %0 : !ndarray.ndarray<?x?xi32> -> tensor<?x?xi32>
    %2 = tensor.cast %1 : tensor<?x?xi32> to tensor<*xi32>
    call @printMemrefI32(%2) : (tensor<*xi32>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 2 offset = 0 sizes = [5, 3] strides = [3, 1] data =
    // CHECK-NEXT{LITERAL}: [[7,   7,   7],
    // CHECK-NEXT{LITERAL}:  [7,   7,   7],
    // CHECK-NEXT{LITERAL}:  [7,   7,   7],
    // CHECK-NEXT{LITERAL}:  [7,   7,   7],
    // CHECK-NEXT{LITERAL}:  [7,   7,   7]]

    %5 = ndarray.ewbin %b, %a {op = 0 : i32} : (!ndarray.ndarray<5x3xi32>, !ndarray.ndarray<i32>) -> !ndarray.ndarray<?x?xi32>

    %6 = ndarray.to_tensor %5 : !ndarray.ndarray<?x?xi32> -> tensor<?x?xi32>
    %7 = tensor.cast %6 : tensor<?x?xi32> to tensor<*xi32>
    call @printMemrefI32(%7) : (tensor<*xi32>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 2 offset = 0 sizes = [5, 3] strides = [3, 1] data =
    // CHECK-NEXT{LITERAL}: [[7,   7,   7],
    // CHECK-NEXT{LITERAL}:  [7,   7,   7],
    // CHECK-NEXT{LITERAL}:  [7,   7,   7],
    // CHECK-NEXT{LITERAL}:  [7,   7,   7],
    // CHECK-NEXT{LITERAL}:  [7,   7,   7]]

    return
}

func.func @test_arith_4D_A(%a : !ndarray.ndarray<2x1x2x1xi32>, %b : !ndarray.ndarray<2x2x2x2xi32>) {

    %0 = ndarray.ewbin %a, %b {op = 0 : i32} : (!ndarray.ndarray<2x1x2x1xi32>, !ndarray.ndarray<2x2x2x2xi32>) -> !ndarray.ndarray<?x?x?x?xi32>

    %1 = ndarray.to_tensor %0 : !ndarray.ndarray<?x?x?x?xi32> -> tensor<?x?x?x?xi32>
    %2 = tensor.cast %1 : tensor<?x?x?x?xi32> to tensor<*xi32>
    call @printMemrefI32(%2) : (tensor<*xi32>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 4 offset = 0 sizes = [2, 2, 2, 2] strides = [8, 4, 2, 1] data =
    // CHECK-NEXT{LITERAL}: [[[[7,     7],
    // CHECK-NEXT{LITERAL}:    [7,     7]],
    // CHECK-NEXT{LITERAL}:   [[7,     7],
    // CHECK-NEXT{LITERAL}:    [7,     7]]],
    // CHECK-NEXT{LITERAL}:  [[[7,     7],
    // CHECK-NEXT{LITERAL}:    [7,     7]],
    // CHECK-NEXT{LITERAL}:   [[7,     7],
    // CHECK-NEXT{LITERAL}:    [7,     7]]]]

    %5 = ndarray.ewbin %b, %a {op = 0 : i32} : (!ndarray.ndarray<2x2x2x2xi32>, !ndarray.ndarray<2x1x2x1xi32>) -> !ndarray.ndarray<?x?x?x?xi32>

    %6 = ndarray.to_tensor %5 : !ndarray.ndarray<?x?x?x?xi32> -> tensor<?x?x?x?xi32>
    %7 = tensor.cast %6 : tensor<?x?x?x?xi32> to tensor<*xi32>
    call @printMemrefI32(%7) : (tensor<*xi32>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 4 offset = 0 sizes = [2, 2, 2, 2] strides = [8, 4, 2, 1] data =
    // CHECK-NEXT{LITERAL}: [[[[7,     7],
    // CHECK-NEXT{LITERAL}:    [7,     7]],
    // CHECK-NEXT{LITERAL}:   [[7,     7],
    // CHECK-NEXT{LITERAL}:    [7,     7]]],
    // CHECK-NEXT{LITERAL}:  [[[7,     7],
    // CHECK-NEXT{LITERAL}:    [7,     7]],
    // CHECK-NEXT{LITERAL}:   [[7,     7],
    // CHECK-NEXT{LITERAL}:    [7,     7]]]]

    return
}

// func.func @test_tosa_1D_B(%a : !ndarray.ndarray<5x1xi32>, %b : !ndarray.ndarray<1xi32>) {

//     %0 = ndarray.ewbin %a, %b {op = 4 : i32} : (!ndarray.ndarray<5x1xi32>, !ndarray.ndarray<1xi32>) -> !ndarray.ndarray<?x?xi32>

//     %1 = ndarray.to_tensor %0 : !ndarray.ndarray<?x?xi32> -> tensor<?x?xi32>
//     %2 = tensor.cast %1 : tensor<?x?xi32> to tensor<*xi32>
//     call @printMemrefI32(%2) : (tensor<*xi32>) -> ()
//     // SKIP-CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
//     // SKIP-CHECK-SAME: rank = 1 offset = 0 sizes = [5] strides = [1] data =
//     // SKIP-CHECK-NEXT: [2,  3,  2,  3,  6]

//     %5 = ndarray.ewbin %b, %a {op = 4 : i32} : (!ndarray.ndarray<1xi32>, !ndarray.ndarray<5x1xi32>) -> !ndarray.ndarray<?x?xi32>

//     %6 = ndarray.to_tensor %5 : !ndarray.ndarray<?x?xi32> -> tensor<?x?xi32>
//     %7 = tensor.cast %6 : tensor<?x?xi32> to tensor<*xi32>
//     call @printMemrefI32(%7) : (tensor<*xi32>) -> ()
//     // SKIP-CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
//     // SKIP-CHECK-SAME: rank = 1 offset = 0 sizes = [5] strides = [1] data =
//     // SKIP-CHECK-NEXT: [2,  3,  2,  3,  6]

//     return
// }

// // func.func @test_tosa_2D_A(%a : !ndarray.ndarray<?xi32>, %b : !ndarray.ndarray<1x?xi32>) {

// //     %0 = ndarray.ewbin %a, %b {op = 4 : i32} : (!ndarray.ndarray<?xi32>, !ndarray.ndarray<1x?xi32>) -> !ndarray.ndarray<?x?xi32>

// //     %1 = ndarray.to_tensor %0 : !ndarray.ndarray<?x?xi32> -> tensor<?x?xi32>
// //     %2 = tensor.cast %1 : tensor<?x?xi32> to tensor<*xi32>
// //     call @printMemrefI32(%2) : (tensor<*xi32>) -> ()
// //     // SKIP-CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
// //     // SKIP-CHECK-SAME: rank = 2 offset = 0 sizes = [5, 3] strides = [3, 1] data =
// //     // SKIP-CHECK-NEXT{LITERAL}: [[2,   2,   2],
// //     // SKIP-CHECK-NEXT{LITERAL}:  [3,   3,   3],
// //     // SKIP-CHECK-NEXT{LITERAL}:  [2,   2,   2],
// //     // SKIP-CHECK-NEXT{LITERAL}:  [3,   3,   3],
// //     // SKIP-CHECK-NEXT{LITERAL}:  [6,   6,   6]]

// //     %5 = ndarray.ewbin %b, %a {op = 4 : i32} : (!ndarray.ndarray<1x?xi32>, !ndarray.ndarray<?xi32>) -> !ndarray.ndarray<?x?xi32>

// //     %6 = "ndarray-to_tensor"(%5) : (!ndarray.ndarray<?x?xi32>) -> tensor<?x?xi32>
// //     %7 = tensor.cast %6 : tensor<?x?xi32> to tensor<*xi32>
// //     call @printMemrefI32(%7) : (tensor<*xi32>) -> ()
// //     // SKIP-CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
// //     // SKIP-CHECK-SAME: rank = 2 offset = 0 sizes = [5, 3] strides = [3, 1] data =
// //     // SKIP-CHECK-NEXT{LITERAL}: [[2,   2,   2],
// //     // SKIP-CHECK-NEXT{LITERAL}:  [3,   3,   3],
// //     // SKIP-CHECK-NEXT{LITERAL}:  [2,   2,   2],
// //     // SKIP-CHECK-NEXT{LITERAL}:  [3,   3,   3],
// //     // SKIP-CHECK-NEXT{LITERAL}:  [6,   6,   6]]

// //     return
// // }

// func.func @test_tosa_2D_B(%a : !ndarray.ndarray<i32>, %b : !ndarray.ndarray<?x?xi32>) {

//     %0 = ndarray.ewbin %a, %b {op = 4 : i32} : (!ndarray.ndarray<i32>, !ndarray.ndarray<?x?xi32>) -> !ndarray.ndarray<?x?xi32>

//     %1 = ndarray.to_tensor %0 : !ndarray.ndarray<?x?xi32> -> tensor<?x?xi32>
//     %2 = tensor.cast %1 : tensor<?x?xi32> to tensor<*xi32>
//     call @printMemrefI32(%2) : (tensor<*xi32>) -> ()
//     // SKIP-CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
//     // SKIP-CHECK-SAME: rank = 2 offset = 0 sizes = [5, 3] strides = [3, 1] data =
//     // SKIP-CHECK-NEXT{LITERAL}: [[7,   7,   7],
//     // SKIP-CHECK-NEXT{LITERAL}:  [7,   7,   7],
//     // SKIP-CHECK-NEXT{LITERAL}:  [7,   7,   7],
//     // SKIP-CHECK-NEXT{LITERAL}:  [7,   7,   7],
//     // SKIP-CHECK-NEXT{LITERAL}:  [7,   7,   7]]

//     %5 = ndarray.ewbin %b, %a {op = 4 : i32} : (!ndarray.ndarray<?x?xi32>, !ndarray.ndarray<i32>) -> !ndarray.ndarray<?x?xi32>

//     %6 = "ndarray-to_tensor"(%5) : (!ndarray.ndarray<?x?xi32>) -> tensor<?x?xi32>
//     %7 = tensor.cast %6 : tensor<?x?xi32> to tensor<*xi32>
//     call @printMemrefI32(%7) : (tensor<*xi32>) -> ()
//     // SKIP-CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
//     // SKIP-CHECK-SAME: rank = 2 offset = 0 sizes = [5, 3] strides = [3, 1] data =
//     // SKIP-CHECK-NEXT{LITERAL}: [[7,   7,   7],
//     // SKIP-CHECK-NEXT{LITERAL}:  [7,   7,   7],
//     // SKIP-CHECK-NEXT{LITERAL}:  [7,   7,   7],
//     // SKIP-CHECK-NEXT{LITERAL}:  [7,   7,   7],
//     // SKIP-CHECK-NEXT{LITERAL}:  [7,   7,   7]]

//     return
// }
