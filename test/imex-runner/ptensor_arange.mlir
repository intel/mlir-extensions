// RUN: %python_executable %imex_runner -i %s --pass-pipeline-file=%p/ptensor.pp -e main -entry-point-result=void --shared-libs=%mlir_c_runner_utils --shared-libs=%mlir_runner_utils | FileCheck %s

module {
    func.func private @printMemrefI64(%ptr : tensor<*xi64>)
    func.func @main() {
        %0 = arith.constant 0 : i64
        %1 = arith.constant 10 : i64
        %2 = arith.constant 2 : i64
        %c0_i64 = arith.constant 0 : i64
        %c10_i64 = arith.constant 10 : i64

        %3 = "ptensor.arange"(%0, %1, %2, %c0_i64, %c0_i64) : (i64, i64, i64, i64, i64) -> !ptensor.ptensor<tensor<?xi64>>
        %4 = "ptensor.extract_rtensor"(%3) : (!ptensor.ptensor<tensor<?xi64>>) -> tensor<?xi64>
        %5 = tensor.cast %4 : tensor<?xi64> to tensor<*xi64>
        call @printMemrefI64(%5) : (tensor<*xi64>) -> ()
        // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
        // CHECK-SAME: rank = 1 offset = 0 sizes = [5] strides = [1] data =
        // CHECK-NEXT: [0,  2,  4,  6,  8]

        %13 = "ptensor.arange"(%0, %1, %2, %c0_i64, %c10_i64) : (i64, i64, i64, i64, i64) -> !ptensor.ptensor<tensor<?xi64>, dist = 1>
        %14 = "ptensor.extract_rtensor"(%13) : (!ptensor.ptensor<tensor<?xi64>, dist = 1>) -> tensor<?xi64>
        %15 = tensor.cast %14 : tensor<?xi64> to tensor<*xi64>
        call @printMemrefI64(%15) : (tensor<*xi64>) -> ()
        // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
        // CHECK-SAME: rank = 1 offset = 0 sizes = [5] strides = [1] data =
        // CHECK-NEXT: [0,  2,  4,  6,  8]

        return
    }
}
