// RUN: %{python_executable} %{imex_tools_dir}/imex-runner.py -i %s --pass-pipeline-file=%p/ptensor.pp -e main -entry-point-result=void --shared-libs=%{mlir_shlib_dir}/libmlir_c_runner_utils%shlibext --shared-libs=%{mlir_shlib_dir}/libmlir_runner_utils%shlibext | FileCheck %s

module {
    func.func @arange(%arg0: i64, %arg1: i64, %arg2: i64) -> !ptensor.ptensor<tensor<?xi64>, 0> {
        %0 = "ptensor.arange"(%arg0, %arg1, %arg2) {dist = false} : (i64, i64, i64) -> !ptensor.ptensor<tensor<?xi64>, 0>
        return  %0 : !ptensor.ptensor<tensor<?xi64>, 0>
    }

    func.func @arange_dist(%arg0: i64, %arg1: i64, %arg2: i64) -> !ptensor.ptensor<tensor<?xi64>, 1> {
        %0 = "ptensor.arange"(%arg0, %arg1, %arg2) {dist = false} : (i64, i64, i64) -> !ptensor.ptensor<tensor<?xi64>, 1>
        return  %0 : !ptensor.ptensor<tensor<?xi64>, 1>
    }


    func.func @main() {
        %0 = arith.constant 0 : i64
        %1 = arith.constant 10 : i64
        %2 = arith.constant 2 : i64

        %3 = call @arange(%0, %1, %2) : (i64, i64, i64) -> (!ptensor.ptensor<tensor<?xi64>, 0>)
        %4 = "builtin.unrealized_conversion_cast"(%3) : (!ptensor.ptensor<tensor<?xi64>, 0>) -> tensor<?xi64>
        %5 = tensor.cast %4 : tensor<?xi64> to tensor<*xi64>
        call @printMemrefI64(%5) : (tensor<*xi64>) -> ()
        // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
        // CHECK-SAME: rank = 1 offset = 0 sizes = [5] strides = [1] data =
        // CHECK-NEXT: [0,  2,  4,  6,  8]

        %13 = call @arange_dist(%0, %1, %2) : (i64, i64, i64) -> (!ptensor.ptensor<tensor<?xi64>, 1>)
        %14 = "builtin.unrealized_conversion_cast"(%13) : (!ptensor.ptensor<tensor<?xi64>, 1>) -> tensor<?xi64>
        %15 = tensor.cast %14 : tensor<?xi64> to tensor<*xi64>
        call @printMemrefI64(%15) : (tensor<*xi64>) -> ()
        // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
        // CHECK-SAME: rank = 1 offset = 0 sizes = [5] strides = [1] data =
        // CHECK-NEXT: [0,  2,  4,  6,  8]

        return
    }

    func.func private @printMemrefI64(%ptr : tensor<*xi64>)
}
