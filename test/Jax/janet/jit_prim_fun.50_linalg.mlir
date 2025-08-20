// RUN: %python_executable %imex_runner -i %s --pass-pipeline-file=%p/linalg-to-cpu.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils \
// RUN:                                       --entry-point-result=void --filecheck
// RUN-GPU: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/linalg-to-llvm.pp \
// RUN-GPU:                                       --runner mlir-runner -e main \
// RUN-GPU:                                       --entry-point-result=void \
// RUN-GPU:                                       --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN-GPU: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/linalg-to-llvm.pp \
// RUN-GPU:                                        --runner mlir-runner -e main \
// RUN-GPU:                                        --entry-point-result=void \
// RUN-GPU:                                        --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
#map = affine_map<(d0, d1) -> (d0, d1)>
module @jit_prim_fun.50 {

  func.func private @printMemrefF32(tensor<*xf32>)

  func.func private @callee(%arg0: tensor<1x6xf32>, %arg1: tensor<1x6xf32>, %arg2: tensor<1x6xf32>, %arg3: tensor<1x6xf32>, %arg4: tensor<1x6xf32>, %arg5: tensor<1x6xf32>, %arg6: tensor<1x6xf32>, %arg7: tensor<1x6xf32>, %arg8: tensor<1x6xf32>, %arg9: tensor<1x6xf32>, %arg10: tensor<1x6xf32>, %arg11: tensor<1x6xf32>, %arg12: tensor<1x6xf32>, %arg13: tensor<1x6xf32>, %arg14: tensor<1x6xf32>, %arg15: tensor<1x6xf32>) -> tensor<16x6xf32> {
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<16x6xf32>
    %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%0 : tensor<16x6xf32>) {
    ^bb0(%arg16: f32):
      %2 = linalg.index 0 : index
      %3 = linalg.index 1 : index
      %4 = linalg.index 0 : index
      %c0_0 = arith.constant 0 : index
      %5 = tensor.dim %arg0, %c0_0 : tensor<1x6xf32>
      %6 = arith.addi %c0, %5 : index
      %7 = arith.cmpi ult, %4, %6 : index
      %8 = scf.if %7 -> (f32) {
        %9 = arith.subi %4, %c0 : index
        %10 = tensor.extract %arg0[%9, %3] : tensor<1x6xf32>
        scf.yield %10 : f32
      } else {
        %c0_1 = arith.constant 0 : index
        %9 = tensor.dim %arg1, %c0_1 : tensor<1x6xf32>
        %10 = arith.addi %6, %9 : index
        %11 = arith.cmpi ult, %4, %10 : index
        %12 = scf.if %11 -> (f32) {
          %13 = arith.subi %4, %6 : index
          %14 = tensor.extract %arg1[%13, %3] : tensor<1x6xf32>
          scf.yield %14 : f32
        } else {
          %c0_2 = arith.constant 0 : index
          %13 = tensor.dim %arg2, %c0_2 : tensor<1x6xf32>
          %14 = arith.addi %10, %13 : index
          %15 = arith.cmpi ult, %4, %14 : index
          %16 = scf.if %15 -> (f32) {
            %17 = arith.subi %4, %10 : index
            %18 = tensor.extract %arg2[%17, %3] : tensor<1x6xf32>
            scf.yield %18 : f32
          } else {
            %c0_3 = arith.constant 0 : index
            %17 = tensor.dim %arg3, %c0_3 : tensor<1x6xf32>
            %18 = arith.addi %14, %17 : index
            %19 = arith.cmpi ult, %4, %18 : index
            %20 = scf.if %19 -> (f32) {
              %21 = arith.subi %4, %14 : index
              %22 = tensor.extract %arg3[%21, %3] : tensor<1x6xf32>
              scf.yield %22 : f32
            } else {
              %c0_4 = arith.constant 0 : index
              %21 = tensor.dim %arg4, %c0_4 : tensor<1x6xf32>
              %22 = arith.addi %18, %21 : index
              %23 = arith.cmpi ult, %4, %22 : index
              %24 = scf.if %23 -> (f32) {
                %25 = arith.subi %4, %18 : index
                %26 = tensor.extract %arg4[%25, %3] : tensor<1x6xf32>
                scf.yield %26 : f32
              } else {
                %c0_5 = arith.constant 0 : index
                %25 = tensor.dim %arg5, %c0_5 : tensor<1x6xf32>
                %26 = arith.addi %22, %25 : index
                %27 = arith.cmpi ult, %4, %26 : index
                %28 = scf.if %27 -> (f32) {
                  %29 = arith.subi %4, %22 : index
                  %30 = tensor.extract %arg5[%29, %3] : tensor<1x6xf32>
                  scf.yield %30 : f32
                } else {
                  %c0_6 = arith.constant 0 : index
                  %29 = tensor.dim %arg6, %c0_6 : tensor<1x6xf32>
                  %30 = arith.addi %26, %29 : index
                  %31 = arith.cmpi ult, %4, %30 : index
                  %32 = scf.if %31 -> (f32) {
                    %33 = arith.subi %4, %26 : index
                    %34 = tensor.extract %arg6[%33, %3] : tensor<1x6xf32>
                    scf.yield %34 : f32
                  } else {
                    %c0_7 = arith.constant 0 : index
                    %33 = tensor.dim %arg7, %c0_7 : tensor<1x6xf32>
                    %34 = arith.addi %30, %33 : index
                    %35 = arith.cmpi ult, %4, %34 : index
                    %36 = scf.if %35 -> (f32) {
                      %37 = arith.subi %4, %30 : index
                      %38 = tensor.extract %arg7[%37, %3] : tensor<1x6xf32>
                      scf.yield %38 : f32
                    } else {
                      %c0_8 = arith.constant 0 : index
                      %37 = tensor.dim %arg8, %c0_8 : tensor<1x6xf32>
                      %38 = arith.addi %34, %37 : index
                      %39 = arith.cmpi ult, %4, %38 : index
                      %40 = scf.if %39 -> (f32) {
                        %41 = arith.subi %4, %34 : index
                        %42 = tensor.extract %arg8[%41, %3] : tensor<1x6xf32>
                        scf.yield %42 : f32
                      } else {
                        %c0_9 = arith.constant 0 : index
                        %41 = tensor.dim %arg9, %c0_9 : tensor<1x6xf32>
                        %42 = arith.addi %38, %41 : index
                        %43 = arith.cmpi ult, %4, %42 : index
                        %44 = scf.if %43 -> (f32) {
                          %45 = arith.subi %4, %38 : index
                          %46 = tensor.extract %arg9[%45, %3] : tensor<1x6xf32>
                          scf.yield %46 : f32
                        } else {
                          %c0_10 = arith.constant 0 : index
                          %45 = tensor.dim %arg10, %c0_10 : tensor<1x6xf32>
                          %46 = arith.addi %42, %45 : index
                          %47 = arith.cmpi ult, %4, %46 : index
                          %48 = scf.if %47 -> (f32) {
                            %49 = arith.subi %4, %42 : index
                            %50 = tensor.extract %arg10[%49, %3] : tensor<1x6xf32>
                            scf.yield %50 : f32
                          } else {
                            %c0_11 = arith.constant 0 : index
                            %49 = tensor.dim %arg11, %c0_11 : tensor<1x6xf32>
                            %50 = arith.addi %46, %49 : index
                            %51 = arith.cmpi ult, %4, %50 : index
                            %52 = scf.if %51 -> (f32) {
                              %53 = arith.subi %4, %46 : index
                              %54 = tensor.extract %arg11[%53, %3] : tensor<1x6xf32>
                              scf.yield %54 : f32
                            } else {
                              %c0_12 = arith.constant 0 : index
                              %53 = tensor.dim %arg12, %c0_12 : tensor<1x6xf32>
                              %54 = arith.addi %50, %53 : index
                              %55 = arith.cmpi ult, %4, %54 : index
                              %56 = scf.if %55 -> (f32) {
                                %57 = arith.subi %4, %50 : index
                                %58 = tensor.extract %arg12[%57, %3] : tensor<1x6xf32>
                                scf.yield %58 : f32
                              } else {
                                %c0_13 = arith.constant 0 : index
                                %57 = tensor.dim %arg13, %c0_13 : tensor<1x6xf32>
                                %58 = arith.addi %54, %57 : index
                                %59 = arith.cmpi ult, %4, %58 : index
                                %60 = scf.if %59 -> (f32) {
                                  %61 = arith.subi %4, %54 : index
                                  %62 = tensor.extract %arg13[%61, %3] : tensor<1x6xf32>
                                  scf.yield %62 : f32
                                } else {
                                  %c0_14 = arith.constant 0 : index
                                  %61 = tensor.dim %arg14, %c0_14 : tensor<1x6xf32>
                                  %62 = arith.addi %58, %61 : index
                                  %63 = arith.cmpi ult, %4, %62 : index
                                  %64 = scf.if %63 -> (f32) {
                                    %65 = arith.subi %4, %58 : index
                                    %66 = tensor.extract %arg14[%65, %3] : tensor<1x6xf32>
                                    scf.yield %66 : f32
                                  } else {
                                    %65 = arith.subi %4, %62 : index
                                    %66 = tensor.extract %arg15[%65, %3] : tensor<1x6xf32>
                                    scf.yield %66 : f32
                                  }
                                  scf.yield %64 : f32
                                }
                                scf.yield %60 : f32
                              }
                              scf.yield %56 : f32
                            }
                            scf.yield %52 : f32
                          }
                          scf.yield %48 : f32
                        }
                        scf.yield %44 : f32
                      }
                      scf.yield %40 : f32
                    }
                    scf.yield %36 : f32
                  }
                  scf.yield %32 : f32
                }
                scf.yield %28 : f32
              }
              scf.yield %24 : f32
            }
            scf.yield %20 : f32
          }
          scf.yield %16 : f32
        }
        scf.yield %12 : f32
      }
      linalg.yield %8 : f32
    } -> tensor<16x6xf32>
    return %1 : tensor<16x6xf32>
  }
  func.func @main() {
    %0 = arith.constant dense<0.01>: tensor<1x6xf32>
    %1 = arith.constant dense<-0.001>: tensor<1x6xf32>
    %2 = arith.constant dense<0.02>: tensor<1x6xf32>
    %3 = arith.constant dense<0.001>: tensor<1x6xf32>
    %4 = arith.constant dense<0.03>: tensor<1x6xf32>
    %5 = arith.constant dense<-0.001>: tensor<1x6xf32>
    %6 = arith.constant dense<-0.04>: tensor<1x6xf32>
    %7 = arith.constant dense<0.001>: tensor<1x6xf32>
    %8 = arith.constant dense<-0.05>: tensor<1x6xf32>
    %9 = arith.constant dense<-0.001>: tensor<1x6xf32>
    %10 = arith.constant dense<-0.06>: tensor<1x6xf32>
    %11 = arith.constant dense<0.001>: tensor<1x6xf32>
    %12 = arith.constant dense<0.02>: tensor<1x6xf32>
    %13 = arith.constant dense<-0.01>: tensor<1x6xf32>
    %14 = arith.constant dense<0.01>: tensor<1x6xf32>
    %15 = arith.constant dense<-0.02>: tensor<1x6xf32>
    %16 = func.call @callee(%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15) : (tensor<1x6xf32>, tensor<1x6xf32>, tensor<1x6xf32>, tensor<1x6xf32>, tensor<1x6xf32>, tensor<1x6xf32>, tensor<1x6xf32>, tensor<1x6xf32>, tensor<1x6xf32>, tensor<1x6xf32>, tensor<1x6xf32>, tensor<1x6xf32>, tensor<1x6xf32>, tensor<1x6xf32>, tensor<1x6xf32>, tensor<1x6xf32>) -> tensor<16x6xf32>
    %unranked = tensor.cast %16 : tensor<16x6xf32> to tensor<*xf32>
    func.call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
    //      CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 2 offset = 0 sizes = [16, 6] strides = [6, 1] data =
    //      CHECK: [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    // CHECK-NEXT: [-0.001, -0.001, -0.001, -0.001, -0.001, -0.001],
    // CHECK-NEXT: [0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
    // CHECK-NEXT: [0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
    // CHECK-NEXT: [0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
    // CHECK-NEXT: [-0.001, -0.001, -0.001, -0.001, -0.001, -0.001],
    // CHECK-NEXT: [-0.04, -0.04, -0.04, -0.04, -0.04, -0.04],
    // CHECK-NEXT: [0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
    // CHECK-NEXT: [-0.05, -0.05, -0.05, -0.05, -0.05, -0.05],
    // CHECK-NEXT: [-0.001, -0.001, -0.001, -0.001, -0.001, -0.001],
    // CHECK-NEXT: [-0.06, -0.06, -0.06, -0.06, -0.06, -0.06],
    // CHECK-NEXT: [0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
    // CHECK-NEXT: [0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
    // CHECK-NEXT: [-0.01, -0.01, -0.01, -0.01, -0.01, -0.01],
    // CHECK-NEXT: [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    // CHECK-NEXT: [-0.02, -0.02, -0.02, -0.02, -0.02, -0.02]
    return
  }
}
