// RUN: imex-opt %s -overlap-comm-and-compute | FileCheck %s

module {
  func.func @test() { // (!ndarray.ndarray<?x?xf64>, !ndarray.ndarray<?x?xf64>, !ndarray.ndarray<?x?xf64>, memref<2xindex>) attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c34 = arith.constant 34 : index
    %c96 = arith.constant 96 : index
    %c32 = arith.constant 32 : index
    %c100 = arith.constant 100 : index
    %c2 = arith.constant 2 : index
    %c38 = arith.constant 38 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 9.000000e+00 : f64
    %0 = ndarray.create %c34, %c100   value %cst {dtype = 0 : i8} : (index, index, f64) -> !ndarray.ndarray<34x100xf64>
    %1 = ndarray.cast %0 : !ndarray.ndarray<34x100xf64> to !ndarray.ndarray<?x?xf64>
    %2 = ndarray.create %c0, %c0   {dtype = 0 : i8} : (index, index) -> !ndarray.ndarray<0x0xf64>
    %3 = ndarray.cast %2 : !ndarray.ndarray<0x0xf64> to !ndarray.ndarray<?x?xf64>
    %handle, %lHalo, %rHalo = "distruntime.get_halo"(%1, %c100, %c100, %c34, %c0, %c32, %c0, %c38, %c100) {team = 22} : (!ndarray.ndarray<?x?xf64>, index, index, index, index, index, index, index, index) -> (!distruntime.asynchandle, !ndarray.ndarray<2x100xf64>, !ndarray.ndarray<2x100xf64>)
    %lHaloCasted = ndarray.cast %lHalo : !ndarray.ndarray<2x100xf64> to !ndarray.ndarray<2x100xf64>
    %4 = ndarray.subview %0[0, 0] [34, 100] [1, 1] : !ndarray.ndarray<34x100xf64> to !ndarray.ndarray<34x100xf64>
    "distruntime.wait"(%handle) : (!distruntime.asynchandle) -> ()
    %5 = ndarray.subview %lHaloCasted[0, 0] [2, 96] [1, 1] : !ndarray.ndarray<2x100xf64> to !ndarray.ndarray<2x96xf64>
    %6 = ndarray.subview %4[0, 0] [32, 96] [1, 1] : !ndarray.ndarray<34x100xf64> to !ndarray.ndarray<32x96xf64>
    %7 = ndarray.subview %4[0, 2] [34, 96] [1, 1] : !ndarray.ndarray<34x100xf64> to !ndarray.ndarray<34x96xf64>
    %8 = ndarray.subview %4[2, 4] [32, 96] [1, 1] : !ndarray.ndarray<34x100xf64> to !ndarray.ndarray<32x96xf64>
    %9 = ndarray.subview %rHalo[0, 4] [2, 96] [1, 1] : !ndarray.ndarray<2x100xf64> to !ndarray.ndarray<2x96xf64>
    %10 = ndarray.create %c34, %c96   {dtype = 0 : i8} : (index, index) -> !ndarray.ndarray<34x96xf64>
    %11 = ndarray.extract_slice %6[0, 0] [30, 96] [1, 1] : !ndarray.ndarray<32x96xf64> to !ndarray.ndarray<30x96xf64>
    %12 = ndarray.extract_slice %7[2, 0] [30, 96] [1, 1] : !ndarray.ndarray<34x96xf64> to !ndarray.ndarray<30x96xf64>
    %13 = ndarray.ewbin %11, %12 {op = 0 : i32} : (!ndarray.ndarray<30x96xf64>, !ndarray.ndarray<30x96xf64>) -> !ndarray.ndarray<30x96xf64>
    %14 = ndarray.extract_slice %5[0, 0] [2, 96] [1, 1] : !ndarray.ndarray<2x96xf64> to !ndarray.ndarray<2x96xf64>
    %15 = ndarray.extract_slice %7[0, 0] [2, 96] [1, 1] : !ndarray.ndarray<34x96xf64> to !ndarray.ndarray<2x96xf64>
    %16 = ndarray.ewbin %14, %15 {op = 0 : i32} : (!ndarray.ndarray<2x96xf64>, !ndarray.ndarray<2x96xf64>) -> !ndarray.ndarray<2x96xf64>
    %17 = ndarray.extract_slice %6[30, 0] [2, 96] [1, 1] : !ndarray.ndarray<32x96xf64> to !ndarray.ndarray<2x96xf64>
    %18 = ndarray.extract_slice %7[32, 0] [2, 96] [1, 1] : !ndarray.ndarray<34x96xf64> to !ndarray.ndarray<2x96xf64>
    %19 = ndarray.ewbin %17, %18 {op = 0 : i32} : (!ndarray.ndarray<2x96xf64>, !ndarray.ndarray<2x96xf64>) -> !ndarray.ndarray<2x96xf64>
    %20 = ndarray.create %c0, %c0   {dtype = 0 : i8} : (index, index) -> !ndarray.ndarray<0x0xf64>
    %21 = ndarray.create %c34, %c96   {dtype = 0 : i8} : (index, index) -> !ndarray.ndarray<34x96xf64>
    %22 = ndarray.extract_slice %8[2, 0] [30, 96] [1, 1] : !ndarray.ndarray<32x96xf64> to !ndarray.ndarray<30x96xf64>
    %23 = ndarray.ewbin %13, %22 {op = 0 : i32} : (!ndarray.ndarray<30x96xf64>, !ndarray.ndarray<30x96xf64>) -> !ndarray.ndarray<30x96xf64>
    %24 = ndarray.immutable_insert_slice %23 into %21 [2, 0] [30, 96] [1, 1] : !ndarray.ndarray<30x96xf64> into !ndarray.ndarray<34x96xf64>
    %25 = ndarray.extract_slice %8[0, 0] [2, 96] [1, 1] : !ndarray.ndarray<32x96xf64> to !ndarray.ndarray<2x96xf64>
    %26 = ndarray.ewbin %16, %25 {op = 0 : i32} : (!ndarray.ndarray<2x96xf64>, !ndarray.ndarray<2x96xf64>) -> !ndarray.ndarray<2x96xf64>
    %27 = ndarray.immutable_insert_slice %26 into %24 [0, 0] [2, 96] [1, 1] : !ndarray.ndarray<2x96xf64> into !ndarray.ndarray<34x96xf64>
    %28 = ndarray.extract_slice %9[0, 0] [2, 96] [1, 1] : !ndarray.ndarray<2x96xf64> to !ndarray.ndarray<2x96xf64>
    %29 = ndarray.ewbin %19, %28 {op = 0 : i32} : (!ndarray.ndarray<2x96xf64>, !ndarray.ndarray<2x96xf64>) -> !ndarray.ndarray<2x96xf64>
    %30 = ndarray.immutable_insert_slice %29 into %27 [32, 0] [2, 96] [1, 1] : !ndarray.ndarray<2x96xf64> into !ndarray.ndarray<34x96xf64>
    return
  }
}

// CHECK-LABEL: func.func @test()
// CHECK: [[handle:%.*]], [[lHalo:%.*]], [[rHalo:%.*]] = "distruntime.get_halo"
// CHECK: [[lHaloCast:%.*]] = ndarray.cast [[lHalo]]
// CHECK: ndarray.ewbin
// CHECK-SAME: {op = 0 : i32} : (!ndarray.ndarray<30x96xf64>, !ndarray.ndarray<30x96xf64>) -> !ndarray.ndarray<30x96xf64>
// CHECK: ndarray.ewbin
// CHECK-SAME: {op = 0 : i32} : (!ndarray.ndarray<30x96xf64>, !ndarray.ndarray<30x96xf64>) -> !ndarray.ndarray<30x96xf64>
// CHECK: "distruntime.wait"([[handle]]) : (!distruntime.asynchandle) -> ()
// CHECK-NEXT: ndarray.subview [[lHaloCast]]
// CHECK: ndarray.ewbin
// CHECK: ndarray.ewbin
// CHECK: ndarray.ewbin
// CHECK: ndarray.subview [[rHalo]]
// CHECK: ndarray.ewbin
