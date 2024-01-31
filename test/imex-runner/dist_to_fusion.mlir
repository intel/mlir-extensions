// RUN: DNDA_NPROCS=3 DNDA_PRANK=1 %python_executable %imex_runner -i %s -f %p/distfusion.pp -n --filecheck

module {
  func.func @ddpt_jit(%arg0: !ndarray.ndarray<512x512xf32, #dist.dist_env<team = 22 loffs = 170,0 lparts = 0x0,171x512,0x0>, #region.gpu_env<device = "XeGPU">>,
                      %arg1: !ndarray.ndarray<5x5xf32, #dist.dist_env<team = 22 loffs = 1,0 lparts = 0x0,2x5,0x0>, #region.gpu_env<device = "XeGPU">>,
                      %arg2: !ndarray.ndarray<512x512xf32, #dist.dist_env<team = 22 loffs = 170,0 lparts = 0x0,171x512,0x0>, #region.gpu_env<device = "XeGPU">>) attributes {llvm.emit_c_interface} {
    %cst = arith.constant 1.000000e+00 : f32
    %c508 = arith.constant 508 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = ndarray.subview %arg0[2, 2] [508, 508] [1, 1] : !ndarray.ndarray<512x512xf32, #dist.dist_env<team = 22 loffs = 170,0 lparts = 0x0,171x512,0x0>, #region.gpu_env<device = "XeGPU">> to !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>
    %1 = ndarray.subview %arg1[2, 2] [1, 1] [1, 1] : !ndarray.ndarray<5x5xf32, #dist.dist_env<team = 22 loffs = 1,0 lparts = 0x0,2x5,0x0>, #region.gpu_env<device = "XeGPU">> to !ndarray.ndarray<1x1xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>
    %2 = ndarray.subview %arg2[2, 2] [508, 508] [1, 1] : !ndarray.ndarray<512x512xf32, #dist.dist_env<team = 22 loffs = 170,0 lparts = 0x0,171x512,0x0>, #region.gpu_env<device = "XeGPU">> to !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>
    %3 = ndarray.ewbin %1, %2 {op = 21 : i32} : (!ndarray.ndarray<1x1xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>, !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>) -> !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>
    %4 = ndarray.subview %arg1[2, 0] [1, 1] [1, 1] : !ndarray.ndarray<5x5xf32, #dist.dist_env<team = 22 loffs = 1,0 lparts = 0x0,2x5,0x0>, #region.gpu_env<device = "XeGPU">> to !ndarray.ndarray<1x1xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>
    %5 = ndarray.subview %arg2[2, 0] [508, 508] [1, 1] : !ndarray.ndarray<512x512xf32, #dist.dist_env<team = 22 loffs = 170,0 lparts = 0x0,171x512,0x0>, #region.gpu_env<device = "XeGPU">> to !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>
    %6 = ndarray.ewbin %4, %5 {op = 21 : i32} : (!ndarray.ndarray<1x1xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>, !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>) -> !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>
    %7 = ndarray.ewbin %3, %6 {op = 0 : i32} : (!ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>, !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>) -> !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>
    %8 = ndarray.subview %arg1[2, 1] [1, 1] [1, 1] : !ndarray.ndarray<5x5xf32, #dist.dist_env<team = 22 loffs = 1,0 lparts = 0x0,2x5,0x0>, #region.gpu_env<device = "XeGPU">> to !ndarray.ndarray<1x1xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>
    %9 = ndarray.subview %arg2[2, 1] [508, 508] [1, 1] : !ndarray.ndarray<512x512xf32, #dist.dist_env<team = 22 loffs = 170,0 lparts = 0x0,171x512,0x0>, #region.gpu_env<device = "XeGPU">> to !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>
    %10 = ndarray.ewbin %8, %9 {op = 21 : i32} : (!ndarray.ndarray<1x1xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>, !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>) -> !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>
    %11 = ndarray.ewbin %7, %10 {op = 0 : i32} : (!ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>, !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>) -> !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>
    %12 = ndarray.subview %arg1[2, 3] [1, 1] [1, 1] : !ndarray.ndarray<5x5xf32, #dist.dist_env<team = 22 loffs = 1,0 lparts = 0x0,2x5,0x0>, #region.gpu_env<device = "XeGPU">> to !ndarray.ndarray<1x1xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>
    %13 = ndarray.subview %arg2[2, 3] [508, 508] [1, 1] : !ndarray.ndarray<512x512xf32, #dist.dist_env<team = 22 loffs = 170,0 lparts = 0x0,171x512,0x0>, #region.gpu_env<device = "XeGPU">> to !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>
    %14 = ndarray.ewbin %12, %13 {op = 21 : i32} : (!ndarray.ndarray<1x1xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>, !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>) -> !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>
    %15 = ndarray.ewbin %11, %14 {op = 0 : i32} : (!ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>, !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>) -> !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>
    %16 = ndarray.subview %arg1[2, 4] [1, 1] [1, 1] : !ndarray.ndarray<5x5xf32, #dist.dist_env<team = 22 loffs = 1,0 lparts = 0x0,2x5,0x0>, #region.gpu_env<device = "XeGPU">> to !ndarray.ndarray<1x1xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>
    %17 = ndarray.subview %arg2[2, 4] [508, 508] [1, 1] : !ndarray.ndarray<512x512xf32, #dist.dist_env<team = 22 loffs = 170,0 lparts = 0x0,171x512,0x0>, #region.gpu_env<device = "XeGPU">> to !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>
    %18 = ndarray.ewbin %16, %17 {op = 21 : i32} : (!ndarray.ndarray<1x1xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>, !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>) -> !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>
    %19 = ndarray.ewbin %15, %18 {op = 0 : i32} : (!ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>, !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>) -> !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>
    %20 = ndarray.subview %arg1[0, 2] [1, 1] [1, 1] : !ndarray.ndarray<5x5xf32, #dist.dist_env<team = 22 loffs = 1,0 lparts = 0x0,2x5,0x0>, #region.gpu_env<device = "XeGPU">> to !ndarray.ndarray<1x1xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>
    %21 = ndarray.subview %arg2[0, 2] [508, 508] [1, 1] : !ndarray.ndarray<512x512xf32, #dist.dist_env<team = 22 loffs = 170,0 lparts = 0x0,171x512,0x0>, #region.gpu_env<device = "XeGPU">> to !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>
    %22 = ndarray.ewbin %20, %21 {op = 21 : i32} : (!ndarray.ndarray<1x1xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>, !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>) -> !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>
    %23 = ndarray.ewbin %19, %22 {op = 0 : i32} : (!ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>, !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>) -> !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>
    %24 = ndarray.subview %arg1[1, 2] [1, 1] [1, 1] : !ndarray.ndarray<5x5xf32, #dist.dist_env<team = 22 loffs = 1,0 lparts = 0x0,2x5,0x0>, #region.gpu_env<device = "XeGPU">> to !ndarray.ndarray<1x1xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>
    %25 = ndarray.subview %arg2[1, 2] [508, 508] [1, 1] : !ndarray.ndarray<512x512xf32, #dist.dist_env<team = 22 loffs = 170,0 lparts = 0x0,171x512,0x0>, #region.gpu_env<device = "XeGPU">> to !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>
    %26 = ndarray.ewbin %24, %25 {op = 21 : i32} : (!ndarray.ndarray<1x1xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>, !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>) -> !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>
    %27 = ndarray.ewbin %23, %26 {op = 0 : i32} : (!ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>, !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>) -> !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>
    %28 = ndarray.subview %arg1[3, 2] [1, 1] [1, 1] : !ndarray.ndarray<5x5xf32, #dist.dist_env<team = 22 loffs = 1,0 lparts = 0x0,2x5,0x0>, #region.gpu_env<device = "XeGPU">> to !ndarray.ndarray<1x1xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>
    %29 = ndarray.subview %arg2[3, 2] [508, 508] [1, 1] : !ndarray.ndarray<512x512xf32, #dist.dist_env<team = 22 loffs = 170,0 lparts = 0x0,171x512,0x0>, #region.gpu_env<device = "XeGPU">> to !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>
    %30 = ndarray.ewbin %28, %29 {op = 21 : i32} : (!ndarray.ndarray<1x1xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>, !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>) -> !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>
    %31 = ndarray.ewbin %27, %30 {op = 0 : i32} : (!ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>, !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>) -> !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>
    %32 = ndarray.subview %arg1[4, 2] [1, 1] [1, 1] : !ndarray.ndarray<5x5xf32, #dist.dist_env<team = 22 loffs = 1,0 lparts = 0x0,2x5,0x0>, #region.gpu_env<device = "XeGPU">> to !ndarray.ndarray<1x1xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>
    %33 = ndarray.subview %arg2[4, 2] [508, 508] [1, 1] : !ndarray.ndarray<512x512xf32, #dist.dist_env<team = 22 loffs = 170,0 lparts = 0x0,171x512,0x0>, #region.gpu_env<device = "XeGPU">> to !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>
    %34 = ndarray.ewbin %32, %33 {op = 21 : i32} : (!ndarray.ndarray<1x1xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>, !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>) -> !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>
    %35 = ndarray.ewbin %31, %34 {op = 0 : i32} : (!ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>, !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>) -> !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>
    %36 = ndarray.ewbin %0, %35 {op = 0 : i32} : (!ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>, !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>) -> !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>
    ndarray.insert_slice %36 into %arg0[%c2, %c2] [%c508, %c508] [%c1, %c1] : !ndarray.ndarray<508x508xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">> into !ndarray.ndarray<512x512xf32, #dist.dist_env<team = 22 loffs = 170,0 lparts = 0x0,171x512,0x0>, #region.gpu_env<device = "XeGPU">>
    %37 = ndarray.create value %cst {team = 94136931182224 : i64, dtype = 1 : i8, device = "XeGPU"} : (f32) -> !ndarray.ndarray<f32, #dist.dist_env<team = 22>, #region.gpu_env<device = "XeGPU">>
    %38 = ndarray.ewbin %arg2, %37 {op = 0 : i32} : (!ndarray.ndarray<512x512xf32, #dist.dist_env<team = 22 loffs = 170,0 lparts = 0x0,171x512,0x0>, #region.gpu_env<device = "XeGPU">>, !ndarray.ndarray<f32, #dist.dist_env<team = 22>, #region.gpu_env<device = "XeGPU">>) -> !ndarray.ndarray<512x512xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">>
    %dim = ndarray.dim %arg2 %c0 : !ndarray.ndarray<512x512xf32, #dist.dist_env<team = 22 loffs = 170,0 lparts = 0x0,171x512,0x0>, #region.gpu_env<device = "XeGPU">> -> index
    %dim_0 = ndarray.dim %arg2 %c1 : !ndarray.ndarray<512x512xf32, #dist.dist_env<team = 22 loffs = 170,0 lparts = 0x0,171x512,0x0>, #region.gpu_env<device = "XeGPU">> -> index
    ndarray.insert_slice %38 into %arg2[%c0, %c0] [%dim, %dim_0] [%c1, %c1] : !ndarray.ndarray<512x512xf32, #dist.dist_env<team = 22 loffs = ?,? lparts = ?x?,?x?,?x?>, #region.gpu_env<device = "XeGPU">> into !ndarray.ndarray<512x512xf32, #dist.dist_env<team = 22 loffs = 170,0 lparts = 0x0,171x512,0x0>, #region.gpu_env<device = "XeGPU">>
    return
  }
}
// CHECK-LABEL: func.func @ddpt_jit
// CHECK: call @_idtr_update_halo_f32
// CHECK: call @_idtr_wait_f32
// CHECK: call @_idtr_update_halo_f32
// CHECK: call @_idtr_update_halo_f32
// CHECK: call @_idtr_wait_f32
// CHECK-LABEL: linalg.generic
// CHECK-NEXT: ^bb0
// CHECK-NEXT-COUNT-9: arith.mulf
// CHECK-COUNT-9: arith.addf
// CHECK-NEXT: linalg.yield
// CHECK-NEXT: } -> tensor<167x508xf32>
// CHECK: call @_idtr_wait_f32
// CHECK-LABEL: linalg.generic
// CHECK-NEXT: ^bb0
// CHECK-NEXT: arith.mulf
// CHECK-NEXT: linalg.yield
// CHECK-NEXT: } -> tensor<1x508xf32>
// CHECK-LABEL: linalg.generic
// CHECK-NEXT: ^bb0
// CHECK-NEXT: arith.mulf
// CHECK-NEXT: linalg.yield
// CHECK-NEXT: } -> tensor<1x508xf32>
// CHECK-LABEL: linalg.generic
// CHECK-NEXT: ^bb0
// CHECK-NEXT-COUNT-8: arith.mulf
// CHECK-COUNT-9: arith.addf
// CHECK-NEXT: linalg.yield
// CHECK-NEXT: } -> tensor<2x508xf32>
// CHECK-LABEL: linalg.generic
// CHECK-NEXT: ^bb0
// CHECK-NEXT: arith.mulf
// CHECK-NEXT: linalg.yield
// CHECK-NEXT: } -> tensor<1x508xf32>
// CHECK-LABEL: linalg.generic
// CHECK-NEXT: ^bb0
// CHECK-NEXT: arith.mulf
// CHECK-NEXT: linalg.yield
// CHECK-NEXT: } -> tensor<1x508xf32>
// CHECK-LABEL: linalg.generic
// CHECK-NEXT: ^bb0
// CHECK-NEXT-COUNT-8: arith.mulf
// CHECK-COUNT-9: arith.addf
// CHECK-NEXT: linalg.yield
// CHECK-NEXT: } -> tensor<2x508xf32>
// CHECK-LABEL: linalg.generic
// CHECK-NEXT: ^bb0
// CHECK-NEXT: arith.addf
// CHECK-NEXT: linalg.yield
// CHECK-NEXT: } -> tensor<171x512xf32>
// CHECK: return
