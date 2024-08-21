// RUN: imex-opt --split-input-file --xetile-init-duplicate --xetile-blocking \
// RUN: --cse --convert-xetile-to-xegpu --cse %s -verify-diagnostics -o -| FileCheck %s
module {
  gpu.module @test_kernel {

    //CHECK: gpu.func @inner_reduction(%[[arg0:.*]]: memref<128x256xf16>, %[[arg1:.*]]: memref<128x256xf16>) {
    gpu.func @inner_reduction(%a: memref<128x256xf16>, %b: memref<128x256xf16>) {
      //CHECK: %[[cst:.*]] = arith.constant dense<0.000000e+00> : vector<16xf16>
      //CHECK: %[[c0:.*]] = arith.constant 0 : index
      %c0 = arith.constant 0 : index
      %acc = arith.constant dense<0.0> : vector<16xf16>
      //CHECK: %[[R0:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[c0]], %[[c0]]] : memref<128x256xf16>
      //CHECK-SAME: !xegpu.tensor_desc<16x32xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
      %t = xetile.init_tile %a[%c0, %c0] : memref<128x256xf16> -> !xetile.tile<16x32xf16>
      //CHECK: %[[R1:.*]] = xegpu.load_nd %[[R0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}>
      //CHECK-SAME : !xegpu.tensor_desc<16x32xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>> -> vector<16x32xf16>
      %v = xetile.load_tile %t : !xetile.tile<16x32xf16> -> vector<16x32xf16>

      //CHECK: %[[R2:.*]] = vector.extract_strided_slice %[[R1]] {offsets = [0, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
      //CHECK: %[[R3:.*]] = vector.extract_strided_slice %[[R1]] {offsets = [1, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
      //CHECK: %[[R4:.*]] = vector.extract_strided_slice %[[R1]] {offsets = [2, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
      //CHECK: %[[R5:.*]] = vector.extract_strided_slice %[[R1]] {offsets = [3, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
      //CHECK: %[[R6:.*]] = vector.extract_strided_slice %[[R1]] {offsets = [4, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
      //CHECK: %[[R7:.*]] = vector.extract_strided_slice %[[R1]] {offsets = [5, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
      //CHECK: %[[R8:.*]] = vector.extract_strided_slice %[[R1]] {offsets = [6, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
      //CHECK: %[[R9:.*]] = vector.extract_strided_slice %[[R1]] {offsets = [7, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
      //CHECK: %[[R10:.*]] = vector.extract_strided_slice %[[R1]] {offsets = [8, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
      //CHECK: %[[R11:.*]] = vector.extract_strided_slice %[[R1]] {offsets = [9, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
      //CHECK: %[[R12:.*]] = vector.extract_strided_slice %[[R1]] {offsets = [10, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
      //CHECK: %[[R13:.*]] = vector.extract_strided_slice %[[R1]] {offsets = [11, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
      //CHECK: %[[R14:.*]] = vector.extract_strided_slice %[[R1]] {offsets = [12, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
      //CHECK: %[[R15:.*]] = vector.extract_strided_slice %[[R1]] {offsets = [13, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
      //CHECK: %[[R16:.*]] = vector.extract_strided_slice %[[R1]] {offsets = [14, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
      //CHECK: %[[R17:.*]] = vector.extract_strided_slice %[[R1]] {offsets = [15, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
      //CHECK: %[[R18:.*]] = vector.extract_strided_slice %[[R2]] {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R19:.*]] = vector.extract_strided_slice %[[R2]] {offsets = [0, 16], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R20:.*]] = vector.extract_strided_slice %[[R3]] {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R21:.*]] = vector.extract_strided_slice %[[R3]] {offsets = [0, 16], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R22:.*]] = vector.extract_strided_slice %[[R4]] {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R23:.*]] = vector.extract_strided_slice %[[R4]] {offsets = [0, 16], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R24:.*]] = vector.extract_strided_slice %[[R5]] {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R25:.*]] = vector.extract_strided_slice %[[R5]] {offsets = [0, 16], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R26:.*]] = vector.extract_strided_slice %[[R6]] {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R27:.*]] = vector.extract_strided_slice %[[R6]] {offsets = [0, 16], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R28:.*]] = vector.extract_strided_slice %[[R7]] {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R29:.*]] = vector.extract_strided_slice %[[R7]] {offsets = [0, 16], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R30:.*]] = vector.extract_strided_slice %[[R8]] {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R31:.*]] = vector.extract_strided_slice %[[R8]] {offsets = [0, 16], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R32:.*]] = vector.extract_strided_slice %[[R9]] {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R33:.*]] = vector.extract_strided_slice %[[R9]] {offsets = [0, 16], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R34:.*]] = vector.extract_strided_slice %[[R10]] {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R35:.*]] = vector.extract_strided_slice %[[R10]] {offsets = [0, 16], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R36:.*]] = vector.extract_strided_slice %[[R11]] {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R37:.*]] = vector.extract_strided_slice %[[R11]] {offsets = [0, 16], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R38:.*]] = vector.extract_strided_slice %[[R12]] {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R39:.*]] = vector.extract_strided_slice %[[R12]] {offsets = [0, 16], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R40:.*]] = vector.extract_strided_slice %[[R13]] {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R41:.*]] = vector.extract_strided_slice %[[R13]] {offsets = [0, 16], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R42:.*]] = vector.extract_strided_slice %[[R14]] {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R43:.*]] = vector.extract_strided_slice %[[R14]] {offsets = [0, 16], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R44:.*]] = vector.extract_strided_slice %[[R15]] {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R45:.*]] = vector.extract_strided_slice %[[R15]] {offsets = [0, 16], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R46:.*]] = vector.extract_strided_slice %[[R16]] {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R47:.*]] = vector.extract_strided_slice %[[R16]] {offsets = [0, 16], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R48:.*]] = vector.extract_strided_slice %[[R17]] {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R49:.*]] = vector.extract_strided_slice %[[R17]] {offsets = [0, 16], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R50:.*]] = math.exp %[[R18]] : vector<1x16xf16>
      //CHECK: %[[R51:.*]] = math.exp %[[R19]] : vector<1x16xf16>
      //CHECK: %[[R52:.*]] = math.exp %[[R20]] : vector<1x16xf16>
      //CHECK: %[[R53:.*]] = math.exp %[[R21]] : vector<1x16xf16>
      //CHECK: %[[R54:.*]] = math.exp %[[R22]] : vector<1x16xf16>
      //CHECK: %[[R55:.*]] = math.exp %[[R23]] : vector<1x16xf16>
      //CHECK: %[[R56:.*]] = math.exp %[[R24]] : vector<1x16xf16>
      //CHECK: %[[R57:.*]] = math.exp %[[R25]] : vector<1x16xf16>
      //CHECK: %[[R58:.*]] = math.exp %[[R26]] : vector<1x16xf16>
      //CHECK: %[[R59:.*]] = math.exp %[[R27]] : vector<1x16xf16>
      //CHECK: %[[R60:.*]] = math.exp %[[R28]] : vector<1x16xf16>
      //CHECK: %[[R61:.*]] = math.exp %[[R29]] : vector<1x16xf16>
      //CHECK: %[[R62:.*]] = math.exp %[[R30]] : vector<1x16xf16>
      //CHECK: %[[R63:.*]] = math.exp %[[R31]] : vector<1x16xf16>
      //CHECK: %[[R64:.*]] = math.exp %[[R32]] : vector<1x16xf16>
      //CHECK: %[[R65:.*]] = math.exp %[[R33]] : vector<1x16xf16>
      //CHECK: %[[R66:.*]] = math.exp %[[R34]] : vector<1x16xf16>
      //CHECK: %[[R67:.*]] = math.exp %[[R35]] : vector<1x16xf16>
      //CHECK: %[[R68:.*]] = math.exp %[[R36]] : vector<1x16xf16>
      //CHECK: %[[R69:.*]] = math.exp %[[R37]] : vector<1x16xf16>
      //CHECK: %[[R70:.*]] = math.exp %[[R38]] : vector<1x16xf16>
      //CHECK: %[[R71:.*]] = math.exp %[[R39]] : vector<1x16xf16>
      //CHECK: %[[R72:.*]] = math.exp %[[R40]] : vector<1x16xf16>
      //CHECK: %[[R73:.*]] = math.exp %[[R41]] : vector<1x16xf16>
      //CHECK: %[[R74:.*]] = math.exp %[[R42]] : vector<1x16xf16>
      //CHECK: %[[R75:.*]] = math.exp %[[R43]] : vector<1x16xf16>
      //CHECK: %[[R76:.*]] = math.exp %[[R44]] : vector<1x16xf16>
      //CHECK: %[[R77:.*]] = math.exp %[[R45]] : vector<1x16xf16>
      //CHECK: %[[R78:.*]] = math.exp %[[R46]] : vector<1x16xf16>
      //CHECK: %[[R79:.*]] = math.exp %[[R47]] : vector<1x16xf16>
      //CHECK: %[[R80:.*]] = math.exp %[[R48]] : vector<1x16xf16>
      //CHECK: %[[R81:.*]] = math.exp %[[R49]] : vector<1x16xf16>
      %e = math.exp %v: vector<16x32xf16>
      //CHECK: %[[R82:.*]] = arith.addf %[[R50]], %[[R51]] : vector<1x16xf16>
      //CHECK: %[[R83:.*]] = vector.shape_cast %[[R82]] : vector<1x16xf16> to vector<16xf16>
      //CHECK: %[[R84:.*]] = arith.addf %[[R52]], %[[R53]] : vector<1x16xf16>
      //CHECK: %[[R85:.*]] = vector.shape_cast %[[R84]] : vector<1x16xf16> to vector<16xf16>
      //CHECK: %[[R86:.*]] = arith.addf %[[R54]], %[[R55]] : vector<1x16xf16>
      //CHECK: %[[R87:.*]] = vector.shape_cast %[[R86]] : vector<1x16xf16> to vector<16xf16>
      //CHECK: %[[R88:.*]] = arith.addf %[[R56]], %[[R57]] : vector<1x16xf16>
      //CHECK: %[[R89:.*]] = vector.shape_cast %[[R88]] : vector<1x16xf16> to vector<16xf16>
      //CHECK: %[[R90:.*]] = arith.addf %[[R58]], %[[R59]] : vector<1x16xf16>
      //CHECK: %[[R91:.*]] = vector.shape_cast %[[R90]] : vector<1x16xf16> to vector<16xf16>
      //CHECK: %[[R92:.*]] = arith.addf %[[R60]], %[[R61]] : vector<1x16xf16>
      //CHECK: %[[R93:.*]] = vector.shape_cast %[[R92]] : vector<1x16xf16> to vector<16xf16>
      //CHECK: %[[R94:.*]] = arith.addf %[[R62]], %[[R63]] : vector<1x16xf16>
      //CHECK: %[[R95:.*]] = vector.shape_cast %[[R94]] : vector<1x16xf16> to vector<16xf16>
      //CHECK: %[[R96:.*]] = arith.addf %[[R64]], %[[R65]] : vector<1x16xf16>
      //CHECK: %[[R97:.*]] = vector.shape_cast %[[R96]] : vector<1x16xf16> to vector<16xf16>
      //CHECK: %[[R98:.*]] = arith.addf %[[R66]], %[[R67]] : vector<1x16xf16>
      //CHECK: %[[R99:.*]] = vector.shape_cast %[[R98]] : vector<1x16xf16> to vector<16xf16>
      //CHECK: %[[R100:.*]] = arith.addf %[[R68]], %[[R69]] : vector<1x16xf16>
      //CHECK: %[[R101:.*]] = vector.shape_cast %[[R100]] : vector<1x16xf16> to vector<16xf16>
      //CHECK: %[[R102:.*]] = arith.addf %[[R70]], %[[R71]] : vector<1x16xf16>
      //CHECK: %[[R103:.*]] = vector.shape_cast %[[R102]] : vector<1x16xf16> to vector<16xf16>
      //CHECK: %[[R104:.*]] = arith.addf %[[R72]], %[[R73]] : vector<1x16xf16>
      //CHECK: %[[R105:.*]] = vector.shape_cast %[[R104]] : vector<1x16xf16> to vector<16xf16>
      //CHECK: %[[R106:.*]] = arith.addf %[[R74]], %[[R75]] : vector<1x16xf16>
      //CHECK: %[[R107:.*]] = vector.shape_cast %[[R106]] : vector<1x16xf16> to vector<16xf16>
      //CHECK: %[[R108:.*]] = arith.addf %[[R76]], %[[R77]] : vector<1x16xf16>
      //CHECK: %[[R109:.*]] = vector.shape_cast %[[R108]] : vector<1x16xf16> to vector<16xf16>
      //CHECK: %[[R110:.*]] = arith.addf %[[R78]], %[[R79]] : vector<1x16xf16>
      //CHECK: %[[R111:.*]] = vector.shape_cast %[[R110]] : vector<1x16xf16> to vector<16xf16>
      //CHECK: %[[R112:.*]] = arith.addf %[[R80]], %[[R81]] : vector<1x16xf16>
      //CHECK: %[[R113:.*]] = vector.shape_cast %[[R112]] : vector<1x16xf16> to vector<16xf16>
      //CHECK: %[[R114:.*]] = vector.shuffle %[[R83]], %[[R85]] [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf16>, vector<16xf16>
      //CHECK: %[[R115:.*]] = vector.shuffle %[[R83]], %[[R85]] [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: %[[R116:.*]] = arith.addf %[[R114]], %[[R115]] : vector<16xf16>
      //CHECK: %[[R117:.*]] = vector.shuffle %[[R87]], %[[R89]] [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf16>, vector<16xf16>
      //CHECK: %[[R118:.*]] = vector.shuffle %[[R87]], %[[R89]] [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: %[[R119:.*]] = arith.addf %[[R117]], %[[R118]] : vector<16xf16>
      //CHECK: %[[R120:.*]] = vector.shuffle %[[R91]], %[[R93]] [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf16>, vector<16xf16>
      //CHECK: %[[R121:.*]] = vector.shuffle %[[R91]], %[[R93]] [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: %[[R122:.*]] = arith.addf %[[R120]], %[[R121]] : vector<16xf16>
      //CHECK: %[[R123:.*]] = vector.shuffle %[[R95]], %[[R97]] [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf16>, vector<16xf16>
      //CHECK: %[[R124:.*]] = vector.shuffle %[[R95]], %[[R97]] [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: %[[R125:.*]] = arith.addf %[[R123]], %[[R124]] : vector<16xf16>
      //CHECK: %[[R126:.*]] = vector.shuffle %[[R99]], %[[R101]] [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf16>, vector<16xf16>
      //CHECK: %[[R127:.*]] = vector.shuffle %[[R99]], %[[R101]] [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: %[[R128:.*]] = arith.addf %[[R126]], %[[R127]] : vector<16xf16>
      //CHECK: %[[R129:.*]] = vector.shuffle %[[R103]], %[[R105]] [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf16>, vector<16xf16>
      //CHECK: %[[R130:.*]] = vector.shuffle %[[R103]], %[[R105]] [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: %[[R131:.*]] = arith.addf %[[R129]], %[[R130]] : vector<16xf16>
      //CHECK: %[[R132:.*]] = vector.shuffle %[[R107]], %[[R109]] [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf16>, vector<16xf16>
      //CHECK: %[[R133:.*]] = vector.shuffle %[[R107]], %[[R109]] [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: %[[R134:.*]] = arith.addf %[[R132]], %[[R133]] : vector<16xf16>
      //CHECK: %[[R135:.*]] = vector.shuffle %[[R111]], %[[R113]] [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf16>, vector<16xf16>
      //CHECK: %[[R136:.*]] = vector.shuffle %[[R111]], %[[R113]] [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: %[[R137:.*]] = arith.addf %[[R135]], %[[R136]] : vector<16xf16>
      //CHECK: %[[R138:.*]] = vector.shuffle %[[R116]], %[[R119]] [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf16>, vector<16xf16>
      //CHECK: %[[R139:.*]] = vector.shuffle %[[R116]], %[[R119]] [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: %[[R140:.*]] = arith.addf %[[R138]], %[[R139]] : vector<16xf16>
      //CHECK: %[[R141:.*]] = vector.shuffle %[[R122]], %[[R125]] [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf16>, vector<16xf16>
      //CHECK: %[[R142:.*]] = vector.shuffle %[[R122]], %[[R125]] [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: %[[R143:.*]] = arith.addf %[[R141]], %[[R142]] : vector<16xf16>
      //CHECK: %[[R144:.*]] = vector.shuffle %[[R128]], %[[R131]] [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf16>, vector<16xf16>
      //CHECK: %[[R145:.*]] = vector.shuffle %[[R128]], %[[R131]] [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: %[[R146:.*]] = arith.addf %[[R144]], %[[R145]] : vector<16xf16>
      //CHECK: %[[R147:.*]] = vector.shuffle %[[R134]], %[[R137]] [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf16>, vector<16xf16>
      //CHECK: %[[R148:.*]] = vector.shuffle %[[R134]], %[[R137]] [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: %[[R149:.*]] = arith.addf %[[R147]], %[[R148]] : vector<16xf16>
      //CHECK: %[[R150:.*]] = vector.shuffle %[[R140]], %[[R143]] [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29] : vector<16xf16>, vector<16xf16>
      //CHECK: %[[R151:.*]] = vector.shuffle %[[R140]], %[[R143]] [2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: %[[R152:.*]] = arith.addf %[[R150]], %[[R151]] : vector<16xf16>
      //CHECK: %[[R153:.*]] = vector.shuffle %[[R146]], %[[R149]] [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29] : vector<16xf16>, vector<16xf16>
      //CHECK: %[[R154:.*]] = vector.shuffle %[[R146]], %[[R149]] [2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: %[[R155:.*]] = arith.addf %[[R153]], %[[R154]] : vector<16xf16>
      //CHECK: %[[R156:.*]] = vector.shuffle %[[R152]], %[[R155]] [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30] : vector<16xf16>, vector<16xf16>
      //CHECK: %[[R157:.*]] = vector.shuffle %[[R152]], %[[R155]] [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: %[[R158:.*]] = arith.addf %[[R156]], %[[R157]] : vector<16xf16>
      //CHECK: %[[R159:.*]] = vector.shape_cast %[[R158]] : vector<16xf16> to vector<16x1xf16>
      //CHECK: %[[R160:.*]] = vector.shape_cast %[[R159]] : vector<16x1xf16> to vector<16xf16>
      %r = vector.multi_reduction <add>, %e, %acc [1] : vector<16x32xf16> to vector<16xf16>
      //CHECK: %[[R161:.*]] = vector.shape_cast %[[R160]] : vector<16xf16> to vector<2x8xf16>
      %c = vector.shape_cast %r: vector<16xf16> to vector<2x8xf16>
      //CHECK: %[[R162:.*]] = xegpu.create_nd_tdesc %[[arg1]][%[[c0]], %[[c0]]] : memref<128x256xf16> -> !xegpu.tensor_desc<2x8xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
      %s = xetile.init_tile %b[%c0, %c0] : memref<128x256xf16> -> !xetile.tile<2x8xf16>
      //CHECK: xegpu.store_nd %[[R161]], %[[R162]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<2x8xf16>, !xegpu.tensor_desc<2x8xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
      xetile.store_tile %c, %s : vector<2x8xf16>, !xetile.tile<2x8xf16>
      gpu.return
    }

    gpu.func @inner_reduction_1(%a: memref<8x32xf32>, %b: memref<8x1xf32>) {
      %c0 = arith.constant 0 : index
      %neg_inf = arith.constant dense<0xFF800000> : vector<8xf32> // -inf

      %a_tile = xetile.init_tile %a[%c0, %c0] : memref<8x32xf32> -> !xetile.tile<8x32xf32>
      %b_tile = xetile.init_tile %b[%c0, %c0] : memref<8x1xf32> -> !xetile.tile<8x1xf32>

      //CHECK: xegpu.load_nd %{{.*}} <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>> -> vector<8x16xf32>
      //CHECK: xegpu.load_nd %{{.*}} <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>> -> vector<8x16xf32>
      %a_loaded = xetile.load_tile %a_tile: !xetile.tile<8x32xf32> -> vector<8x32xf32>

      //CHECK: %[[R1:.*]] = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf32>, vector<16xf32>
      //CHECK: %[[R2:.*]] = vector.shuffle %{{.*}}, %{{.*}} [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
      //CHECK: %[[R3:.*]] = arith.maximumf %[[R1]], %[[R2]] : vector<16xf32>
      //CHECK: %[[R4:.*]] = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf32>, vector<16xf32>
      //CHECK: %[[R5:.*]] = vector.shuffle %{{.*}}, %{{.*}} [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
      //CHECK: %[[R6:.*]] = arith.maximumf %[[R4]], %[[R5]] : vector<16xf32>
      //CHECK: %[[R7:.*]] = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf32>, vector<16xf32>
      //CHECK: %[[R8:.*]] = vector.shuffle %{{.*}}, %{{.*}} [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
      //CHECK: %[[R9:.*]] = arith.maximumf %[[R7]], %[[R8]] : vector<16xf32>
      //CHECK: %[[R10:.*]] = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf32>, vector<16xf32>
      //CHECK: %[[R11:.*]] = vector.shuffle %{{.*}}, %{{.*}} [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
      //CHECK: %[[R12:.*]] = arith.maximumf %[[R10]], %[[R11]] : vector<16xf32>
      //CHECK: %[[R13:.*]] = vector.shuffle %[[R3]], %[[R6]] [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
      //CHECK: %[[R14:.*]] = vector.shuffle %[[R3]], %[[R6]] [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
      //CHECK: %[[R15:.*]] = arith.maximumf %[[R13]], %[[R14]] : vector<16xf32>
      //CHECK: %[[R16:.*]] = vector.shuffle %[[R9]], %[[R12]] [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
      //CHECK: %[[R17:.*]] = vector.shuffle %[[R9]], %[[R12]] [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
      //CHECK: %[[R18:.*]] = arith.maximumf %[[R16]], %[[R17]] : vector<16xf32>
      //CHECK: %[[R19:.*]] = vector.shuffle %[[R15]], %[[R18]] [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29] : vector<16xf32>, vector<16xf32>
      //CHECK: %[[R20:.*]] = vector.shuffle %[[R15]], %[[R18]] [2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31] : vector<16xf32>, vector<16xf32>
      //CHECK: %[[R21:.*]] = arith.maximumf %[[R19]], %[[R20]] : vector<16xf32>
      //CHECK: %[[R22:.*]] = vector.shuffle %[[R21]], %[[R21]] [0, 2, 4, 6, 8, 10, 12, 14] : vector<16xf32>, vector<16xf32>
      //CHECK: %[[R23:.*]] = vector.shuffle %[[R21]], %[[R21]] [1, 3, 5, 7, 9, 11, 13, 15] : vector<16xf32>, vector<16xf32>
      //CHECK: %[[R24:.*]] = arith.maximumf %[[R22]], %[[R23]] : vector<8xf32>
      %3 = vector.multi_reduction <maximumf>, %a_loaded, %neg_inf [1] : vector<8x32xf32> to vector<8xf32> // fastmath<nnan> is implicit here
      %reduced = vector.shape_cast %3 : vector<8xf32> to vector<8x1xf32>
      xetile.store_tile %reduced, %b_tile : vector<8x1xf32>, !xetile.tile<8x1xf32>
      gpu.return
    }

    //CHECK: gpu.func @outter_reduction(%[[arg0:.*]]: memref<128x256xf16>, %[[arg1:.*]]: memref<128x256xf16>) {
    gpu.func @outter_reduction(%a: memref<128x256xf16>, %b: memref<128x256xf16>) {
      //CHECK: %[[cst:.*]] = arith.constant dense<0.000000e+00> : vector<32xf16>
      //CHECK: %[[c0:.*]] = arith.constant 0 : index
      %c0 = arith.constant 0 : index
      %acc = arith.constant dense<0.0> : vector<32xf16>
      //CHECK: %[[R0:.*]] = xegpu.create_nd_tdesc %[[arg0]][%[[c0]], %[[c0]]] : memref<128x256xf16>
      //CHECK-SAME: !xegpu.tensor_desc<16x32xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
      %t = xetile.init_tile %a[%c0, %c0] : memref<128x256xf16> -> !xetile.tile<16x32xf16>
      //CHECK: %[[R1:.*]] = xegpu.load_nd %[[R0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}>
      //CHECK-SAME: !xegpu.tensor_desc<16x32xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>> -> vector<16x32xf16>
      %v = xetile.load_tile %t : !xetile.tile<16x32xf16> -> vector<16x32xf16>
      //CHECK: %[[R2:.*]] = vector.extract_strided_slice %[[R1]] {offsets = [0, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
      //CHECK: %[[R3:.*]] = vector.extract_strided_slice %[[R1]] {offsets = [1, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
      //CHECK: %[[R4:.*]] = vector.extract_strided_slice %[[R1]] {offsets = [2, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
      //CHECK: %[[R5:.*]] = vector.extract_strided_slice %[[R1]] {offsets = [3, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
      //CHECK: %[[R6:.*]] = vector.extract_strided_slice %[[R1]] {offsets = [4, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
      //CHECK: %[[R7:.*]] = vector.extract_strided_slice %[[R1]] {offsets = [5, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
      //CHECK: %[[R8:.*]] = vector.extract_strided_slice %[[R1]] {offsets = [6, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
      //CHECK: %[[R9:.*]] = vector.extract_strided_slice %[[R1]] {offsets = [7, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
      //CHECK: %[[R10:.*]] = vector.extract_strided_slice %[[R1]] {offsets = [8, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
      //CHECK: %[[R11:.*]] = vector.extract_strided_slice %[[R1]] {offsets = [9, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
      //CHECK: %[[R12:.*]] = vector.extract_strided_slice %[[R1]] {offsets = [10, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
      //CHECK: %[[R13:.*]] = vector.extract_strided_slice %[[R1]] {offsets = [11, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
      //CHECK: %[[R14:.*]] = vector.extract_strided_slice %[[R1]] {offsets = [12, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
      //CHECK: %[[R15:.*]] = vector.extract_strided_slice %[[R1]] {offsets = [13, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
      //CHECK: %[[R16:.*]] = vector.extract_strided_slice %[[R1]] {offsets = [14, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
      //CHECK: %[[R17:.*]] = vector.extract_strided_slice %[[R1]] {offsets = [15, 0], sizes = [1, 32], strides = [1, 1]} : vector<16x32xf16> to vector<1x32xf16>
      //CHECK: %[[R18:.*]] = vector.extract_strided_slice %[[R2]] {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R19:.*]] = vector.extract_strided_slice %[[R2]] {offsets = [0, 16], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R20:.*]] = vector.extract_strided_slice %[[R3]] {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R21:.*]] = vector.extract_strided_slice %[[R3]] {offsets = [0, 16], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R22:.*]] = vector.extract_strided_slice %[[R4]] {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R23:.*]] = vector.extract_strided_slice %[[R4]] {offsets = [0, 16], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R24:.*]] = vector.extract_strided_slice %[[R5]] {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R25:.*]] = vector.extract_strided_slice %[[R5]] {offsets = [0, 16], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R26:.*]] = vector.extract_strided_slice %[[R6]] {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R27:.*]] = vector.extract_strided_slice %[[R6]] {offsets = [0, 16], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R28:.*]] = vector.extract_strided_slice %[[R7]] {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R29:.*]] = vector.extract_strided_slice %[[R7]] {offsets = [0, 16], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R30:.*]] = vector.extract_strided_slice %[[R8]] {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R31:.*]] = vector.extract_strided_slice %[[R8]] {offsets = [0, 16], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R32:.*]] = vector.extract_strided_slice %[[R9]] {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R33:.*]] = vector.extract_strided_slice %[[R9]] {offsets = [0, 16], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R34:.*]] = vector.extract_strided_slice %[[R10]] {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R35:.*]] = vector.extract_strided_slice %[[R10]] {offsets = [0, 16], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R36:.*]] = vector.extract_strided_slice %[[R11]] {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R37:.*]] = vector.extract_strided_slice %[[R11]] {offsets = [0, 16], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R38:.*]] = vector.extract_strided_slice %[[R12]] {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R39:.*]] = vector.extract_strided_slice %[[R12]] {offsets = [0, 16], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R40:.*]] = vector.extract_strided_slice %[[R13]] {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R41:.*]] = vector.extract_strided_slice %[[R13]] {offsets = [0, 16], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R42:.*]] = vector.extract_strided_slice %[[R14]] {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R43:.*]] = vector.extract_strided_slice %[[R14]] {offsets = [0, 16], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R44:.*]] = vector.extract_strided_slice %[[R15]] {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R45:.*]] = vector.extract_strided_slice %[[R15]] {offsets = [0, 16], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R46:.*]] = vector.extract_strided_slice %[[R16]] {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R47:.*]] = vector.extract_strided_slice %[[R16]] {offsets = [0, 16], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R48:.*]] = vector.extract_strided_slice %[[R17]] {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R49:.*]] = vector.extract_strided_slice %[[R17]] {offsets = [0, 16], sizes = [1, 16], strides = [1, 1]} : vector<1x32xf16> to vector<1x16xf16>
      //CHECK: %[[R50:.*]] = math.exp %[[R18]] : vector<1x16xf16>
      //CHECK: %[[R51:.*]] = math.exp %[[R19]] : vector<1x16xf16>
      //CHECK: %[[R52:.*]] = math.exp %[[R20]] : vector<1x16xf16>
      //CHECK: %[[R53:.*]] = math.exp %[[R21]] : vector<1x16xf16>
      //CHECK: %[[R54:.*]] = math.exp %[[R22]] : vector<1x16xf16>
      //CHECK: %[[R55:.*]] = math.exp %[[R23]] : vector<1x16xf16>
      //CHECK: %[[R56:.*]] = math.exp %[[R24]] : vector<1x16xf16>
      //CHECK: %[[R57:.*]] = math.exp %[[R25]] : vector<1x16xf16>
      //CHECK: %[[R58:.*]] = math.exp %[[R26]] : vector<1x16xf16>
      //CHECK: %[[R59:.*]] = math.exp %[[R27]] : vector<1x16xf16>
      //CHECK: %[[R60:.*]] = math.exp %[[R28]] : vector<1x16xf16>
      //CHECK: %[[R61:.*]] = math.exp %[[R29]] : vector<1x16xf16>
      //CHECK: %[[R62:.*]] = math.exp %[[R30]] : vector<1x16xf16>
      //CHECK: %[[R63:.*]] = math.exp %[[R31]] : vector<1x16xf16>
      //CHECK: %[[R64:.*]] = math.exp %[[R32]] : vector<1x16xf16>
      //CHECK: %[[R65:.*]] = math.exp %[[R33]] : vector<1x16xf16>
      //CHECK: %[[R66:.*]] = math.exp %[[R34]] : vector<1x16xf16>
      //CHECK: %[[R67:.*]] = math.exp %[[R35]] : vector<1x16xf16>
      //CHECK: %[[R68:.*]] = math.exp %[[R36]] : vector<1x16xf16>
      //CHECK: %[[R69:.*]] = math.exp %[[R37]] : vector<1x16xf16>
      //CHECK: %[[R70:.*]] = math.exp %[[R38]] : vector<1x16xf16>
      //CHECK: %[[R71:.*]] = math.exp %[[R39]] : vector<1x16xf16>
      //CHECK: %[[R72:.*]] = math.exp %[[R40]] : vector<1x16xf16>
      //CHECK: %[[R73:.*]] = math.exp %[[R41]] : vector<1x16xf16>
      //CHECK: %[[R74:.*]] = math.exp %[[R42]] : vector<1x16xf16>
      //CHECK: %[[R75:.*]] = math.exp %[[R43]] : vector<1x16xf16>
      //CHECK: %[[R76:.*]] = math.exp %[[R44]] : vector<1x16xf16>
      //CHECK: %[[R77:.*]] = math.exp %[[R45]] : vector<1x16xf16>
      //CHECK: %[[R78:.*]] = math.exp %[[R46]] : vector<1x16xf16>
      //CHECK: %[[R79:.*]] = math.exp %[[R47]] : vector<1x16xf16>
      //CHECK: %[[R80:.*]] = math.exp %[[R48]] : vector<1x16xf16>
      //CHECK: %[[R81:.*]] = math.exp %[[R49]] : vector<1x16xf16>
      %e = math.exp %v: vector<16x32xf16>
      //CHECK: %[[R82:.*]] = arith.addf %[[R50]], %[[R52]] : vector<1x16xf16>
      //CHECK: %[[R83:.*]] = arith.addf %[[R82]], %[[R54]] : vector<1x16xf16>
      //CHECK: %[[R84:.*]] = arith.addf %[[R83]], %[[R56]] : vector<1x16xf16>
      //CHECK: %[[R85:.*]] = arith.addf %[[R84]], %[[R58]] : vector<1x16xf16>
      //CHECK: %[[R86:.*]] = arith.addf %[[R85]], %[[R60]] : vector<1x16xf16>
      //CHECK: %[[R87:.*]] = arith.addf %[[R86]], %[[R62]] : vector<1x16xf16>
      //CHECK: %[[R88:.*]] = arith.addf %[[R87]], %[[R64]] : vector<1x16xf16>
      //CHECK: %[[R89:.*]] = arith.addf %[[R88]], %[[R66]] : vector<1x16xf16>
      //CHECK: %[[R90:.*]] = arith.addf %[[R89]], %[[R68]] : vector<1x16xf16>
      //CHECK: %[[R91:.*]] = arith.addf %[[R90]], %[[R70]] : vector<1x16xf16>
      //CHECK: %[[R92:.*]] = arith.addf %[[R91]], %[[R72]] : vector<1x16xf16>
      //CHECK: %[[R93:.*]] = arith.addf %[[R92]], %[[R74]] : vector<1x16xf16>
      //CHECK: %[[R94:.*]] = arith.addf %[[R93]], %[[R76]] : vector<1x16xf16>
      //CHECK: %[[R95:.*]] = arith.addf %[[R94]], %[[R78]] : vector<1x16xf16>
      //CHECK: %[[R96:.*]] = arith.addf %[[R95]], %[[R80]] : vector<1x16xf16>
      //CHECK: %[[R97:.*]] = arith.addf %[[R51]], %[[R53]] : vector<1x16xf16>
      //CHECK: %[[R98:.*]] = arith.addf %[[R97]], %[[R55]] : vector<1x16xf16>
      //CHECK: %[[R99:.*]] = arith.addf %[[R98]], %[[R57]] : vector<1x16xf16>
      //CHECK: %[[R100:.*]] = arith.addf %[[R99]], %[[R59]] : vector<1x16xf16>
      //CHECK: %[[R101:.*]] = arith.addf %[[R100]], %[[R61]] : vector<1x16xf16>
      //CHECK: %[[R102:.*]] = arith.addf %[[R101]], %[[R63]] : vector<1x16xf16>
      //CHECK: %[[R103:.*]] = arith.addf %[[R102]], %[[R65]] : vector<1x16xf16>
      //CHECK: %[[R104:.*]] = arith.addf %[[R103]], %[[R67]] : vector<1x16xf16>
      //CHECK: %[[R105:.*]] = arith.addf %[[R104]], %[[R69]] : vector<1x16xf16>
      //CHECK: %[[R106:.*]] = arith.addf %[[R105]], %[[R71]] : vector<1x16xf16>
      //CHECK: %[[R107:.*]] = arith.addf %[[R106]], %[[R73]] : vector<1x16xf16>
      //CHECK: %[[R108:.*]] = arith.addf %[[R107]], %[[R75]] : vector<1x16xf16>
      //CHECK: %[[R109:.*]] = arith.addf %[[R108]], %[[R77]] : vector<1x16xf16>
      //CHECK: %[[R110:.*]] = arith.addf %[[R109]], %[[R79]] : vector<1x16xf16>
      //CHECK: %[[R111:.*]] = arith.addf %[[R110]], %[[R81]] : vector<1x16xf16>
      //CHECK: %[[R112:.*]] = vector.shape_cast %[[R96]] : vector<1x16xf16> to vector<16xf16>
      //CHECK: %[[R113:.*]] = vector.shape_cast %[[R111]] : vector<1x16xf16> to vector<16xf16>
      //CHECK: %[[R114:.*]] = vector.shuffle %[[R112]], %[[R113]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf16>, vector<16xf16>
      //CHECK: %[[R115:.*]] = vector.shape_cast %[[R114]] : vector<32xf16> to vector<1x32xf16>
      //CHECK: %[[R116:.*]] = vector.shape_cast %[[R115]] : vector<1x32xf16> to vector<2x16xf16>
      //CHECK: %[[R117:.*]] = vector.shape_cast %[[R116]] : vector<2x16xf16> to vector<32xf16>
      %r = vector.multi_reduction <add>, %e, %acc [0] : vector<16x32xf16> to vector<32xf16>
      //CHECK: %[[R118:.*]] = vector.shape_cast %[[R117]] : vector<32xf16> to vector<4x8xf16>
      %c = vector.shape_cast %r: vector<32xf16> to vector<4x8xf16>
      //CHECK: %[[R119:.*]] = xegpu.create_nd_tdesc %[[arg1]][%[[c0]], %[[c0]]] : memref<128x256xf16> -> !xegpu.tensor_desc<4x8xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
      %s = xetile.init_tile %b[%c0, %c0] : memref<128x256xf16> -> !xetile.tile<4x8xf16>
      //CHECK: xegpu.store_nd %[[R118]], %[[R119]] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<4x8xf16>, !xegpu.tensor_desc<4x8xf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
      xetile.store_tile %c, %s : vector<4x8xf16>, !xetile.tile<4x8xf16>
      gpu.return
    }
  }
}
