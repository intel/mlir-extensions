// RUN: imex-opt --split-input-file --convert-xetile-to-xegpu %s -verify-diagnostics -o -| FileCheck %s
  gpu.module @test_kernel {
    gpu.func @arith_binary_ops() {
        //CHECK: %[[c0:.*]] = arith.constant dense<8.999020e-01> : vector<16x16xf16>
        //CHECK: %[[c1:.*]] = arith.constant dense<8.999020e-01> : vector<16x16xf16>
        //CHECK: %[[c2:.*]] = arith.constant dense<8.999020e-01> : vector<16x16xf16>
        //CHECK: %[[c3:.*]] = arith.constant dense<8.999020e-01> : vector<16x16xf16>
        //CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[c0]], %[[c1]], %[[c2]], %[[c3]] : vector<16x16xf16>, vector<16x16xf16>, vector<16x16xf16>, vector<16x16xf16> to vector<2x2x16x16xf16>

        %0 = arith.constant dense<0.9>: vector<2x2x16x16xf16>

        //CHECK: %[[c4:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c5:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c6:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c7:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c8:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c9:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c10:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c11:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c12:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c13:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c14:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c15:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c16:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c17:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c18:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c19:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c20:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c21:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c22:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c23:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c24:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c25:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c26:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c27:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c28:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c29:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c30:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c31:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c32:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c33:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c34:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c35:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c36:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c37:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c38:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c39:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c40:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c41:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c42:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c43:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c44:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c45:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c46:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c47:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c48:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c49:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c50:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c51:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c52:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c53:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c54:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c55:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c56:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c57:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c58:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c59:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c60:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c61:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c62:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c63:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c64:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c65:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c66:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>
        //CHECK: %[[c67:.*]] = arith.constant dense<2.300780e+00> : vector<1x16xf16>

        %1 = arith.constant dense<2.3>: vector<32x2x1x16xf16>

        //CHECK: %[[CAST1:.*]] = builtin.unrealized_conversion_cast %[[c4]], %[[c5]], %[[c6]], %[[c7]], %[[c8]], %[[c9]], %[[c10]], %[[c11]], %[[c12]], %[[c13]], %[[c14]], %[[c15]], %[[c16]], %[[c17]], %[[c18]], %[[c19]], %[[c20]], %[[c21]], %[[c22]], %[[c23]], %[[c24]], %[[c25]], %[[c26]], %[[c27]], %[[c28]], %[[c29]], %[[c30]], %[[c31]], %[[c32]], %[[c33]], %[[c34]], %[[c35]], %[[c36]], %[[c37]], %[[c38]], %[[c39]], %[[c40]], %[[c41]], %[[c42]], %[[c43]], %[[c44]], %[[c45]], %[[c46]], %[[c47]], %[[c48]], %[[c49]], %[[c50]], %[[c51]], %[[c52]], %[[c53]], %[[c54]], %[[c55]], %[[c56]], %[[c57]], %[[c58]], %[[c59]], %[[c60]], %[[c61]], %[[c62]], %[[c63]], %[[c64]], %[[c65]], %[[c66]], %[[c67]] : vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16>, vector<1x16xf16> to vector<32x2x1x16xf16>
        //CHECK: %[[SLICE1:.*]] = vector.extract_strided_slice %[[c0]] {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE2:.*]] = vector.extract_strided_slice %[[c0]] {offsets = [1, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE3:.*]] = vector.extract_strided_slice %[[c0]] {offsets = [2, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE4:.*]] = vector.extract_strided_slice %[[c0]] {offsets = [3, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE5:.*]] = vector.extract_strided_slice %[[c0]] {offsets = [4, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE6:.*]] = vector.extract_strided_slice %[[c0]] {offsets = [5, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE7:.*]] = vector.extract_strided_slice %[[c0]] {offsets = [6, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE8:.*]] = vector.extract_strided_slice %[[c0]] {offsets = [7, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE9:.*]] = vector.extract_strided_slice %[[c0]] {offsets = [8, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE10:.*]] = vector.extract_strided_slice %[[c0]] {offsets = [9, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE11:.*]] = vector.extract_strided_slice %[[c0]] {offsets = [10, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE12:.*]] = vector.extract_strided_slice %[[c0]] {offsets = [11, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE13:.*]] = vector.extract_strided_slice %[[c0]] {offsets = [12, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE14:.*]] = vector.extract_strided_slice %[[c0]] {offsets = [13, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE15:.*]] = vector.extract_strided_slice %[[c0]] {offsets = [14, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE16:.*]] = vector.extract_strided_slice %[[c0]] {offsets = [15, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE17:.*]] = vector.extract_strided_slice %[[c1]] {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE18:.*]] = vector.extract_strided_slice %[[c1]] {offsets = [1, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE19:.*]] = vector.extract_strided_slice %[[c1]] {offsets = [2, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE20:.*]] = vector.extract_strided_slice %[[c1]] {offsets = [3, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE21:.*]] = vector.extract_strided_slice %[[c1]] {offsets = [4, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE22:.*]] = vector.extract_strided_slice %[[c1]] {offsets = [5, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE23:.*]] = vector.extract_strided_slice %[[c1]] {offsets = [6, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE24:.*]] = vector.extract_strided_slice %[[c1]] {offsets = [7, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE25:.*]] = vector.extract_strided_slice %[[c1]] {offsets = [8, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE26:.*]] = vector.extract_strided_slice %[[c1]] {offsets = [9, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE27:.*]] = vector.extract_strided_slice %[[c1]] {offsets = [10, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE28:.*]] = vector.extract_strided_slice %[[c1]] {offsets = [11, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE29:.*]] = vector.extract_strided_slice %[[c1]] {offsets = [12, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE30:.*]] = vector.extract_strided_slice %[[c1]] {offsets = [13, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE31:.*]] = vector.extract_strided_slice %[[c1]] {offsets = [14, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE32:.*]] = vector.extract_strided_slice %[[c1]] {offsets = [15, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE33:.*]] = vector.extract_strided_slice %[[c2]] {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE34:.*]] = vector.extract_strided_slice %[[c2]] {offsets = [1, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE35:.*]] = vector.extract_strided_slice %[[c2]] {offsets = [2, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE36:.*]] = vector.extract_strided_slice %[[c2]] {offsets = [3, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE37:.*]] = vector.extract_strided_slice %[[c2]] {offsets = [4, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE38:.*]] = vector.extract_strided_slice %[[c2]] {offsets = [5, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE39:.*]] = vector.extract_strided_slice %[[c2]] {offsets = [6, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE40:.*]] = vector.extract_strided_slice %[[c2]] {offsets = [7, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE41:.*]] = vector.extract_strided_slice %[[c2]] {offsets = [8, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE42:.*]] = vector.extract_strided_slice %[[c2]] {offsets = [9, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE43:.*]] = vector.extract_strided_slice %[[c2]] {offsets = [10, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE44:.*]] = vector.extract_strided_slice %[[c2]] {offsets = [11, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE45:.*]] = vector.extract_strided_slice %[[c2]] {offsets = [12, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE46:.*]] = vector.extract_strided_slice %[[c2]] {offsets = [13, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE47:.*]] = vector.extract_strided_slice %[[c2]] {offsets = [14, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE48:.*]] = vector.extract_strided_slice %[[c2]] {offsets = [15, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE49:.*]] = vector.extract_strided_slice %[[c3]] {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE50:.*]] = vector.extract_strided_slice %[[c3]] {offsets = [1, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE51:.*]] = vector.extract_strided_slice %[[c3]] {offsets = [2, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE52:.*]] = vector.extract_strided_slice %[[c3]] {offsets = [3, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE53:.*]] = vector.extract_strided_slice %[[c3]] {offsets = [4, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE54:.*]] = vector.extract_strided_slice %[[c3]] {offsets = [5, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE55:.*]] = vector.extract_strided_slice %[[c3]] {offsets = [6, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE56:.*]] = vector.extract_strided_slice %[[c3]] {offsets = [7, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE57:.*]] = vector.extract_strided_slice %[[c3]] {offsets = [8, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE58:.*]] = vector.extract_strided_slice %[[c3]] {offsets = [9, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE59:.*]] = vector.extract_strided_slice %[[c3]] {offsets = [10, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE60:.*]] = vector.extract_strided_slice %[[c3]] {offsets = [11, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE61:.*]] = vector.extract_strided_slice %[[c3]] {offsets = [12, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE62:.*]] = vector.extract_strided_slice %[[c3]] {offsets = [13, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE63:.*]] = vector.extract_strided_slice %[[c3]] {offsets = [14, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16>
        //CHECK: %[[SLICE64:.*]] = vector.extract_strided_slice %[[c3]] {offsets = [15, 0], sizes = [1, 16], strides = [1, 1]} : vector<16x16xf16> to vector<1x16xf16
        %2 = xetile.tile_unpack %0 {inner_blocks = array<i64: 16, 16>}: vector<2x2x16x16xf16> -> vector<32x32xf16>
        %3 = xetile.tile_pack %2 {inner_blocks = array<i64: 1, 16>}: vector<32x32xf16> -> vector<32x2x1x16xf16>
        //CHECK: %[[ADD1:.*]] = arith.addf %[[SLICE1]], %[[c4]] : vector<1x16xf16>
        //CHECK: %[[ADD2:.*]] = arith.addf %[[SLICE17]], %[[c5]] : vector<1x16xf16>
        //CHECK: %[[ADD3:.*]] = arith.addf %[[SLICE2]], %[[c6]] : vector<1x16xf16>
        //CHECK: %[[ADD4:.*]] = arith.addf %[[SLICE18]], %[[c7]] : vector<1x16xf16>
        //CHECK: %[[ADD5:.*]] = arith.addf %[[SLICE3]], %[[c8]] : vector<1x16xf16>
        //CHECK: %[[ADD6:.*]] = arith.addf %[[SLICE19]], %[[c9]] : vector<1x16xf16>
        //CHECK: %[[ADD7:.*]] = arith.addf %[[SLICE4]], %[[c10]] : vector<1x16xf16>
        //CHECK: %[[ADD8:.*]] = arith.addf %[[SLICE20]], %[[c11]] : vector<1x16xf16>
        //CHECK: %[[ADD9:.*]] = arith.addf %[[SLICE5]], %[[c12]] : vector<1x16xf16>
        //CHECK: %[[ADD10:.*]] = arith.addf %[[SLICE21]], %[[c13]] : vector<1x16xf16>
        //CHECK: %[[ADD11:.*]] = arith.addf %[[SLICE6]], %[[c14]] : vector<1x16xf16>
        //CHECK: %[[ADD12:.*]] = arith.addf %[[SLICE22]], %[[c15]] : vector<1x16xf16>
        //CHECK: %[[ADD13:.*]] = arith.addf %[[SLICE7]], %[[c16]] : vector<1x16xf16>
        //CHECK: %[[ADD14:.*]] = arith.addf %[[SLICE23]], %[[c17]] : vector<1x16xf16>
        //CHECK: %[[ADD15:.*]] = arith.addf %[[SLICE8]], %[[c18]] : vector<1x16xf16>
        //CHECK: %[[ADD16:.*]] = arith.addf %[[SLICE24]], %[[c19]] : vector<1x16xf16>
        //CHECK: %[[ADD17:.*]] = arith.addf %[[SLICE9]], %[[c20]] : vector<1x16xf16>
        //CHECK: %[[ADD18:.*]] = arith.addf %[[SLICE25]], %[[c21]] : vector<1x16xf16>
        //CHECK: %[[ADD19:.*]] = arith.addf %[[SLICE10]], %[[c22]] : vector<1x16xf16>
        //CHECK: %[[ADD20:.*]] = arith.addf %[[SLICE26]], %[[c23]] : vector<1x16xf16>
        //CHECK: %[[ADD21:.*]] = arith.addf %[[SLICE11]], %[[c24]] : vector<1x16xf16>
        //CHECK: %[[ADD22:.*]] = arith.addf %[[SLICE27]], %[[c25]] : vector<1x16xf16>
        //CHECK: %[[ADD23:.*]] = arith.addf %[[SLICE12]], %[[c26]] : vector<1x16xf16>
        //CHECK: %[[ADD24:.*]] = arith.addf %[[SLICE28]], %[[c27]] : vector<1x16xf16>
        //CHECK: %[[ADD25:.*]] = arith.addf %[[SLICE13]], %[[c28]] : vector<1x16xf16>
        //CHECK: %[[ADD26:.*]] = arith.addf %[[SLICE29]], %[[c29]] : vector<1x16xf16>
        //CHECK: %[[ADD27:.*]] = arith.addf %[[SLICE14]], %[[c30]] : vector<1x16xf16>
        //CHECK: %[[ADD28:.*]] = arith.addf %[[SLICE30]], %[[c31]] : vector<1x16xf16>
        //CHECK: %[[ADD29:.*]] = arith.addf %[[SLICE15]], %[[c32]] : vector<1x16xf16>
        //CHECK: %[[ADD30:.*]] = arith.addf %[[SLICE31]], %[[c33]] : vector<1x16xf16>
        //CHECK: %[[ADD31:.*]] = arith.addf %[[SLICE16]], %[[c34]] : vector<1x16xf16>
        //CHECK: %[[ADD32:.*]] = arith.addf %[[SLICE32]], %[[c35]] : vector<1x16xf16>
        //CHECK: %[[ADD33:.*]] = arith.addf %[[SLICE33]], %[[c36]] : vector<1x16xf16>
        //CHECK: %[[ADD34:.*]] = arith.addf %[[SLICE49]], %[[c37]] : vector<1x16xf16>
        //CHECK: %[[ADD35:.*]] = arith.addf %[[SLICE34]], %[[c38]] : vector<1x16xf16>
        //CHECK: %[[ADD36:.*]] = arith.addf %[[SLICE50]], %[[c39]] : vector<1x16xf16>
        //CHECK: %[[ADD37:.*]] = arith.addf %[[SLICE35]], %[[c40]] : vector<1x16xf16>
        //CHECK: %[[ADD38:.*]] = arith.addf %[[SLICE51]], %[[c41]] : vector<1x16xf16>
        //CHECK: %[[ADD39:.*]] = arith.addf %[[SLICE36]], %[[c42]] : vector<1x16xf16>
        //CHECK: %[[ADD40:.*]] = arith.addf %[[SLICE52]], %[[c43]] : vector<1x16xf16>
        //CHECK: %[[ADD41:.*]] = arith.addf %[[SLICE37]], %[[c44]] : vector<1x16xf16>
        //CHECK: %[[ADD42:.*]] = arith.addf %[[SLICE53]], %[[c45]] : vector<1x16xf16>
        //CHECK: %[[ADD43:.*]] = arith.addf %[[SLICE38]], %[[c46]] : vector<1x16xf16>
        //CHECK: %[[ADD44:.*]] = arith.addf %[[SLICE54]], %[[c47]] : vector<1x16xf16>
        //CHECK: %[[ADD45:.*]] = arith.addf %[[SLICE39]], %[[c48]] : vector<1x16xf16>
        //CHECK: %[[ADD46:.*]] = arith.addf %[[SLICE55]], %[[c49]] : vector<1x16xf16>
        //CHECK: %[[ADD47:.*]] = arith.addf %[[SLICE40]], %[[c50]] : vector<1x16xf16>
        //CHECK: %[[ADD48:.*]] = arith.addf %[[SLICE56]], %[[c51]] : vector<1x16xf16>
        //CHECK: %[[ADD49:.*]] = arith.addf %[[SLICE41]], %[[c52]] : vector<1x16xf16>
        //CHECK: %[[ADD50:.*]] = arith.addf %[[SLICE57]], %[[c53]] : vector<1x16xf16>
        //CHECK: %[[ADD51:.*]] = arith.addf %[[SLICE42]], %[[c54]] : vector<1x16xf16>
        //CHECK: %[[ADD52:.*]] = arith.addf %[[SLICE58]], %[[c55]] : vector<1x16xf16>
        //CHECK: %[[ADD53:.*]] = arith.addf %[[SLICE43]], %[[c56]] : vector<1x16xf16>
        //CHECK: %[[ADD54:.*]] = arith.addf %[[SLICE59]], %[[c57]] : vector<1x16xf16>
        //CHECK: %[[ADD55:.*]] = arith.addf %[[SLICE44]], %[[c58]] : vector<1x16xf16>
        //CHECK: %[[ADD56:.*]] = arith.addf %[[SLICE60]], %[[c59]] : vector<1x16xf16>
        //CHECK: %[[ADD57:.*]] = arith.addf %[[SLICE45]], %[[c60]] : vector<1x16xf16>
        //CHECK: %[[ADD58:.*]] = arith.addf %[[SLICE61]], %[[c61]] : vector<1x16xf16>
        //CHECK: %[[ADD59:.*]] = arith.addf %[[SLICE46]], %[[c62]] : vector<1x16xf16>
        //CHECK: %[[ADD60:.*]] = arith.addf %[[SLICE62]], %[[c63]] : vector<1x16xf16>
        //CHECK: %[[ADD61:.*]] = arith.addf %[[SLICE47]], %[[c64]] : vector<1x16xf16>
        //CHECK: %[[ADD62:.*]] = arith.addf %[[SLICE63]], %[[c65]] : vector<1x16xf16>
        //CHECK: %[[ADD63:.*]] = arith.addf %[[SLICE48]], %[[c66]] : vector<1x16xf16>
        //CHECK: %[[ADD64:.*]] = arith.addf %[[SLICE64]], %[[c67]] : vector<1x16xf16>

        %result = arith.addf %3, %1 : vector<32x2x1x16xf16>
        gpu.return
    }
  }
