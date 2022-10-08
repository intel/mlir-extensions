#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> ()>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, 0, 0, d3)>
#map6 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @main() {
    %cst = arith.constant dense<1.000000e+00> : tensor<1x224x224x3xf32>
    %0 = call @predict(%cst) : (tensor<1x224x224x3xf32>) -> tensor<1x1000xf32>
    %1 = tensor.cast %0 : tensor<1x1000xf32> to tensor<*xf32>
    call @printMemrefF32(%1) : (tensor<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF32(tensor<*xf32>)
  func.func @predict(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x1000xf32> {
    %cst = arith.constant dense<0.448330253> : tensor<1x112x112x16xf32>
    %cst_0 = arith.constant dense<5.000000e-01> : tensor<1x112x112x16xf32>
    %cst_1 = arith.constant dense<0.333333343> : tensor<1x112x112x16xf32>
    %cst_2 = arith.constant dense<2.500000e-01> : tensor<1x112x112x16xf32>
    %cst_3 = arith.constant dense<0.166666672> : tensor<3x3x1x16xf32>
    %cst_4 = arith.constant dense<0.317804962> : tensor<1x56x56x16xf32>
    %cst_5 = arith.constant dense<0.142857149> : tensor<1x56x56x16xf32>
    %cst_6 = arith.constant dense<1.250000e-01> : tensor<1x56x56x16xf32>
    %cst_7 = arith.constant dense<0.111111112> : tensor<1x56x56x16xf32>
    %cst_8 = arith.constant dense<0.0833333358> : tensor<1x1x1x8xf32>
    %cst_9 = arith.constant dense<0.0714285746> : tensor<1x1x1x16xf32>
    %cst_10 = arith.constant dense<0.231584921> : tensor<1x56x56x16xf32>
    %cst_11 = arith.constant dense<6.250000e-02> : tensor<1x56x56x16xf32>
    %cst_12 = arith.constant dense<0.0588235296> : tensor<1x56x56x16xf32>
    %cst_13 = arith.constant dense<0.055555556> : tensor<1x56x56x16xf32>
    %cst_14 = arith.constant dense<0.206559107> : tensor<1x56x56x72xf32>
    %cst_15 = arith.constant dense<0.0476190485> : tensor<1x56x56x72xf32>
    %cst_16 = arith.constant dense<0.0454545468> : tensor<1x56x56x72xf32>
    %cst_17 = arith.constant dense<0.0434782617> : tensor<1x56x56x72xf32>
    %cst_18 = arith.constant dense<4.000000e-02> : tensor<3x3x1x72xf32>
    %cst_19 = arith.constant dense<0.188368678> : tensor<1x28x28x72xf32>
    %cst_20 = arith.constant dense<0.0384615399> : tensor<1x28x28x72xf32>
    %cst_21 = arith.constant dense<0.0370370373> : tensor<1x28x28x72xf32>
    %cst_22 = arith.constant dense<0.0357142873> : tensor<1x28x28x72xf32>
    %cst_23 = arith.constant dense<0.17438969> : tensor<1x28x28x24xf32>
    %cst_24 = arith.constant dense<0.0322580636> : tensor<1x28x28x24xf32>
    %cst_25 = arith.constant dense<3.125000e-02> : tensor<1x28x28x24xf32>
    %cst_26 = arith.constant dense<0.0303030312> : tensor<1x28x28x24xf32>
    %cst_27 = arith.constant dense<0.163220793> : tensor<1x28x28x88xf32>
    %cst_28 = arith.constant dense<0.027777778> : tensor<1x28x28x88xf32>
    %cst_29 = arith.constant dense<0.0270270277> : tensor<1x28x28x88xf32>
    %cst_30 = arith.constant dense<0.0263157897> : tensor<1x28x28x88xf32>
    %cst_31 = arith.constant dense<2.500000e-02> : tensor<3x3x1x88xf32>
    %cst_32 = arith.constant dense<0.154036596> : tensor<1x28x28x88xf32>
    %cst_33 = arith.constant dense<0.024390243> : tensor<1x28x28x88xf32>
    %cst_34 = arith.constant dense<0.0238095243> : tensor<1x28x28x88xf32>
    %cst_35 = arith.constant dense<0.0232558139> : tensor<1x28x28x88xf32>
    %cst_36 = arith.constant dense<0.146315292> : tensor<1x28x28x24xf32>
    %cst_37 = arith.constant dense<0.0217391308> : tensor<1x28x28x24xf32>
    %cst_38 = arith.constant dense<0.0212765951> : tensor<1x28x28x24xf32>
    %cst_39 = arith.constant dense<0.020833334> : tensor<1x28x28x24xf32>
    %cst_40 = arith.constant dense<0.139708698> : tensor<1x28x28x96xf32>
    %cst_41 = arith.constant dense<0.0196078438> : tensor<1x28x28x96xf32>
    %cst_42 = arith.constant dense<0.0192307699> : tensor<1x28x28x96xf32>
    %cst_43 = arith.constant dense<0.0188679248> : tensor<1x28x28x96xf32>
    %cst_44 = arith.constant dense<0.0181818176> : tensor<5x5x1x96xf32>
    %cst_45 = arith.constant dense<0.133974448> : tensor<1x14x14x96xf32>
    %cst_46 = arith.constant dense<0.0178571437> : tensor<1x14x14x96xf32>
    %cst_47 = arith.constant dense<0.0175438598> : tensor<1x14x14x96xf32>
    %cst_48 = arith.constant dense<0.0172413792> : tensor<1x14x14x96xf32>
    %cst_49 = arith.constant dense<0.0163934417> : tensor<1x1x1x24xf32>
    %cst_50 = arith.constant dense<0.0158730168> : tensor<1x1x1x96xf32>
    %cst_51 = arith.constant dense<0.125323102> : tensor<1x14x14x40xf32>
    %cst_52 = arith.constant dense<0.0153846154> : tensor<1x14x14x40xf32>
    %cst_53 = arith.constant dense<0.0151515156> : tensor<1x14x14x40xf32>
    %cst_54 = arith.constant dense<0.0149253728> : tensor<1x14x14x40xf32>
    %cst_55 = arith.constant dense<0.121237904> : tensor<1x14x14x240xf32>
    %cst_56 = arith.constant dense<0.0142857144> : tensor<1x14x14x240xf32>
    %cst_57 = arith.constant dense<0.0140845068> : tensor<1x14x14x240xf32>
    %cst_58 = arith.constant dense<0.013888889> : tensor<1x14x14x240xf32>
    %cst_59 = arith.constant dense<0.0135135138> : tensor<5x5x1x240xf32>
    %cst_60 = arith.constant dense<0.117560677> : tensor<1x14x14x240xf32>
    %cst_61 = arith.constant dense<0.0133333337> : tensor<1x14x14x240xf32>
    %cst_62 = arith.constant dense<0.0131578948> : tensor<1x14x14x240xf32>
    %cst_63 = arith.constant dense<0.012987013> : tensor<1x14x14x240xf32>
    %cst_64 = arith.constant dense<1.250000e-02> : tensor<1x1x1x64xf32>
    %cst_65 = arith.constant dense<0.0121951215> : tensor<1x1x1x240xf32>
    %cst_66 = arith.constant dense<0.111777693> : tensor<1x14x14x40xf32>
    %cst_67 = arith.constant dense<0.0119047621> : tensor<1x14x14x40xf32>
    %cst_68 = arith.constant dense<0.0117647061> : tensor<1x14x14x40xf32>
    %cst_69 = arith.constant dense<0.0116279069> : tensor<1x14x14x40xf32>
    %cst_70 = arith.constant dense<0.108947538> : tensor<1x14x14x240xf32>
    %cst_71 = arith.constant dense<0.0112359552> : tensor<1x14x14x240xf32>
    %cst_72 = arith.constant dense<0.0111111114> : tensor<1x14x14x240xf32>
    %cst_73 = arith.constant dense<0.0109890113> : tensor<1x14x14x240xf32>
    %cst_74 = arith.constant dense<0.0107526882> : tensor<5x5x1x240xf32>
    %cst_75 = arith.constant dense<0.106345087> : tensor<1x14x14x240xf32>
    %cst_76 = arith.constant dense<0.0106382975> : tensor<1x14x14x240xf32>
    %cst_77 = arith.constant dense<0.0105263162> : tensor<1x14x14x240xf32>
    %cst_78 = arith.constant dense<0.010416667> : tensor<1x14x14x240xf32>
    %cst_79 = arith.constant dense<0.0101010101> : tensor<1x1x1x64xf32>
    %cst_80 = arith.constant dense<9.900990e-03> : tensor<1x1x1x240xf32>
    %cst_81 = arith.constant dense<0.102146767> : tensor<1x14x14x40xf32>
    %cst_82 = arith.constant dense<0.00970873795> : tensor<1x14x14x40xf32>
    %cst_83 = arith.constant dense<0.00961538497> : tensor<1x14x14x40xf32>
    %cst_84 = arith.constant dense<9.523810e-03> : tensor<1x14x14x40xf32>
    %cst_85 = arith.constant dense<0.10004504> : tensor<1x14x14x120xf32>
    %cst_86 = arith.constant dense<0.00925925932> : tensor<1x14x14x120xf32>
    %cst_87 = arith.constant dense<0.00917431153> : tensor<1x14x14x120xf32>
    %cst_88 = arith.constant dense<0.0090909088> : tensor<1x14x14x120xf32>
    %cst_89 = arith.constant dense<0.00892857183> : tensor<5x5x1x120xf32>
    %cst_90 = arith.constant dense<0.0980851128> : tensor<1x14x14x120xf32>
    %cst_91 = arith.constant dense<0.00884955748> : tensor<1x14x14x120xf32>
    %cst_92 = arith.constant dense<0.00877192988> : tensor<1x14x14x120xf32>
    %cst_93 = arith.constant dense<0.00869565178> : tensor<1x14x14x120xf32>
    %cst_94 = arith.constant dense<0.00847457629> : tensor<1x1x1x32xf32>
    %cst_95 = arith.constant dense<0.00833333377> : tensor<1x1x1x120xf32>
    %cst_96 = arith.constant dense<0.0948683321> : tensor<1x14x14x48xf32>
    %cst_97 = arith.constant dense<0.00819672085> : tensor<1x14x14x48xf32>
    %cst_98 = arith.constant dense<0.008130081> : tensor<1x14x14x48xf32>
    %cst_99 = arith.constant dense<0.00806451589> : tensor<1x14x14x48xf32>
    %cst_100 = arith.constant dense<0.0932325422> : tensor<1x14x14x144xf32>
    %cst_101 = arith.constant dense<0.00787401571> : tensor<1x14x14x144xf32>
    %cst_102 = arith.constant dense<7.812500e-03> : tensor<1x14x14x144xf32>
    %cst_103 = arith.constant dense<0.00775193795> : tensor<1x14x14x144xf32>
    %cst_104 = arith.constant dense<0.00763358781> : tensor<5x5x1x144xf32>
    %cst_105 = arith.constant dense<0.0916919186> : tensor<1x14x14x144xf32>
    %cst_106 = arith.constant dense<0.0075757578> : tensor<1x14x14x144xf32>
    %cst_107 = arith.constant dense<0.00751879718> : tensor<1x14x14x144xf32>
    %cst_108 = arith.constant dense<0.00746268639> : tensor<1x14x14x144xf32>
    %cst_109 = arith.constant dense<7.299270e-03> : tensor<1x1x1x40xf32>
    %cst_110 = arith.constant dense<0.00719424477> : tensor<1x1x1x144xf32>
    %cst_111 = arith.constant dense<0.0891316086> : tensor<1x14x14x48xf32>
    %cst_112 = arith.constant dense<0.00709219835> : tensor<1x14x14x48xf32>
    %cst_113 = arith.constant dense<0.00704225338> : tensor<1x14x14x48xf32>
    %cst_114 = arith.constant dense<0.00699300691> : tensor<1x14x14x48xf32>
    %cst_115 = arith.constant dense<0.087814629> : tensor<1x14x14x288xf32>
    %cst_116 = arith.constant dense<0.00684931502> : tensor<1x14x14x288xf32>
    %cst_117 = arith.constant dense<0.00680272094> : tensor<1x14x14x288xf32>
    %cst_118 = arith.constant dense<0.00675675692> : tensor<1x14x14x288xf32>
    %cst_119 = arith.constant dense<0.00666666683> : tensor<5x5x1x288xf32>
    %cst_120 = arith.constant dense<0.08656504> : tensor<1x7x7x288xf32>
    %cst_121 = arith.constant dense<0.00662251655> : tensor<1x7x7x288xf32>
    %cst_122 = arith.constant dense<0.00657894742> : tensor<1x7x7x288xf32>
    %cst_123 = arith.constant dense<0.00653594779> : tensor<1x7x7x288xf32>
    %cst_124 = arith.constant dense<0.00641025649> : tensor<1x1x1x72xf32>
    %cst_125 = arith.constant dense<0.00632911408> : tensor<1x1x1x288xf32>
    %cst_126 = arith.constant dense<0.0844687446> : tensor<1x7x7x96xf32>
    %cst_127 = arith.constant dense<6.250000e-03> : tensor<1x7x7x96xf32>
    %cst_128 = arith.constant dense<0.00621118024> : tensor<1x7x7x96xf32>
    %cst_129 = arith.constant dense<0.00617283955> : tensor<1x7x7x96xf32>
    %cst_130 = arith.constant dense<0.0833809375> : tensor<1x7x7x576xf32>
    %cst_131 = arith.constant dense<0.00606060587> : tensor<1x7x7x576xf32>
    %cst_132 = arith.constant dense<0.00602409616> : tensor<1x7x7x576xf32>
    %cst_133 = arith.constant dense<0.00598802418> : tensor<1x7x7x576xf32>
    %cst_134 = arith.constant dense<5.917160e-03> : tensor<5x5x1x576xf32>
    %cst_135 = arith.constant dense<0.082342863> : tensor<1x7x7x576xf32>
    %cst_136 = arith.constant dense<0.00588235306> : tensor<1x7x7x576xf32>
    %cst_137 = arith.constant dense<0.00584795326> : tensor<1x7x7x576xf32>
    %cst_138 = arith.constant dense<0.00581395347> : tensor<1x7x7x576xf32>
    %cst_139 = arith.constant dense<0.00571428565> : tensor<1x1x1x144xf32>
    %cst_140 = arith.constant dense<0.00564971752> : tensor<1x1x1x576xf32>
    %cst_141 = arith.constant dense<0.0805884972> : tensor<1x7x7x96xf32>
    %cst_142 = arith.constant dense<0.00558659201> : tensor<1x7x7x96xf32>
    %cst_143 = arith.constant dense<0.00555555569> : tensor<1x7x7x96xf32>
    %cst_144 = arith.constant dense<0.00552486209> : tensor<1x7x7x96xf32>
    %cst_145 = arith.constant dense<0.0796717852> : tensor<1x7x7x576xf32>
    %cst_146 = arith.constant dense<0.00543478271> : tensor<1x7x7x576xf32>
    %cst_147 = arith.constant dense<0.00540540554> : tensor<1x7x7x576xf32>
    %cst_148 = arith.constant dense<0.00537634408> : tensor<1x7x7x576xf32>
    %cst_149 = arith.constant dense<0.00531914877> : tensor<5x5x1x576xf32>
    %cst_150 = arith.constant dense<0.0787929818> : tensor<1x7x7x576xf32>
    %cst_151 = arith.constant dense<0.00529100513> : tensor<1x7x7x576xf32>
    %cst_152 = arith.constant dense<0.00526315812> : tensor<1x7x7x576xf32>
    %cst_153 = arith.constant dense<0.00523560215> : tensor<1x7x7x576xf32>
    %cst_154 = arith.constant dense<0.00515463902> : tensor<1x1x1x144xf32>
    %cst_155 = arith.constant dense<0.00510204071> : tensor<1x1x1x576xf32>
    %cst_156 = arith.constant dense<0.0772989318> : tensor<1x7x7x96xf32>
    %cst_157 = arith.constant dense<0.00505050505> : tensor<1x7x7x96xf32>
    %cst_158 = arith.constant dense<0.00502512557> : tensor<1x7x7x96xf32>
    %cst_159 = arith.constant dense<5.000000e-03> : tensor<1x7x7x96xf32>
    %cst_160 = arith.constant dense<0.0765138492> : tensor<1x7x7x576xf32>
    %cst_161 = arith.constant dense<0.00492610829> : tensor<1x7x7x576xf32>
    %cst_162 = arith.constant dense<0.00490196096> : tensor<1x7x7x576xf32>
    %cst_163 = arith.constant dense<0.00487804879> : tensor<1x7x7x576xf32>
    %cst_164 = arith.constant dense<0.00480769249> : tensor<1x1x1x1024xf32>
    %cst_165 = arith.constant dense<0.00476190494> : tensor<1x1x1x1000xf32>
    %cst_166 = arith.constant dense<0.00784313772> : tensor<1x224x224x3xf32>
    %cst_167 = arith.constant dense<-1.000000e+00> : tensor<1x224x224x3xf32>
    %cst_168 = arith.constant dense<3.000000e+00> : tensor<1x112x112x16xf32>
    %cst_169 = arith.constant dense<3.000000e+00> : tensor<1x1x1x16xf32>
    %cst_170 = arith.constant dense<3.000000e+00> : tensor<1x28x28x96xf32>
    %cst_171 = arith.constant dense<3.000000e+00> : tensor<1x14x14x96xf32>
    %cst_172 = arith.constant dense<3.000000e+00> : tensor<1x1x1x96xf32>
    %cst_173 = arith.constant dense<3.000000e+00> : tensor<1x14x14x240xf32>
    %cst_174 = arith.constant dense<3.000000e+00> : tensor<1x1x1x240xf32>
    %cst_175 = arith.constant dense<3.000000e+00> : tensor<1x14x14x120xf32>
    %cst_176 = arith.constant dense<3.000000e+00> : tensor<1x1x1x120xf32>
    %cst_177 = arith.constant dense<3.000000e+00> : tensor<1x14x14x144xf32>
    %cst_178 = arith.constant dense<3.000000e+00> : tensor<1x1x1x144xf32>
    %cst_179 = arith.constant dense<3.000000e+00> : tensor<1x14x14x288xf32>
    %cst_180 = arith.constant dense<3.000000e+00> : tensor<1x7x7x288xf32>
    %cst_181 = arith.constant dense<3.000000e+00> : tensor<1x1x1x288xf32>
    %cst_182 = arith.constant dense<3.000000e+00> : tensor<1x1x1x576xf32>
    %cst_183 = arith.constant dense<3.000000e+00> : tensor<1x7x7x576xf32>
    %cst_184 = arith.constant dense<3.000000e+00> : tensor<1x1x1x1024xf32>
    %cst_185 = arith.constant dense<0.166666672> : tensor<1x112x112x16xf32>
    %cst_186 = arith.constant dense<0.166666672> : tensor<1x1x1x16xf32>
    %cst_187 = arith.constant dense<0.166666672> : tensor<1x28x28x96xf32>
    %cst_188 = arith.constant dense<0.166666672> : tensor<1x14x14x96xf32>
    %cst_189 = arith.constant dense<0.166666672> : tensor<1x1x1x96xf32>
    %cst_190 = arith.constant dense<0.166666672> : tensor<1x14x14x240xf32>
    %cst_191 = arith.constant dense<0.166666672> : tensor<1x1x1x240xf32>
    %cst_192 = arith.constant dense<0.166666672> : tensor<1x14x14x120xf32>
    %cst_193 = arith.constant dense<0.166666672> : tensor<1x1x1x120xf32>
    %cst_194 = arith.constant dense<0.166666672> : tensor<1x14x14x144xf32>
    %cst_195 = arith.constant dense<0.166666672> : tensor<1x1x1x144xf32>
    %cst_196 = arith.constant dense<0.166666672> : tensor<1x14x14x288xf32>
    %cst_197 = arith.constant dense<0.166666672> : tensor<1x7x7x288xf32>
    %cst_198 = arith.constant dense<0.166666672> : tensor<1x1x1x288xf32>
    %cst_199 = arith.constant dense<0.166666672> : tensor<1x1x1x576xf32>
    %cst_200 = arith.constant dense<0.166666672> : tensor<1x7x7x576xf32>
    %cst_201 = arith.constant dense<0.166666672> : tensor<1x1x1x1024xf32>
    %cst_202 = arith.constant dense<0.000000e+00> : tensor<1x56x56x16xf32>
    %cst_203 = arith.constant dense<3.136000e+03> : tensor<1x16xf32>
    %cst_204 = arith.constant dense<0.000000e+00> : tensor<1x1x1x8xf32>
    %cst_205 = arith.constant dense<0.000000e+00> : tensor<1x56x56x72xf32>
    %cst_206 = arith.constant dense<0.000000e+00> : tensor<1x28x28x72xf32>
    %cst_207 = arith.constant dense<0.000000e+00> : tensor<1x28x28x88xf32>
    %cst_208 = arith.constant dense<1.960000e+02> : tensor<1x96xf32>
    %cst_209 = arith.constant dense<0.000000e+00> : tensor<1x1x1x24xf32>
    %cst_210 = arith.constant dense<1.960000e+02> : tensor<1x240xf32>
    %cst_211 = arith.constant dense<0.000000e+00> : tensor<1x1x1x64xf32>
    %cst_212 = arith.constant dense<1.960000e+02> : tensor<1x120xf32>
    %cst_213 = arith.constant dense<0.000000e+00> : tensor<1x1x1x32xf32>
    %cst_214 = arith.constant dense<1.960000e+02> : tensor<1x144xf32>
    %cst_215 = arith.constant dense<0.000000e+00> : tensor<1x1x1x40xf32>
    %cst_216 = arith.constant dense<4.900000e+01> : tensor<1x288xf32>
    %cst_217 = arith.constant dense<0.000000e+00> : tensor<1x1x1x72xf32>
    %cst_218 = arith.constant dense<0.000000e+00> : tensor<1x1x1x144xf32>
    %cst_219 = arith.constant dense<4.900000e+01> : tensor<1x576xf32>
    %cst_220 = arith.constant dense<6.000000e+00> : tensor<f32>
    %cst_221 = arith.constant dense<0xFF800000> : tensor<f32>
    %cst_222 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_223 = arith.constant dense<0.00495049497> : tensor<1x1x96x576xf32>
    %cst_224 = arith.constant dense<0.00483091781> : tensor<1x1x576x1024xf32>
    %cst_225 = arith.constant dense<1.000000e+00> : tensor<3x3x3x16xf32>
    %cst_226 = arith.constant dense<0.00546448072> : tensor<1x1x96x576xf32>
    %cst_227 = arith.constant dense<0.00507614203> : tensor<1x1x576x96xf32>
    %cst_228 = arith.constant dense<0.00512820529> : tensor<1x1x144x576xf32>
    %cst_229 = arith.constant dense<0.00518134702> : tensor<1x1x576x144xf32>
    %cst_230 = arith.constant dense<5.000000e-02> : tensor<1x1x16x72xf32>
    %cst_231 = arith.constant dense<0.0333333351> : tensor<1x1x72x24xf32>
    %cst_232 = arith.constant dense<0.0285714287> : tensor<1x1x24x88xf32>
    %cst_233 = arith.constant dense<0.0222222228> : tensor<1x1x88x24xf32>
    %cst_234 = arith.constant dense<2.000000e-02> : tensor<1x1x24x96xf32>
    %cst_235 = arith.constant dense<1.562500e-02> : tensor<1x1x96x40xf32>
    %cst_236 = arith.constant dense<0.0161290318> : tensor<1x1x24x96xf32>
    %cst_237 = arith.constant dense<0.0166666675> : tensor<1x1x96x24xf32>
    %cst_238 = arith.constant dense<0.0144927539> : tensor<1x1x40x240xf32>
    %cst_239 = arith.constant dense<0.0120481923> : tensor<1x1x240x40xf32>
    %cst_240 = arith.constant dense<0.0123456791> : tensor<1x1x64x240xf32>
    %cst_241 = arith.constant dense<0.0126582282> : tensor<1x1x240x64xf32>
    %cst_242 = arith.constant dense<0.0113636367> : tensor<1x1x40x240xf32>
    %cst_243 = arith.constant dense<0.00980392192> : tensor<1x1x240x40xf32>
    %cst_244 = arith.constant dense<0.00999999977> : tensor<1x1x64x240xf32>
    %cst_245 = arith.constant dense<0.0102040814> : tensor<1x1x240x64xf32>
    %cst_246 = arith.constant dense<0.00934579409> : tensor<1x1x40x120xf32>
    %cst_247 = arith.constant dense<0.00826446246> : tensor<1x1x120x48xf32>
    %cst_248 = arith.constant dense<0.00840336177> : tensor<1x1x32x120xf32>
    %cst_249 = arith.constant dense<0.00854700897> : tensor<1x1x120x32xf32>
    %cst_250 = arith.constant dense<0.00793650839> : tensor<1x1x48x144xf32>
    %cst_251 = arith.constant dense<0.00714285718> : tensor<1x1x144x48xf32>
    %cst_252 = arith.constant dense<0.00724637694> : tensor<1x1x40x144xf32>
    %cst_253 = arith.constant dense<0.0073529412> : tensor<1x1x144x40xf32>
    %cst_254 = arith.constant dense<0.0068965517> : tensor<1x1x48x288xf32>
    %cst_255 = arith.constant dense<0.00628930796> : tensor<1x1x288x96xf32>
    %cst_256 = arith.constant dense<0.00636942684> : tensor<1x1x72x288xf32>
    %cst_257 = arith.constant dense<0.0064516128> : tensor<1x1x288x72xf32>
    %cst_258 = arith.constant dense<0.00609756075> : tensor<1x1x96x576xf32>
    %cst_259 = arith.constant dense<0.00561797759> : tensor<1x1x576x96xf32>
    %cst_260 = arith.constant dense<0.00568181835> : tensor<1x1x144x576xf32>
    %cst_261 = arith.constant dense<0.00574712642> : tensor<1x1x576x144xf32>
    %cst_262 = arith.constant dense<0.0666666701> : tensor<1x1x16x16xf32>
    %cst_263 = arith.constant dense<0.0769230798> : tensor<1x1x8x16xf32>
    %cst_264 = arith.constant dense<0.0909090936> : tensor<1x1x16x8xf32>
    %cst_265 = arith.constant dense<0.00478468882> : tensor<1x1x1024x1000xf32>
    %0 = linalg.init_tensor [1, 224, 224, 3] : tensor<1x224x224x3xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %cst_166 : tensor<1x224x224x3xf32>, tensor<1x224x224x3xf32>) outs(%0 : tensor<1x224x224x3xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x224x224x3xf32>
    %2 = linalg.init_tensor [1, 224, 224, 3] : tensor<1x224x224x3xf32>
    %3 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1, %cst_167 : tensor<1x224x224x3xf32>, tensor<1x224x224x3xf32>) outs(%2 : tensor<1x224x224x3xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x224x224x3xf32>
    %4 = linalg.init_tensor [1, 112, 112, 16] : tensor<1x112x112x16xf32>
    %cst_266 = arith.constant 0.000000e+00 : f32
    %5 = linalg.fill ins(%cst_266 : f32) outs(%4 : tensor<1x112x112x16xf32>) -> tensor<1x112x112x16xf32>
    %cst_267 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_268 = arith.constant 0.000000e+00 : f32
    %6 = linalg.init_tensor [1, 225, 225, 3] : tensor<1x225x225x3xf32>
    %7 = linalg.fill ins(%cst_268 : f32) outs(%6 : tensor<1x225x225x3xf32>) -> tensor<1x225x225x3xf32>
    %8 = tensor.insert_slice %3 into %7[0, 0, 0, 0] [1, 224, 224, 3] [1, 1, 1, 1] : tensor<1x224x224x3xf32> into tensor<1x225x225x3xf32>
    %9 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%8, %cst_225 : tensor<1x225x225x3xf32>, tensor<3x3x3x16xf32>) outs(%5 : tensor<1x112x112x16xf32>) -> tensor<1x112x112x16xf32>
    %10 = linalg.init_tensor [1, 112, 112, 16] : tensor<1x112x112x16xf32>
    %11 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%9, %cst_2 : tensor<1x112x112x16xf32>, tensor<1x112x112x16xf32>) outs(%10 : tensor<1x112x112x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.subf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x112x112x16xf32>
    %12 = linalg.init_tensor [1, 112, 112, 16] : tensor<1x112x112x16xf32>
    %13 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%11, %cst_0 : tensor<1x112x112x16xf32>, tensor<1x112x112x16xf32>) outs(%12 : tensor<1x112x112x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x112x112x16xf32>
    %14 = linalg.init_tensor [1, 112, 112, 16] : tensor<1x112x112x16xf32>
    %15 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%13, %cst : tensor<1x112x112x16xf32>, tensor<1x112x112x16xf32>) outs(%14 : tensor<1x112x112x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x112x112x16xf32>
    %16 = linalg.init_tensor [1, 112, 112, 16] : tensor<1x112x112x16xf32>
    %17 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%15, %cst_1 : tensor<1x112x112x16xf32>, tensor<1x112x112x16xf32>) outs(%16 : tensor<1x112x112x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x112x112x16xf32>
    %18 = linalg.init_tensor [1, 112, 112, 16] : tensor<1x112x112x16xf32>
    %19 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%17, %cst_168 : tensor<1x112x112x16xf32>, tensor<1x112x112x16xf32>) outs(%18 : tensor<1x112x112x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x112x112x16xf32>
    %20 = linalg.init_tensor [1, 112, 112, 16] : tensor<1x112x112x16xf32>
    %21 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %19, %cst_220 : tensor<f32>, tensor<1x112x112x16xf32>, tensor<f32>) outs(%20 : tensor<1x112x112x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      %885 = arith.minf %884, %arg3 : f32
      linalg.yield %885 : f32
    } -> tensor<1x112x112x16xf32>
    %22 = linalg.init_tensor [1, 112, 112, 16] : tensor<1x112x112x16xf32>
    %23 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%21, %cst_185 : tensor<1x112x112x16xf32>, tensor<1x112x112x16xf32>) outs(%22 : tensor<1x112x112x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x112x112x16xf32>
    %24 = linalg.init_tensor [1, 112, 112, 16] : tensor<1x112x112x16xf32>
    %25 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%23, %17 : tensor<1x112x112x16xf32>, tensor<1x112x112x16xf32>) outs(%24 : tensor<1x112x112x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x112x112x16xf32>
    %cst_269 = arith.constant 0.000000e+00 : f32
    %26 = linalg.init_tensor [1, 113, 113, 16] : tensor<1x113x113x16xf32>
    %27 = linalg.fill ins(%cst_269 : f32) outs(%26 : tensor<1x113x113x16xf32>) -> tensor<1x113x113x16xf32>
    %28 = tensor.insert_slice %25 into %27[0, 0, 0, 0] [1, 112, 112, 16] [1, 1, 1, 1] : tensor<1x112x112x16xf32> into tensor<1x113x113x16xf32>
    %29 = linalg.init_tensor [1, 56, 56, 16] : tensor<1x56x56x16xf32>
    %cst_270 = arith.constant 0.000000e+00 : f32
    %30 = linalg.fill ins(%cst_270 : f32) outs(%29 : tensor<1x56x56x16xf32>) -> tensor<1x56x56x16xf32>
    %31 = tensor.collapse_shape %cst_3 [[0], [1], [2, 3]] : tensor<3x3x1x16xf32> into tensor<3x3x16xf32>
    %32 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%28, %31 : tensor<1x113x113x16xf32>, tensor<3x3x16xf32>) outs(%30 : tensor<1x56x56x16xf32>) -> tensor<1x56x56x16xf32>
    %33 = linalg.init_tensor [1, 56, 56, 16] : tensor<1x56x56x16xf32>
    %34 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%32, %cst_7 : tensor<1x56x56x16xf32>, tensor<1x56x56x16xf32>) outs(%33 : tensor<1x56x56x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.subf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x56x56x16xf32>
    %35 = linalg.init_tensor [1, 56, 56, 16] : tensor<1x56x56x16xf32>
    %36 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%34, %cst_5 : tensor<1x56x56x16xf32>, tensor<1x56x56x16xf32>) outs(%35 : tensor<1x56x56x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x56x56x16xf32>
    %37 = linalg.init_tensor [1, 56, 56, 16] : tensor<1x56x56x16xf32>
    %38 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%36, %cst_4 : tensor<1x56x56x16xf32>, tensor<1x56x56x16xf32>) outs(%37 : tensor<1x56x56x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x56x56x16xf32>
    %39 = linalg.init_tensor [1, 56, 56, 16] : tensor<1x56x56x16xf32>
    %40 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%38, %cst_6 : tensor<1x56x56x16xf32>, tensor<1x56x56x16xf32>) outs(%39 : tensor<1x56x56x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x56x56x16xf32>
    %41 = linalg.init_tensor [1, 56, 56, 16] : tensor<1x56x56x16xf32>
    %42 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%40, %cst_202 : tensor<1x56x56x16xf32>, tensor<1x56x56x16xf32>) outs(%41 : tensor<1x56x56x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x56x56x16xf32>
    %cst_271 = arith.constant 0.000000e+00 : f32
    %43 = linalg.init_tensor [1, 16] : tensor<1x16xf32>
    %44 = linalg.fill ins(%cst_271 : f32) outs(%43 : tensor<1x16xf32>) -> tensor<1x16xf32>
    %45 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%42 : tensor<1x56x56x16xf32>) outs(%44 : tensor<1x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x16xf32>
    %46 = linalg.init_tensor [1, 16] : tensor<1x16xf32>
    %47 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%45, %cst_203 : tensor<1x16xf32>, tensor<1x16xf32>) outs(%46 : tensor<1x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x16xf32>
    %48 = tensor.expand_shape %47 [[0], [1, 2, 3]] : tensor<1x16xf32> into tensor<1x1x1x16xf32>
    %49 = linalg.init_tensor [1, 1, 1, 8] : tensor<1x1x1x8xf32>
    %cst_272 = arith.constant 0.000000e+00 : f32
    %50 = linalg.fill ins(%cst_272 : f32) outs(%49 : tensor<1x1x1x8xf32>) -> tensor<1x1x1x8xf32>
    %51 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%48, %cst_264 : tensor<1x1x1x16xf32>, tensor<1x1x16x8xf32>) outs(%50 : tensor<1x1x1x8xf32>) -> tensor<1x1x1x8xf32>
    %52 = linalg.init_tensor [1, 1, 1, 8] : tensor<1x1x1x8xf32>
    %53 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%51, %cst_8 : tensor<1x1x1x8xf32>, tensor<1x1x1x8xf32>) outs(%52 : tensor<1x1x1x8xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x8xf32>
    %54 = linalg.init_tensor [1, 1, 1, 8] : tensor<1x1x1x8xf32>
    %55 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%53, %cst_204 : tensor<1x1x1x8xf32>, tensor<1x1x1x8xf32>) outs(%54 : tensor<1x1x1x8xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x8xf32>
    %56 = linalg.init_tensor [1, 1, 1, 16] : tensor<1x1x1x16xf32>
    %cst_273 = arith.constant 0.000000e+00 : f32
    %57 = linalg.fill ins(%cst_273 : f32) outs(%56 : tensor<1x1x1x16xf32>) -> tensor<1x1x1x16xf32>
    %58 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%55, %cst_263 : tensor<1x1x1x8xf32>, tensor<1x1x8x16xf32>) outs(%57 : tensor<1x1x1x16xf32>) -> tensor<1x1x1x16xf32>
    %59 = linalg.init_tensor [1, 1, 1, 16] : tensor<1x1x1x16xf32>
    %60 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%58, %cst_9 : tensor<1x1x1x16xf32>, tensor<1x1x1x16xf32>) outs(%59 : tensor<1x1x1x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x16xf32>
    %61 = linalg.init_tensor [1, 1, 1, 16] : tensor<1x1x1x16xf32>
    %62 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%60, %cst_169 : tensor<1x1x1x16xf32>, tensor<1x1x1x16xf32>) outs(%61 : tensor<1x1x1x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x16xf32>
    %63 = linalg.init_tensor [1, 1, 1, 16] : tensor<1x1x1x16xf32>
    %64 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %62, %cst_220 : tensor<f32>, tensor<1x1x1x16xf32>, tensor<f32>) outs(%63 : tensor<1x1x1x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      %885 = arith.minf %884, %arg3 : f32
      linalg.yield %885 : f32
    } -> tensor<1x1x1x16xf32>
    %65 = linalg.init_tensor [1, 1, 1, 16] : tensor<1x1x1x16xf32>
    %66 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%64, %cst_186 : tensor<1x1x1x16xf32>, tensor<1x1x1x16xf32>) outs(%65 : tensor<1x1x1x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x16xf32>
    %67 = linalg.init_tensor [1, 56, 56, 16] : tensor<1x56x56x16xf32>
    %68 = linalg.generic {indexing_maps = [#map5, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%66 : tensor<1x1x1x16xf32>) outs(%67 : tensor<1x56x56x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<1x56x56x16xf32>
    %69 = linalg.init_tensor [1, 56, 56, 16] : tensor<1x56x56x16xf32>
    %70 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%42, %68 : tensor<1x56x56x16xf32>, tensor<1x56x56x16xf32>) outs(%69 : tensor<1x56x56x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x56x56x16xf32>
    %71 = linalg.init_tensor [1, 56, 56, 16] : tensor<1x56x56x16xf32>
    %cst_274 = arith.constant 0.000000e+00 : f32
    %72 = linalg.fill ins(%cst_274 : f32) outs(%71 : tensor<1x56x56x16xf32>) -> tensor<1x56x56x16xf32>
    %73 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%70, %cst_262 : tensor<1x56x56x16xf32>, tensor<1x1x16x16xf32>) outs(%72 : tensor<1x56x56x16xf32>) -> tensor<1x56x56x16xf32>
    %74 = linalg.init_tensor [1, 56, 56, 16] : tensor<1x56x56x16xf32>
    %75 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%73, %cst_13 : tensor<1x56x56x16xf32>, tensor<1x56x56x16xf32>) outs(%74 : tensor<1x56x56x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.subf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x56x56x16xf32>
    %76 = linalg.init_tensor [1, 56, 56, 16] : tensor<1x56x56x16xf32>
    %77 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%75, %cst_11 : tensor<1x56x56x16xf32>, tensor<1x56x56x16xf32>) outs(%76 : tensor<1x56x56x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x56x56x16xf32>
    %78 = linalg.init_tensor [1, 56, 56, 16] : tensor<1x56x56x16xf32>
    %79 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%77, %cst_10 : tensor<1x56x56x16xf32>, tensor<1x56x56x16xf32>) outs(%78 : tensor<1x56x56x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x56x56x16xf32>
    %80 = linalg.init_tensor [1, 56, 56, 16] : tensor<1x56x56x16xf32>
    %81 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%79, %cst_12 : tensor<1x56x56x16xf32>, tensor<1x56x56x16xf32>) outs(%80 : tensor<1x56x56x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x56x56x16xf32>
    %82 = linalg.init_tensor [1, 56, 56, 72] : tensor<1x56x56x72xf32>
    %cst_275 = arith.constant 0.000000e+00 : f32
    %83 = linalg.fill ins(%cst_275 : f32) outs(%82 : tensor<1x56x56x72xf32>) -> tensor<1x56x56x72xf32>
    %84 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%81, %cst_230 : tensor<1x56x56x16xf32>, tensor<1x1x16x72xf32>) outs(%83 : tensor<1x56x56x72xf32>) -> tensor<1x56x56x72xf32>
    %85 = linalg.init_tensor [1, 56, 56, 72] : tensor<1x56x56x72xf32>
    %86 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%84, %cst_17 : tensor<1x56x56x72xf32>, tensor<1x56x56x72xf32>) outs(%85 : tensor<1x56x56x72xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.subf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x56x56x72xf32>
    %87 = linalg.init_tensor [1, 56, 56, 72] : tensor<1x56x56x72xf32>
    %88 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%86, %cst_15 : tensor<1x56x56x72xf32>, tensor<1x56x56x72xf32>) outs(%87 : tensor<1x56x56x72xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x56x56x72xf32>
    %89 = linalg.init_tensor [1, 56, 56, 72] : tensor<1x56x56x72xf32>
    %90 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%88, %cst_14 : tensor<1x56x56x72xf32>, tensor<1x56x56x72xf32>) outs(%89 : tensor<1x56x56x72xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x56x56x72xf32>
    %91 = linalg.init_tensor [1, 56, 56, 72] : tensor<1x56x56x72xf32>
    %92 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%90, %cst_16 : tensor<1x56x56x72xf32>, tensor<1x56x56x72xf32>) outs(%91 : tensor<1x56x56x72xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x56x56x72xf32>
    %93 = linalg.init_tensor [1, 56, 56, 72] : tensor<1x56x56x72xf32>
    %94 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%92, %cst_205 : tensor<1x56x56x72xf32>, tensor<1x56x56x72xf32>) outs(%93 : tensor<1x56x56x72xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x56x56x72xf32>
    %cst_276 = arith.constant 0.000000e+00 : f32
    %95 = linalg.init_tensor [1, 57, 57, 72] : tensor<1x57x57x72xf32>
    %96 = linalg.fill ins(%cst_276 : f32) outs(%95 : tensor<1x57x57x72xf32>) -> tensor<1x57x57x72xf32>
    %97 = tensor.insert_slice %94 into %96[0, 0, 0, 0] [1, 56, 56, 72] [1, 1, 1, 1] : tensor<1x56x56x72xf32> into tensor<1x57x57x72xf32>
    %98 = linalg.init_tensor [1, 28, 28, 72] : tensor<1x28x28x72xf32>
    %cst_277 = arith.constant 0.000000e+00 : f32
    %99 = linalg.fill ins(%cst_277 : f32) outs(%98 : tensor<1x28x28x72xf32>) -> tensor<1x28x28x72xf32>
    %100 = tensor.collapse_shape %cst_18 [[0], [1], [2, 3]] : tensor<3x3x1x72xf32> into tensor<3x3x72xf32>
    %101 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%97, %100 : tensor<1x57x57x72xf32>, tensor<3x3x72xf32>) outs(%99 : tensor<1x28x28x72xf32>) -> tensor<1x28x28x72xf32>
    %102 = linalg.init_tensor [1, 28, 28, 72] : tensor<1x28x28x72xf32>
    %103 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%101, %cst_22 : tensor<1x28x28x72xf32>, tensor<1x28x28x72xf32>) outs(%102 : tensor<1x28x28x72xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.subf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x28x28x72xf32>
    %104 = linalg.init_tensor [1, 28, 28, 72] : tensor<1x28x28x72xf32>
    %105 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%103, %cst_20 : tensor<1x28x28x72xf32>, tensor<1x28x28x72xf32>) outs(%104 : tensor<1x28x28x72xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x28x28x72xf32>
    %106 = linalg.init_tensor [1, 28, 28, 72] : tensor<1x28x28x72xf32>
    %107 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%105, %cst_19 : tensor<1x28x28x72xf32>, tensor<1x28x28x72xf32>) outs(%106 : tensor<1x28x28x72xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x28x28x72xf32>
    %108 = linalg.init_tensor [1, 28, 28, 72] : tensor<1x28x28x72xf32>
    %109 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%107, %cst_21 : tensor<1x28x28x72xf32>, tensor<1x28x28x72xf32>) outs(%108 : tensor<1x28x28x72xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x28x28x72xf32>
    %110 = linalg.init_tensor [1, 28, 28, 72] : tensor<1x28x28x72xf32>
    %111 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%109, %cst_206 : tensor<1x28x28x72xf32>, tensor<1x28x28x72xf32>) outs(%110 : tensor<1x28x28x72xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x28x28x72xf32>
    %112 = linalg.init_tensor [1, 28, 28, 24] : tensor<1x28x28x24xf32>
    %cst_278 = arith.constant 0.000000e+00 : f32
    %113 = linalg.fill ins(%cst_278 : f32) outs(%112 : tensor<1x28x28x24xf32>) -> tensor<1x28x28x24xf32>
    %114 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%111, %cst_231 : tensor<1x28x28x72xf32>, tensor<1x1x72x24xf32>) outs(%113 : tensor<1x28x28x24xf32>) -> tensor<1x28x28x24xf32>
    %115 = linalg.init_tensor [1, 28, 28, 24] : tensor<1x28x28x24xf32>
    %116 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%114, %cst_26 : tensor<1x28x28x24xf32>, tensor<1x28x28x24xf32>) outs(%115 : tensor<1x28x28x24xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.subf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x28x28x24xf32>
    %117 = linalg.init_tensor [1, 28, 28, 24] : tensor<1x28x28x24xf32>
    %118 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%116, %cst_24 : tensor<1x28x28x24xf32>, tensor<1x28x28x24xf32>) outs(%117 : tensor<1x28x28x24xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x28x28x24xf32>
    %119 = linalg.init_tensor [1, 28, 28, 24] : tensor<1x28x28x24xf32>
    %120 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%118, %cst_23 : tensor<1x28x28x24xf32>, tensor<1x28x28x24xf32>) outs(%119 : tensor<1x28x28x24xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x28x28x24xf32>
    %121 = linalg.init_tensor [1, 28, 28, 24] : tensor<1x28x28x24xf32>
    %122 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%120, %cst_25 : tensor<1x28x28x24xf32>, tensor<1x28x28x24xf32>) outs(%121 : tensor<1x28x28x24xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x28x28x24xf32>
    %123 = linalg.init_tensor [1, 28, 28, 88] : tensor<1x28x28x88xf32>
    %cst_279 = arith.constant 0.000000e+00 : f32
    %124 = linalg.fill ins(%cst_279 : f32) outs(%123 : tensor<1x28x28x88xf32>) -> tensor<1x28x28x88xf32>
    %125 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%122, %cst_232 : tensor<1x28x28x24xf32>, tensor<1x1x24x88xf32>) outs(%124 : tensor<1x28x28x88xf32>) -> tensor<1x28x28x88xf32>
    %126 = linalg.init_tensor [1, 28, 28, 88] : tensor<1x28x28x88xf32>
    %127 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%125, %cst_30 : tensor<1x28x28x88xf32>, tensor<1x28x28x88xf32>) outs(%126 : tensor<1x28x28x88xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.subf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x28x28x88xf32>
    %128 = linalg.init_tensor [1, 28, 28, 88] : tensor<1x28x28x88xf32>
    %129 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%127, %cst_28 : tensor<1x28x28x88xf32>, tensor<1x28x28x88xf32>) outs(%128 : tensor<1x28x28x88xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x28x28x88xf32>
    %130 = linalg.init_tensor [1, 28, 28, 88] : tensor<1x28x28x88xf32>
    %131 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%129, %cst_27 : tensor<1x28x28x88xf32>, tensor<1x28x28x88xf32>) outs(%130 : tensor<1x28x28x88xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x28x28x88xf32>
    %132 = linalg.init_tensor [1, 28, 28, 88] : tensor<1x28x28x88xf32>
    %133 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%131, %cst_29 : tensor<1x28x28x88xf32>, tensor<1x28x28x88xf32>) outs(%132 : tensor<1x28x28x88xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x28x28x88xf32>
    %134 = linalg.init_tensor [1, 28, 28, 88] : tensor<1x28x28x88xf32>
    %135 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%133, %cst_207 : tensor<1x28x28x88xf32>, tensor<1x28x28x88xf32>) outs(%134 : tensor<1x28x28x88xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x28x28x88xf32>
    %cst_280 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_281 = arith.constant 0.000000e+00 : f32
    %136 = linalg.init_tensor [1, 30, 30, 88] : tensor<1x30x30x88xf32>
    %137 = linalg.fill ins(%cst_281 : f32) outs(%136 : tensor<1x30x30x88xf32>) -> tensor<1x30x30x88xf32>
    %138 = tensor.insert_slice %135 into %137[0, 1, 1, 0] [1, 28, 28, 88] [1, 1, 1, 1] : tensor<1x28x28x88xf32> into tensor<1x30x30x88xf32>
    %139 = linalg.init_tensor [1, 28, 28, 88] : tensor<1x28x28x88xf32>
    %cst_282 = arith.constant 0.000000e+00 : f32
    %140 = linalg.fill ins(%cst_282 : f32) outs(%139 : tensor<1x28x28x88xf32>) -> tensor<1x28x28x88xf32>
    %141 = tensor.collapse_shape %cst_31 [[0], [1], [2, 3]] : tensor<3x3x1x88xf32> into tensor<3x3x88xf32>
    %142 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%138, %141 : tensor<1x30x30x88xf32>, tensor<3x3x88xf32>) outs(%140 : tensor<1x28x28x88xf32>) -> tensor<1x28x28x88xf32>
    %143 = linalg.init_tensor [1, 28, 28, 88] : tensor<1x28x28x88xf32>
    %144 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%142, %cst_35 : tensor<1x28x28x88xf32>, tensor<1x28x28x88xf32>) outs(%143 : tensor<1x28x28x88xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.subf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x28x28x88xf32>
    %145 = linalg.init_tensor [1, 28, 28, 88] : tensor<1x28x28x88xf32>
    %146 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%144, %cst_33 : tensor<1x28x28x88xf32>, tensor<1x28x28x88xf32>) outs(%145 : tensor<1x28x28x88xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x28x28x88xf32>
    %147 = linalg.init_tensor [1, 28, 28, 88] : tensor<1x28x28x88xf32>
    %148 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%146, %cst_32 : tensor<1x28x28x88xf32>, tensor<1x28x28x88xf32>) outs(%147 : tensor<1x28x28x88xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x28x28x88xf32>
    %149 = linalg.init_tensor [1, 28, 28, 88] : tensor<1x28x28x88xf32>
    %150 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%148, %cst_34 : tensor<1x28x28x88xf32>, tensor<1x28x28x88xf32>) outs(%149 : tensor<1x28x28x88xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x28x28x88xf32>
    %151 = linalg.init_tensor [1, 28, 28, 88] : tensor<1x28x28x88xf32>
    %152 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%150, %cst_207 : tensor<1x28x28x88xf32>, tensor<1x28x28x88xf32>) outs(%151 : tensor<1x28x28x88xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x28x28x88xf32>
    %153 = linalg.init_tensor [1, 28, 28, 24] : tensor<1x28x28x24xf32>
    %cst_283 = arith.constant 0.000000e+00 : f32
    %154 = linalg.fill ins(%cst_283 : f32) outs(%153 : tensor<1x28x28x24xf32>) -> tensor<1x28x28x24xf32>
    %155 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%152, %cst_233 : tensor<1x28x28x88xf32>, tensor<1x1x88x24xf32>) outs(%154 : tensor<1x28x28x24xf32>) -> tensor<1x28x28x24xf32>
    %156 = linalg.init_tensor [1, 28, 28, 24] : tensor<1x28x28x24xf32>
    %157 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%155, %cst_39 : tensor<1x28x28x24xf32>, tensor<1x28x28x24xf32>) outs(%156 : tensor<1x28x28x24xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.subf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x28x28x24xf32>
    %158 = linalg.init_tensor [1, 28, 28, 24] : tensor<1x28x28x24xf32>
    %159 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%157, %cst_37 : tensor<1x28x28x24xf32>, tensor<1x28x28x24xf32>) outs(%158 : tensor<1x28x28x24xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x28x28x24xf32>
    %160 = linalg.init_tensor [1, 28, 28, 24] : tensor<1x28x28x24xf32>
    %161 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%159, %cst_36 : tensor<1x28x28x24xf32>, tensor<1x28x28x24xf32>) outs(%160 : tensor<1x28x28x24xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x28x28x24xf32>
    %162 = linalg.init_tensor [1, 28, 28, 24] : tensor<1x28x28x24xf32>
    %163 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%161, %cst_38 : tensor<1x28x28x24xf32>, tensor<1x28x28x24xf32>) outs(%162 : tensor<1x28x28x24xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x28x28x24xf32>
    %164 = linalg.init_tensor [1, 28, 28, 24] : tensor<1x28x28x24xf32>
    %165 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%122, %163 : tensor<1x28x28x24xf32>, tensor<1x28x28x24xf32>) outs(%164 : tensor<1x28x28x24xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x28x28x24xf32>
    %166 = linalg.init_tensor [1, 28, 28, 96] : tensor<1x28x28x96xf32>
    %cst_284 = arith.constant 0.000000e+00 : f32
    %167 = linalg.fill ins(%cst_284 : f32) outs(%166 : tensor<1x28x28x96xf32>) -> tensor<1x28x28x96xf32>
    %168 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%165, %cst_234 : tensor<1x28x28x24xf32>, tensor<1x1x24x96xf32>) outs(%167 : tensor<1x28x28x96xf32>) -> tensor<1x28x28x96xf32>
    %169 = linalg.init_tensor [1, 28, 28, 96] : tensor<1x28x28x96xf32>
    %170 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%168, %cst_43 : tensor<1x28x28x96xf32>, tensor<1x28x28x96xf32>) outs(%169 : tensor<1x28x28x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.subf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x28x28x96xf32>
    %171 = linalg.init_tensor [1, 28, 28, 96] : tensor<1x28x28x96xf32>
    %172 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%170, %cst_41 : tensor<1x28x28x96xf32>, tensor<1x28x28x96xf32>) outs(%171 : tensor<1x28x28x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x28x28x96xf32>
    %173 = linalg.init_tensor [1, 28, 28, 96] : tensor<1x28x28x96xf32>
    %174 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%172, %cst_40 : tensor<1x28x28x96xf32>, tensor<1x28x28x96xf32>) outs(%173 : tensor<1x28x28x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x28x28x96xf32>
    %175 = linalg.init_tensor [1, 28, 28, 96] : tensor<1x28x28x96xf32>
    %176 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%174, %cst_42 : tensor<1x28x28x96xf32>, tensor<1x28x28x96xf32>) outs(%175 : tensor<1x28x28x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x28x28x96xf32>
    %177 = linalg.init_tensor [1, 28, 28, 96] : tensor<1x28x28x96xf32>
    %178 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%176, %cst_170 : tensor<1x28x28x96xf32>, tensor<1x28x28x96xf32>) outs(%177 : tensor<1x28x28x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x28x28x96xf32>
    %179 = linalg.init_tensor [1, 28, 28, 96] : tensor<1x28x28x96xf32>
    %180 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %178, %cst_220 : tensor<f32>, tensor<1x28x28x96xf32>, tensor<f32>) outs(%179 : tensor<1x28x28x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      %885 = arith.minf %884, %arg3 : f32
      linalg.yield %885 : f32
    } -> tensor<1x28x28x96xf32>
    %181 = linalg.init_tensor [1, 28, 28, 96] : tensor<1x28x28x96xf32>
    %182 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%180, %cst_187 : tensor<1x28x28x96xf32>, tensor<1x28x28x96xf32>) outs(%181 : tensor<1x28x28x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x28x28x96xf32>
    %183 = linalg.init_tensor [1, 28, 28, 96] : tensor<1x28x28x96xf32>
    %184 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%182, %176 : tensor<1x28x28x96xf32>, tensor<1x28x28x96xf32>) outs(%183 : tensor<1x28x28x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x28x28x96xf32>
    %cst_285 = arith.constant 0.000000e+00 : f32
    %185 = linalg.init_tensor [1, 31, 31, 96] : tensor<1x31x31x96xf32>
    %186 = linalg.fill ins(%cst_285 : f32) outs(%185 : tensor<1x31x31x96xf32>) -> tensor<1x31x31x96xf32>
    %187 = tensor.insert_slice %184 into %186[0, 1, 1, 0] [1, 28, 28, 96] [1, 1, 1, 1] : tensor<1x28x28x96xf32> into tensor<1x31x31x96xf32>
    %188 = linalg.init_tensor [1, 14, 14, 96] : tensor<1x14x14x96xf32>
    %cst_286 = arith.constant 0.000000e+00 : f32
    %189 = linalg.fill ins(%cst_286 : f32) outs(%188 : tensor<1x14x14x96xf32>) -> tensor<1x14x14x96xf32>
    %190 = tensor.collapse_shape %cst_44 [[0], [1], [2, 3]] : tensor<5x5x1x96xf32> into tensor<5x5x96xf32>
    %191 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%187, %190 : tensor<1x31x31x96xf32>, tensor<5x5x96xf32>) outs(%189 : tensor<1x14x14x96xf32>) -> tensor<1x14x14x96xf32>
    %192 = linalg.init_tensor [1, 14, 14, 96] : tensor<1x14x14x96xf32>
    %193 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%191, %cst_48 : tensor<1x14x14x96xf32>, tensor<1x14x14x96xf32>) outs(%192 : tensor<1x14x14x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.subf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x96xf32>
    %194 = linalg.init_tensor [1, 14, 14, 96] : tensor<1x14x14x96xf32>
    %195 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%193, %cst_46 : tensor<1x14x14x96xf32>, tensor<1x14x14x96xf32>) outs(%194 : tensor<1x14x14x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x96xf32>
    %196 = linalg.init_tensor [1, 14, 14, 96] : tensor<1x14x14x96xf32>
    %197 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%195, %cst_45 : tensor<1x14x14x96xf32>, tensor<1x14x14x96xf32>) outs(%196 : tensor<1x14x14x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x96xf32>
    %198 = linalg.init_tensor [1, 14, 14, 96] : tensor<1x14x14x96xf32>
    %199 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%197, %cst_47 : tensor<1x14x14x96xf32>, tensor<1x14x14x96xf32>) outs(%198 : tensor<1x14x14x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x96xf32>
    %200 = linalg.init_tensor [1, 14, 14, 96] : tensor<1x14x14x96xf32>
    %201 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%199, %cst_171 : tensor<1x14x14x96xf32>, tensor<1x14x14x96xf32>) outs(%200 : tensor<1x14x14x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x96xf32>
    %202 = linalg.init_tensor [1, 14, 14, 96] : tensor<1x14x14x96xf32>
    %203 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %201, %cst_220 : tensor<f32>, tensor<1x14x14x96xf32>, tensor<f32>) outs(%202 : tensor<1x14x14x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      %885 = arith.minf %884, %arg3 : f32
      linalg.yield %885 : f32
    } -> tensor<1x14x14x96xf32>
    %204 = linalg.init_tensor [1, 14, 14, 96] : tensor<1x14x14x96xf32>
    %205 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%203, %cst_188 : tensor<1x14x14x96xf32>, tensor<1x14x14x96xf32>) outs(%204 : tensor<1x14x14x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x96xf32>
    %206 = linalg.init_tensor [1, 14, 14, 96] : tensor<1x14x14x96xf32>
    %207 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%205, %199 : tensor<1x14x14x96xf32>, tensor<1x14x14x96xf32>) outs(%206 : tensor<1x14x14x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x96xf32>
    %cst_287 = arith.constant 0.000000e+00 : f32
    %208 = linalg.init_tensor [1, 96] : tensor<1x96xf32>
    %209 = linalg.fill ins(%cst_287 : f32) outs(%208 : tensor<1x96xf32>) -> tensor<1x96xf32>
    %210 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%207 : tensor<1x14x14x96xf32>) outs(%209 : tensor<1x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x96xf32>
    %211 = linalg.init_tensor [1, 96] : tensor<1x96xf32>
    %212 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%210, %cst_208 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%211 : tensor<1x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x96xf32>
    %213 = tensor.expand_shape %212 [[0], [1, 2, 3]] : tensor<1x96xf32> into tensor<1x1x1x96xf32>
    %214 = linalg.init_tensor [1, 1, 1, 24] : tensor<1x1x1x24xf32>
    %cst_288 = arith.constant 0.000000e+00 : f32
    %215 = linalg.fill ins(%cst_288 : f32) outs(%214 : tensor<1x1x1x24xf32>) -> tensor<1x1x1x24xf32>
    %216 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%213, %cst_237 : tensor<1x1x1x96xf32>, tensor<1x1x96x24xf32>) outs(%215 : tensor<1x1x1x24xf32>) -> tensor<1x1x1x24xf32>
    %217 = linalg.init_tensor [1, 1, 1, 24] : tensor<1x1x1x24xf32>
    %218 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%216, %cst_49 : tensor<1x1x1x24xf32>, tensor<1x1x1x24xf32>) outs(%217 : tensor<1x1x1x24xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x24xf32>
    %219 = linalg.init_tensor [1, 1, 1, 24] : tensor<1x1x1x24xf32>
    %220 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%218, %cst_209 : tensor<1x1x1x24xf32>, tensor<1x1x1x24xf32>) outs(%219 : tensor<1x1x1x24xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x24xf32>
    %221 = linalg.init_tensor [1, 1, 1, 96] : tensor<1x1x1x96xf32>
    %cst_289 = arith.constant 0.000000e+00 : f32
    %222 = linalg.fill ins(%cst_289 : f32) outs(%221 : tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32>
    %223 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%220, %cst_236 : tensor<1x1x1x24xf32>, tensor<1x1x24x96xf32>) outs(%222 : tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32>
    %224 = linalg.init_tensor [1, 1, 1, 96] : tensor<1x1x1x96xf32>
    %225 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%223, %cst_50 : tensor<1x1x1x96xf32>, tensor<1x1x1x96xf32>) outs(%224 : tensor<1x1x1x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x96xf32>
    %226 = linalg.init_tensor [1, 1, 1, 96] : tensor<1x1x1x96xf32>
    %227 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%225, %cst_172 : tensor<1x1x1x96xf32>, tensor<1x1x1x96xf32>) outs(%226 : tensor<1x1x1x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x96xf32>
    %228 = linalg.init_tensor [1, 1, 1, 96] : tensor<1x1x1x96xf32>
    %229 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %227, %cst_220 : tensor<f32>, tensor<1x1x1x96xf32>, tensor<f32>) outs(%228 : tensor<1x1x1x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      %885 = arith.minf %884, %arg3 : f32
      linalg.yield %885 : f32
    } -> tensor<1x1x1x96xf32>
    %230 = linalg.init_tensor [1, 1, 1, 96] : tensor<1x1x1x96xf32>
    %231 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%229, %cst_189 : tensor<1x1x1x96xf32>, tensor<1x1x1x96xf32>) outs(%230 : tensor<1x1x1x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x96xf32>
    %232 = linalg.init_tensor [1, 14, 14, 96] : tensor<1x14x14x96xf32>
    %233 = linalg.generic {indexing_maps = [#map5, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%231 : tensor<1x1x1x96xf32>) outs(%232 : tensor<1x14x14x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<1x14x14x96xf32>
    %234 = linalg.init_tensor [1, 14, 14, 96] : tensor<1x14x14x96xf32>
    %235 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%207, %233 : tensor<1x14x14x96xf32>, tensor<1x14x14x96xf32>) outs(%234 : tensor<1x14x14x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x96xf32>
    %236 = linalg.init_tensor [1, 14, 14, 40] : tensor<1x14x14x40xf32>
    %cst_290 = arith.constant 0.000000e+00 : f32
    %237 = linalg.fill ins(%cst_290 : f32) outs(%236 : tensor<1x14x14x40xf32>) -> tensor<1x14x14x40xf32>
    %238 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%235, %cst_235 : tensor<1x14x14x96xf32>, tensor<1x1x96x40xf32>) outs(%237 : tensor<1x14x14x40xf32>) -> tensor<1x14x14x40xf32>
    %239 = linalg.init_tensor [1, 14, 14, 40] : tensor<1x14x14x40xf32>
    %240 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%238, %cst_54 : tensor<1x14x14x40xf32>, tensor<1x14x14x40xf32>) outs(%239 : tensor<1x14x14x40xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.subf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x40xf32>
    %241 = linalg.init_tensor [1, 14, 14, 40] : tensor<1x14x14x40xf32>
    %242 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%240, %cst_52 : tensor<1x14x14x40xf32>, tensor<1x14x14x40xf32>) outs(%241 : tensor<1x14x14x40xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x40xf32>
    %243 = linalg.init_tensor [1, 14, 14, 40] : tensor<1x14x14x40xf32>
    %244 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%242, %cst_51 : tensor<1x14x14x40xf32>, tensor<1x14x14x40xf32>) outs(%243 : tensor<1x14x14x40xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x40xf32>
    %245 = linalg.init_tensor [1, 14, 14, 40] : tensor<1x14x14x40xf32>
    %246 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%244, %cst_53 : tensor<1x14x14x40xf32>, tensor<1x14x14x40xf32>) outs(%245 : tensor<1x14x14x40xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x40xf32>
    %247 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %cst_291 = arith.constant 0.000000e+00 : f32
    %248 = linalg.fill ins(%cst_291 : f32) outs(%247 : tensor<1x14x14x240xf32>) -> tensor<1x14x14x240xf32>
    %249 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%246, %cst_238 : tensor<1x14x14x40xf32>, tensor<1x1x40x240xf32>) outs(%248 : tensor<1x14x14x240xf32>) -> tensor<1x14x14x240xf32>
    %250 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %251 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%249, %cst_58 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%250 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.subf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x240xf32>
    %252 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %253 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%251, %cst_56 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%252 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x240xf32>
    %254 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %255 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%253, %cst_55 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%254 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x240xf32>
    %256 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %257 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%255, %cst_57 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%256 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x240xf32>
    %258 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %259 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%257, %cst_173 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%258 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x240xf32>
    %260 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %261 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %259, %cst_220 : tensor<f32>, tensor<1x14x14x240xf32>, tensor<f32>) outs(%260 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      %885 = arith.minf %884, %arg3 : f32
      linalg.yield %885 : f32
    } -> tensor<1x14x14x240xf32>
    %262 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %263 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%261, %cst_190 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%262 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x240xf32>
    %264 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %265 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%263, %257 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%264 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x240xf32>
    %cst_292 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_293 = arith.constant 0.000000e+00 : f32
    %266 = linalg.init_tensor [1, 18, 18, 240] : tensor<1x18x18x240xf32>
    %267 = linalg.fill ins(%cst_293 : f32) outs(%266 : tensor<1x18x18x240xf32>) -> tensor<1x18x18x240xf32>
    %268 = tensor.insert_slice %265 into %267[0, 2, 2, 0] [1, 14, 14, 240] [1, 1, 1, 1] : tensor<1x14x14x240xf32> into tensor<1x18x18x240xf32>
    %269 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %cst_294 = arith.constant 0.000000e+00 : f32
    %270 = linalg.fill ins(%cst_294 : f32) outs(%269 : tensor<1x14x14x240xf32>) -> tensor<1x14x14x240xf32>
    %271 = tensor.collapse_shape %cst_59 [[0], [1], [2, 3]] : tensor<5x5x1x240xf32> into tensor<5x5x240xf32>
    %272 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%268, %271 : tensor<1x18x18x240xf32>, tensor<5x5x240xf32>) outs(%270 : tensor<1x14x14x240xf32>) -> tensor<1x14x14x240xf32>
    %273 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %274 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%272, %cst_63 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%273 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.subf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x240xf32>
    %275 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %276 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%274, %cst_61 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%275 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x240xf32>
    %277 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %278 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%276, %cst_60 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%277 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x240xf32>
    %279 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %280 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%278, %cst_62 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%279 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x240xf32>
    %281 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %282 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%280, %cst_173 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%281 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x240xf32>
    %283 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %284 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %282, %cst_220 : tensor<f32>, tensor<1x14x14x240xf32>, tensor<f32>) outs(%283 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      %885 = arith.minf %884, %arg3 : f32
      linalg.yield %885 : f32
    } -> tensor<1x14x14x240xf32>
    %285 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %286 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%284, %cst_190 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%285 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x240xf32>
    %287 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %288 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%286, %280 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%287 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x240xf32>
    %cst_295 = arith.constant 0.000000e+00 : f32
    %289 = linalg.init_tensor [1, 240] : tensor<1x240xf32>
    %290 = linalg.fill ins(%cst_295 : f32) outs(%289 : tensor<1x240xf32>) -> tensor<1x240xf32>
    %291 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%288 : tensor<1x14x14x240xf32>) outs(%290 : tensor<1x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x240xf32>
    %292 = linalg.init_tensor [1, 240] : tensor<1x240xf32>
    %293 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%291, %cst_210 : tensor<1x240xf32>, tensor<1x240xf32>) outs(%292 : tensor<1x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x240xf32>
    %294 = tensor.expand_shape %293 [[0], [1, 2, 3]] : tensor<1x240xf32> into tensor<1x1x1x240xf32>
    %295 = linalg.init_tensor [1, 1, 1, 64] : tensor<1x1x1x64xf32>
    %cst_296 = arith.constant 0.000000e+00 : f32
    %296 = linalg.fill ins(%cst_296 : f32) outs(%295 : tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xf32>
    %297 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%294, %cst_241 : tensor<1x1x1x240xf32>, tensor<1x1x240x64xf32>) outs(%296 : tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xf32>
    %298 = linalg.init_tensor [1, 1, 1, 64] : tensor<1x1x1x64xf32>
    %299 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%297, %cst_64 : tensor<1x1x1x64xf32>, tensor<1x1x1x64xf32>) outs(%298 : tensor<1x1x1x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x64xf32>
    %300 = linalg.init_tensor [1, 1, 1, 64] : tensor<1x1x1x64xf32>
    %301 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%299, %cst_211 : tensor<1x1x1x64xf32>, tensor<1x1x1x64xf32>) outs(%300 : tensor<1x1x1x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x64xf32>
    %302 = linalg.init_tensor [1, 1, 1, 240] : tensor<1x1x1x240xf32>
    %cst_297 = arith.constant 0.000000e+00 : f32
    %303 = linalg.fill ins(%cst_297 : f32) outs(%302 : tensor<1x1x1x240xf32>) -> tensor<1x1x1x240xf32>
    %304 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%301, %cst_240 : tensor<1x1x1x64xf32>, tensor<1x1x64x240xf32>) outs(%303 : tensor<1x1x1x240xf32>) -> tensor<1x1x1x240xf32>
    %305 = linalg.init_tensor [1, 1, 1, 240] : tensor<1x1x1x240xf32>
    %306 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%304, %cst_65 : tensor<1x1x1x240xf32>, tensor<1x1x1x240xf32>) outs(%305 : tensor<1x1x1x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x240xf32>
    %307 = linalg.init_tensor [1, 1, 1, 240] : tensor<1x1x1x240xf32>
    %308 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%306, %cst_174 : tensor<1x1x1x240xf32>, tensor<1x1x1x240xf32>) outs(%307 : tensor<1x1x1x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x240xf32>
    %309 = linalg.init_tensor [1, 1, 1, 240] : tensor<1x1x1x240xf32>
    %310 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %308, %cst_220 : tensor<f32>, tensor<1x1x1x240xf32>, tensor<f32>) outs(%309 : tensor<1x1x1x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      %885 = arith.minf %884, %arg3 : f32
      linalg.yield %885 : f32
    } -> tensor<1x1x1x240xf32>
    %311 = linalg.init_tensor [1, 1, 1, 240] : tensor<1x1x1x240xf32>
    %312 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%310, %cst_191 : tensor<1x1x1x240xf32>, tensor<1x1x1x240xf32>) outs(%311 : tensor<1x1x1x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x240xf32>
    %313 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %314 = linalg.generic {indexing_maps = [#map5, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%312 : tensor<1x1x1x240xf32>) outs(%313 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<1x14x14x240xf32>
    %315 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %316 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%288, %314 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%315 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x240xf32>
    %317 = linalg.init_tensor [1, 14, 14, 40] : tensor<1x14x14x40xf32>
    %cst_298 = arith.constant 0.000000e+00 : f32
    %318 = linalg.fill ins(%cst_298 : f32) outs(%317 : tensor<1x14x14x40xf32>) -> tensor<1x14x14x40xf32>
    %319 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%316, %cst_239 : tensor<1x14x14x240xf32>, tensor<1x1x240x40xf32>) outs(%318 : tensor<1x14x14x40xf32>) -> tensor<1x14x14x40xf32>
    %320 = linalg.init_tensor [1, 14, 14, 40] : tensor<1x14x14x40xf32>
    %321 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%319, %cst_69 : tensor<1x14x14x40xf32>, tensor<1x14x14x40xf32>) outs(%320 : tensor<1x14x14x40xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.subf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x40xf32>
    %322 = linalg.init_tensor [1, 14, 14, 40] : tensor<1x14x14x40xf32>
    %323 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%321, %cst_67 : tensor<1x14x14x40xf32>, tensor<1x14x14x40xf32>) outs(%322 : tensor<1x14x14x40xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x40xf32>
    %324 = linalg.init_tensor [1, 14, 14, 40] : tensor<1x14x14x40xf32>
    %325 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%323, %cst_66 : tensor<1x14x14x40xf32>, tensor<1x14x14x40xf32>) outs(%324 : tensor<1x14x14x40xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x40xf32>
    %326 = linalg.init_tensor [1, 14, 14, 40] : tensor<1x14x14x40xf32>
    %327 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%325, %cst_68 : tensor<1x14x14x40xf32>, tensor<1x14x14x40xf32>) outs(%326 : tensor<1x14x14x40xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x40xf32>
    %328 = linalg.init_tensor [1, 14, 14, 40] : tensor<1x14x14x40xf32>
    %329 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%246, %327 : tensor<1x14x14x40xf32>, tensor<1x14x14x40xf32>) outs(%328 : tensor<1x14x14x40xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x40xf32>
    %330 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %cst_299 = arith.constant 0.000000e+00 : f32
    %331 = linalg.fill ins(%cst_299 : f32) outs(%330 : tensor<1x14x14x240xf32>) -> tensor<1x14x14x240xf32>
    %332 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%329, %cst_242 : tensor<1x14x14x40xf32>, tensor<1x1x40x240xf32>) outs(%331 : tensor<1x14x14x240xf32>) -> tensor<1x14x14x240xf32>
    %333 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %334 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%332, %cst_73 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%333 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.subf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x240xf32>
    %335 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %336 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%334, %cst_71 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%335 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x240xf32>
    %337 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %338 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%336, %cst_70 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%337 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x240xf32>
    %339 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %340 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%338, %cst_72 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%339 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x240xf32>
    %341 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %342 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%340, %cst_173 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%341 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x240xf32>
    %343 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %344 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %342, %cst_220 : tensor<f32>, tensor<1x14x14x240xf32>, tensor<f32>) outs(%343 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      %885 = arith.minf %884, %arg3 : f32
      linalg.yield %885 : f32
    } -> tensor<1x14x14x240xf32>
    %345 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %346 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%344, %cst_190 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%345 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x240xf32>
    %347 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %348 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%346, %340 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%347 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x240xf32>
    %cst_300 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_301 = arith.constant 0.000000e+00 : f32
    %349 = linalg.init_tensor [1, 18, 18, 240] : tensor<1x18x18x240xf32>
    %350 = linalg.fill ins(%cst_301 : f32) outs(%349 : tensor<1x18x18x240xf32>) -> tensor<1x18x18x240xf32>
    %351 = tensor.insert_slice %348 into %350[0, 2, 2, 0] [1, 14, 14, 240] [1, 1, 1, 1] : tensor<1x14x14x240xf32> into tensor<1x18x18x240xf32>
    %352 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %cst_302 = arith.constant 0.000000e+00 : f32
    %353 = linalg.fill ins(%cst_302 : f32) outs(%352 : tensor<1x14x14x240xf32>) -> tensor<1x14x14x240xf32>
    %354 = tensor.collapse_shape %cst_74 [[0], [1], [2, 3]] : tensor<5x5x1x240xf32> into tensor<5x5x240xf32>
    %355 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%351, %354 : tensor<1x18x18x240xf32>, tensor<5x5x240xf32>) outs(%353 : tensor<1x14x14x240xf32>) -> tensor<1x14x14x240xf32>
    %356 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %357 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%355, %cst_78 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%356 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.subf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x240xf32>
    %358 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %359 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%357, %cst_76 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%358 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x240xf32>
    %360 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %361 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%359, %cst_75 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%360 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x240xf32>
    %362 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %363 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%361, %cst_77 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%362 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x240xf32>
    %364 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %365 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%363, %cst_173 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%364 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x240xf32>
    %366 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %367 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %365, %cst_220 : tensor<f32>, tensor<1x14x14x240xf32>, tensor<f32>) outs(%366 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      %885 = arith.minf %884, %arg3 : f32
      linalg.yield %885 : f32
    } -> tensor<1x14x14x240xf32>
    %368 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %369 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%367, %cst_190 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%368 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x240xf32>
    %370 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %371 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%369, %363 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%370 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x240xf32>
    %cst_303 = arith.constant 0.000000e+00 : f32
    %372 = linalg.init_tensor [1, 240] : tensor<1x240xf32>
    %373 = linalg.fill ins(%cst_303 : f32) outs(%372 : tensor<1x240xf32>) -> tensor<1x240xf32>
    %374 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%371 : tensor<1x14x14x240xf32>) outs(%373 : tensor<1x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x240xf32>
    %375 = linalg.init_tensor [1, 240] : tensor<1x240xf32>
    %376 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%374, %cst_210 : tensor<1x240xf32>, tensor<1x240xf32>) outs(%375 : tensor<1x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x240xf32>
    %377 = tensor.expand_shape %376 [[0], [1, 2, 3]] : tensor<1x240xf32> into tensor<1x1x1x240xf32>
    %378 = linalg.init_tensor [1, 1, 1, 64] : tensor<1x1x1x64xf32>
    %cst_304 = arith.constant 0.000000e+00 : f32
    %379 = linalg.fill ins(%cst_304 : f32) outs(%378 : tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xf32>
    %380 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%377, %cst_245 : tensor<1x1x1x240xf32>, tensor<1x1x240x64xf32>) outs(%379 : tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xf32>
    %381 = linalg.init_tensor [1, 1, 1, 64] : tensor<1x1x1x64xf32>
    %382 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%380, %cst_79 : tensor<1x1x1x64xf32>, tensor<1x1x1x64xf32>) outs(%381 : tensor<1x1x1x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x64xf32>
    %383 = linalg.init_tensor [1, 1, 1, 64] : tensor<1x1x1x64xf32>
    %384 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%382, %cst_211 : tensor<1x1x1x64xf32>, tensor<1x1x1x64xf32>) outs(%383 : tensor<1x1x1x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x64xf32>
    %385 = linalg.init_tensor [1, 1, 1, 240] : tensor<1x1x1x240xf32>
    %cst_305 = arith.constant 0.000000e+00 : f32
    %386 = linalg.fill ins(%cst_305 : f32) outs(%385 : tensor<1x1x1x240xf32>) -> tensor<1x1x1x240xf32>
    %387 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%384, %cst_244 : tensor<1x1x1x64xf32>, tensor<1x1x64x240xf32>) outs(%386 : tensor<1x1x1x240xf32>) -> tensor<1x1x1x240xf32>
    %388 = linalg.init_tensor [1, 1, 1, 240] : tensor<1x1x1x240xf32>
    %389 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%387, %cst_80 : tensor<1x1x1x240xf32>, tensor<1x1x1x240xf32>) outs(%388 : tensor<1x1x1x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x240xf32>
    %390 = linalg.init_tensor [1, 1, 1, 240] : tensor<1x1x1x240xf32>
    %391 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%389, %cst_174 : tensor<1x1x1x240xf32>, tensor<1x1x1x240xf32>) outs(%390 : tensor<1x1x1x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x240xf32>
    %392 = linalg.init_tensor [1, 1, 1, 240] : tensor<1x1x1x240xf32>
    %393 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %391, %cst_220 : tensor<f32>, tensor<1x1x1x240xf32>, tensor<f32>) outs(%392 : tensor<1x1x1x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      %885 = arith.minf %884, %arg3 : f32
      linalg.yield %885 : f32
    } -> tensor<1x1x1x240xf32>
    %394 = linalg.init_tensor [1, 1, 1, 240] : tensor<1x1x1x240xf32>
    %395 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%393, %cst_191 : tensor<1x1x1x240xf32>, tensor<1x1x1x240xf32>) outs(%394 : tensor<1x1x1x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x240xf32>
    %396 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %397 = linalg.generic {indexing_maps = [#map5, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%395 : tensor<1x1x1x240xf32>) outs(%396 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<1x14x14x240xf32>
    %398 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %399 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%371, %397 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%398 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x240xf32>
    %400 = linalg.init_tensor [1, 14, 14, 40] : tensor<1x14x14x40xf32>
    %cst_306 = arith.constant 0.000000e+00 : f32
    %401 = linalg.fill ins(%cst_306 : f32) outs(%400 : tensor<1x14x14x40xf32>) -> tensor<1x14x14x40xf32>
    %402 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%399, %cst_243 : tensor<1x14x14x240xf32>, tensor<1x1x240x40xf32>) outs(%401 : tensor<1x14x14x40xf32>) -> tensor<1x14x14x40xf32>
    %403 = linalg.init_tensor [1, 14, 14, 40] : tensor<1x14x14x40xf32>
    %404 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%402, %cst_84 : tensor<1x14x14x40xf32>, tensor<1x14x14x40xf32>) outs(%403 : tensor<1x14x14x40xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.subf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x40xf32>
    %405 = linalg.init_tensor [1, 14, 14, 40] : tensor<1x14x14x40xf32>
    %406 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%404, %cst_82 : tensor<1x14x14x40xf32>, tensor<1x14x14x40xf32>) outs(%405 : tensor<1x14x14x40xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x40xf32>
    %407 = linalg.init_tensor [1, 14, 14, 40] : tensor<1x14x14x40xf32>
    %408 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%406, %cst_81 : tensor<1x14x14x40xf32>, tensor<1x14x14x40xf32>) outs(%407 : tensor<1x14x14x40xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x40xf32>
    %409 = linalg.init_tensor [1, 14, 14, 40] : tensor<1x14x14x40xf32>
    %410 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%408, %cst_83 : tensor<1x14x14x40xf32>, tensor<1x14x14x40xf32>) outs(%409 : tensor<1x14x14x40xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x40xf32>
    %411 = linalg.init_tensor [1, 14, 14, 40] : tensor<1x14x14x40xf32>
    %412 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%329, %410 : tensor<1x14x14x40xf32>, tensor<1x14x14x40xf32>) outs(%411 : tensor<1x14x14x40xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x40xf32>
    %413 = linalg.init_tensor [1, 14, 14, 120] : tensor<1x14x14x120xf32>
    %cst_307 = arith.constant 0.000000e+00 : f32
    %414 = linalg.fill ins(%cst_307 : f32) outs(%413 : tensor<1x14x14x120xf32>) -> tensor<1x14x14x120xf32>
    %415 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%412, %cst_246 : tensor<1x14x14x40xf32>, tensor<1x1x40x120xf32>) outs(%414 : tensor<1x14x14x120xf32>) -> tensor<1x14x14x120xf32>
    %416 = linalg.init_tensor [1, 14, 14, 120] : tensor<1x14x14x120xf32>
    %417 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%415, %cst_88 : tensor<1x14x14x120xf32>, tensor<1x14x14x120xf32>) outs(%416 : tensor<1x14x14x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.subf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x120xf32>
    %418 = linalg.init_tensor [1, 14, 14, 120] : tensor<1x14x14x120xf32>
    %419 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%417, %cst_86 : tensor<1x14x14x120xf32>, tensor<1x14x14x120xf32>) outs(%418 : tensor<1x14x14x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x120xf32>
    %420 = linalg.init_tensor [1, 14, 14, 120] : tensor<1x14x14x120xf32>
    %421 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%419, %cst_85 : tensor<1x14x14x120xf32>, tensor<1x14x14x120xf32>) outs(%420 : tensor<1x14x14x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x120xf32>
    %422 = linalg.init_tensor [1, 14, 14, 120] : tensor<1x14x14x120xf32>
    %423 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%421, %cst_87 : tensor<1x14x14x120xf32>, tensor<1x14x14x120xf32>) outs(%422 : tensor<1x14x14x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x120xf32>
    %424 = linalg.init_tensor [1, 14, 14, 120] : tensor<1x14x14x120xf32>
    %425 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%423, %cst_175 : tensor<1x14x14x120xf32>, tensor<1x14x14x120xf32>) outs(%424 : tensor<1x14x14x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x120xf32>
    %426 = linalg.init_tensor [1, 14, 14, 120] : tensor<1x14x14x120xf32>
    %427 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %425, %cst_220 : tensor<f32>, tensor<1x14x14x120xf32>, tensor<f32>) outs(%426 : tensor<1x14x14x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      %885 = arith.minf %884, %arg3 : f32
      linalg.yield %885 : f32
    } -> tensor<1x14x14x120xf32>
    %428 = linalg.init_tensor [1, 14, 14, 120] : tensor<1x14x14x120xf32>
    %429 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%427, %cst_192 : tensor<1x14x14x120xf32>, tensor<1x14x14x120xf32>) outs(%428 : tensor<1x14x14x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x120xf32>
    %430 = linalg.init_tensor [1, 14, 14, 120] : tensor<1x14x14x120xf32>
    %431 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%429, %423 : tensor<1x14x14x120xf32>, tensor<1x14x14x120xf32>) outs(%430 : tensor<1x14x14x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x120xf32>
    %cst_308 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_309 = arith.constant 0.000000e+00 : f32
    %432 = linalg.init_tensor [1, 18, 18, 120] : tensor<1x18x18x120xf32>
    %433 = linalg.fill ins(%cst_309 : f32) outs(%432 : tensor<1x18x18x120xf32>) -> tensor<1x18x18x120xf32>
    %434 = tensor.insert_slice %431 into %433[0, 2, 2, 0] [1, 14, 14, 120] [1, 1, 1, 1] : tensor<1x14x14x120xf32> into tensor<1x18x18x120xf32>
    %435 = linalg.init_tensor [1, 14, 14, 120] : tensor<1x14x14x120xf32>
    %cst_310 = arith.constant 0.000000e+00 : f32
    %436 = linalg.fill ins(%cst_310 : f32) outs(%435 : tensor<1x14x14x120xf32>) -> tensor<1x14x14x120xf32>
    %437 = tensor.collapse_shape %cst_89 [[0], [1], [2, 3]] : tensor<5x5x1x120xf32> into tensor<5x5x120xf32>
    %438 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%434, %437 : tensor<1x18x18x120xf32>, tensor<5x5x120xf32>) outs(%436 : tensor<1x14x14x120xf32>) -> tensor<1x14x14x120xf32>
    %439 = linalg.init_tensor [1, 14, 14, 120] : tensor<1x14x14x120xf32>
    %440 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%438, %cst_93 : tensor<1x14x14x120xf32>, tensor<1x14x14x120xf32>) outs(%439 : tensor<1x14x14x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.subf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x120xf32>
    %441 = linalg.init_tensor [1, 14, 14, 120] : tensor<1x14x14x120xf32>
    %442 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%440, %cst_91 : tensor<1x14x14x120xf32>, tensor<1x14x14x120xf32>) outs(%441 : tensor<1x14x14x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x120xf32>
    %443 = linalg.init_tensor [1, 14, 14, 120] : tensor<1x14x14x120xf32>
    %444 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%442, %cst_90 : tensor<1x14x14x120xf32>, tensor<1x14x14x120xf32>) outs(%443 : tensor<1x14x14x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x120xf32>
    %445 = linalg.init_tensor [1, 14, 14, 120] : tensor<1x14x14x120xf32>
    %446 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%444, %cst_92 : tensor<1x14x14x120xf32>, tensor<1x14x14x120xf32>) outs(%445 : tensor<1x14x14x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x120xf32>
    %447 = linalg.init_tensor [1, 14, 14, 120] : tensor<1x14x14x120xf32>
    %448 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%446, %cst_175 : tensor<1x14x14x120xf32>, tensor<1x14x14x120xf32>) outs(%447 : tensor<1x14x14x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x120xf32>
    %449 = linalg.init_tensor [1, 14, 14, 120] : tensor<1x14x14x120xf32>
    %450 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %448, %cst_220 : tensor<f32>, tensor<1x14x14x120xf32>, tensor<f32>) outs(%449 : tensor<1x14x14x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      %885 = arith.minf %884, %arg3 : f32
      linalg.yield %885 : f32
    } -> tensor<1x14x14x120xf32>
    %451 = linalg.init_tensor [1, 14, 14, 120] : tensor<1x14x14x120xf32>
    %452 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%450, %cst_192 : tensor<1x14x14x120xf32>, tensor<1x14x14x120xf32>) outs(%451 : tensor<1x14x14x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x120xf32>
    %453 = linalg.init_tensor [1, 14, 14, 120] : tensor<1x14x14x120xf32>
    %454 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%452, %446 : tensor<1x14x14x120xf32>, tensor<1x14x14x120xf32>) outs(%453 : tensor<1x14x14x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x120xf32>
    %cst_311 = arith.constant 0.000000e+00 : f32
    %455 = linalg.init_tensor [1, 120] : tensor<1x120xf32>
    %456 = linalg.fill ins(%cst_311 : f32) outs(%455 : tensor<1x120xf32>) -> tensor<1x120xf32>
    %457 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%454 : tensor<1x14x14x120xf32>) outs(%456 : tensor<1x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x120xf32>
    %458 = linalg.init_tensor [1, 120] : tensor<1x120xf32>
    %459 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%457, %cst_212 : tensor<1x120xf32>, tensor<1x120xf32>) outs(%458 : tensor<1x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x120xf32>
    %460 = tensor.expand_shape %459 [[0], [1, 2, 3]] : tensor<1x120xf32> into tensor<1x1x1x120xf32>
    %461 = linalg.init_tensor [1, 1, 1, 32] : tensor<1x1x1x32xf32>
    %cst_312 = arith.constant 0.000000e+00 : f32
    %462 = linalg.fill ins(%cst_312 : f32) outs(%461 : tensor<1x1x1x32xf32>) -> tensor<1x1x1x32xf32>
    %463 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%460, %cst_249 : tensor<1x1x1x120xf32>, tensor<1x1x120x32xf32>) outs(%462 : tensor<1x1x1x32xf32>) -> tensor<1x1x1x32xf32>
    %464 = linalg.init_tensor [1, 1, 1, 32] : tensor<1x1x1x32xf32>
    %465 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%463, %cst_94 : tensor<1x1x1x32xf32>, tensor<1x1x1x32xf32>) outs(%464 : tensor<1x1x1x32xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x32xf32>
    %466 = linalg.init_tensor [1, 1, 1, 32] : tensor<1x1x1x32xf32>
    %467 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%465, %cst_213 : tensor<1x1x1x32xf32>, tensor<1x1x1x32xf32>) outs(%466 : tensor<1x1x1x32xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x32xf32>
    %468 = linalg.init_tensor [1, 1, 1, 120] : tensor<1x1x1x120xf32>
    %cst_313 = arith.constant 0.000000e+00 : f32
    %469 = linalg.fill ins(%cst_313 : f32) outs(%468 : tensor<1x1x1x120xf32>) -> tensor<1x1x1x120xf32>
    %470 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%467, %cst_248 : tensor<1x1x1x32xf32>, tensor<1x1x32x120xf32>) outs(%469 : tensor<1x1x1x120xf32>) -> tensor<1x1x1x120xf32>
    %471 = linalg.init_tensor [1, 1, 1, 120] : tensor<1x1x1x120xf32>
    %472 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%470, %cst_95 : tensor<1x1x1x120xf32>, tensor<1x1x1x120xf32>) outs(%471 : tensor<1x1x1x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x120xf32>
    %473 = linalg.init_tensor [1, 1, 1, 120] : tensor<1x1x1x120xf32>
    %474 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%472, %cst_176 : tensor<1x1x1x120xf32>, tensor<1x1x1x120xf32>) outs(%473 : tensor<1x1x1x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x120xf32>
    %475 = linalg.init_tensor [1, 1, 1, 120] : tensor<1x1x1x120xf32>
    %476 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %474, %cst_220 : tensor<f32>, tensor<1x1x1x120xf32>, tensor<f32>) outs(%475 : tensor<1x1x1x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      %885 = arith.minf %884, %arg3 : f32
      linalg.yield %885 : f32
    } -> tensor<1x1x1x120xf32>
    %477 = linalg.init_tensor [1, 1, 1, 120] : tensor<1x1x1x120xf32>
    %478 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%476, %cst_193 : tensor<1x1x1x120xf32>, tensor<1x1x1x120xf32>) outs(%477 : tensor<1x1x1x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x120xf32>
    %479 = linalg.init_tensor [1, 14, 14, 120] : tensor<1x14x14x120xf32>
    %480 = linalg.generic {indexing_maps = [#map5, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%478 : tensor<1x1x1x120xf32>) outs(%479 : tensor<1x14x14x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<1x14x14x120xf32>
    %481 = linalg.init_tensor [1, 14, 14, 120] : tensor<1x14x14x120xf32>
    %482 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%454, %480 : tensor<1x14x14x120xf32>, tensor<1x14x14x120xf32>) outs(%481 : tensor<1x14x14x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x120xf32>
    %483 = linalg.init_tensor [1, 14, 14, 48] : tensor<1x14x14x48xf32>
    %cst_314 = arith.constant 0.000000e+00 : f32
    %484 = linalg.fill ins(%cst_314 : f32) outs(%483 : tensor<1x14x14x48xf32>) -> tensor<1x14x14x48xf32>
    %485 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%482, %cst_247 : tensor<1x14x14x120xf32>, tensor<1x1x120x48xf32>) outs(%484 : tensor<1x14x14x48xf32>) -> tensor<1x14x14x48xf32>
    %486 = linalg.init_tensor [1, 14, 14, 48] : tensor<1x14x14x48xf32>
    %487 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%485, %cst_99 : tensor<1x14x14x48xf32>, tensor<1x14x14x48xf32>) outs(%486 : tensor<1x14x14x48xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.subf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x48xf32>
    %488 = linalg.init_tensor [1, 14, 14, 48] : tensor<1x14x14x48xf32>
    %489 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%487, %cst_97 : tensor<1x14x14x48xf32>, tensor<1x14x14x48xf32>) outs(%488 : tensor<1x14x14x48xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x48xf32>
    %490 = linalg.init_tensor [1, 14, 14, 48] : tensor<1x14x14x48xf32>
    %491 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%489, %cst_96 : tensor<1x14x14x48xf32>, tensor<1x14x14x48xf32>) outs(%490 : tensor<1x14x14x48xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x48xf32>
    %492 = linalg.init_tensor [1, 14, 14, 48] : tensor<1x14x14x48xf32>
    %493 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%491, %cst_98 : tensor<1x14x14x48xf32>, tensor<1x14x14x48xf32>) outs(%492 : tensor<1x14x14x48xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x48xf32>
    %494 = linalg.init_tensor [1, 14, 14, 144] : tensor<1x14x14x144xf32>
    %cst_315 = arith.constant 0.000000e+00 : f32
    %495 = linalg.fill ins(%cst_315 : f32) outs(%494 : tensor<1x14x14x144xf32>) -> tensor<1x14x14x144xf32>
    %496 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%493, %cst_250 : tensor<1x14x14x48xf32>, tensor<1x1x48x144xf32>) outs(%495 : tensor<1x14x14x144xf32>) -> tensor<1x14x14x144xf32>
    %497 = linalg.init_tensor [1, 14, 14, 144] : tensor<1x14x14x144xf32>
    %498 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%496, %cst_103 : tensor<1x14x14x144xf32>, tensor<1x14x14x144xf32>) outs(%497 : tensor<1x14x14x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.subf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x144xf32>
    %499 = linalg.init_tensor [1, 14, 14, 144] : tensor<1x14x14x144xf32>
    %500 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%498, %cst_101 : tensor<1x14x14x144xf32>, tensor<1x14x14x144xf32>) outs(%499 : tensor<1x14x14x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x144xf32>
    %501 = linalg.init_tensor [1, 14, 14, 144] : tensor<1x14x14x144xf32>
    %502 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%500, %cst_100 : tensor<1x14x14x144xf32>, tensor<1x14x14x144xf32>) outs(%501 : tensor<1x14x14x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x144xf32>
    %503 = linalg.init_tensor [1, 14, 14, 144] : tensor<1x14x14x144xf32>
    %504 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%502, %cst_102 : tensor<1x14x14x144xf32>, tensor<1x14x14x144xf32>) outs(%503 : tensor<1x14x14x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x144xf32>
    %505 = linalg.init_tensor [1, 14, 14, 144] : tensor<1x14x14x144xf32>
    %506 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%504, %cst_177 : tensor<1x14x14x144xf32>, tensor<1x14x14x144xf32>) outs(%505 : tensor<1x14x14x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x144xf32>
    %507 = linalg.init_tensor [1, 14, 14, 144] : tensor<1x14x14x144xf32>
    %508 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %506, %cst_220 : tensor<f32>, tensor<1x14x14x144xf32>, tensor<f32>) outs(%507 : tensor<1x14x14x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      %885 = arith.minf %884, %arg3 : f32
      linalg.yield %885 : f32
    } -> tensor<1x14x14x144xf32>
    %509 = linalg.init_tensor [1, 14, 14, 144] : tensor<1x14x14x144xf32>
    %510 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%508, %cst_194 : tensor<1x14x14x144xf32>, tensor<1x14x14x144xf32>) outs(%509 : tensor<1x14x14x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x144xf32>
    %511 = linalg.init_tensor [1, 14, 14, 144] : tensor<1x14x14x144xf32>
    %512 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%510, %504 : tensor<1x14x14x144xf32>, tensor<1x14x14x144xf32>) outs(%511 : tensor<1x14x14x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x144xf32>
    %cst_316 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_317 = arith.constant 0.000000e+00 : f32
    %513 = linalg.init_tensor [1, 18, 18, 144] : tensor<1x18x18x144xf32>
    %514 = linalg.fill ins(%cst_317 : f32) outs(%513 : tensor<1x18x18x144xf32>) -> tensor<1x18x18x144xf32>
    %515 = tensor.insert_slice %512 into %514[0, 2, 2, 0] [1, 14, 14, 144] [1, 1, 1, 1] : tensor<1x14x14x144xf32> into tensor<1x18x18x144xf32>
    %516 = linalg.init_tensor [1, 14, 14, 144] : tensor<1x14x14x144xf32>
    %cst_318 = arith.constant 0.000000e+00 : f32
    %517 = linalg.fill ins(%cst_318 : f32) outs(%516 : tensor<1x14x14x144xf32>) -> tensor<1x14x14x144xf32>
    %518 = tensor.collapse_shape %cst_104 [[0], [1], [2, 3]] : tensor<5x5x1x144xf32> into tensor<5x5x144xf32>
    %519 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%515, %518 : tensor<1x18x18x144xf32>, tensor<5x5x144xf32>) outs(%517 : tensor<1x14x14x144xf32>) -> tensor<1x14x14x144xf32>
    %520 = linalg.init_tensor [1, 14, 14, 144] : tensor<1x14x14x144xf32>
    %521 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%519, %cst_108 : tensor<1x14x14x144xf32>, tensor<1x14x14x144xf32>) outs(%520 : tensor<1x14x14x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.subf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x144xf32>
    %522 = linalg.init_tensor [1, 14, 14, 144] : tensor<1x14x14x144xf32>
    %523 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%521, %cst_106 : tensor<1x14x14x144xf32>, tensor<1x14x14x144xf32>) outs(%522 : tensor<1x14x14x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x144xf32>
    %524 = linalg.init_tensor [1, 14, 14, 144] : tensor<1x14x14x144xf32>
    %525 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%523, %cst_105 : tensor<1x14x14x144xf32>, tensor<1x14x14x144xf32>) outs(%524 : tensor<1x14x14x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x144xf32>
    %526 = linalg.init_tensor [1, 14, 14, 144] : tensor<1x14x14x144xf32>
    %527 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%525, %cst_107 : tensor<1x14x14x144xf32>, tensor<1x14x14x144xf32>) outs(%526 : tensor<1x14x14x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x144xf32>
    %528 = linalg.init_tensor [1, 14, 14, 144] : tensor<1x14x14x144xf32>
    %529 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%527, %cst_177 : tensor<1x14x14x144xf32>, tensor<1x14x14x144xf32>) outs(%528 : tensor<1x14x14x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x144xf32>
    %530 = linalg.init_tensor [1, 14, 14, 144] : tensor<1x14x14x144xf32>
    %531 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %529, %cst_220 : tensor<f32>, tensor<1x14x14x144xf32>, tensor<f32>) outs(%530 : tensor<1x14x14x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      %885 = arith.minf %884, %arg3 : f32
      linalg.yield %885 : f32
    } -> tensor<1x14x14x144xf32>
    %532 = linalg.init_tensor [1, 14, 14, 144] : tensor<1x14x14x144xf32>
    %533 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%531, %cst_194 : tensor<1x14x14x144xf32>, tensor<1x14x14x144xf32>) outs(%532 : tensor<1x14x14x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x144xf32>
    %534 = linalg.init_tensor [1, 14, 14, 144] : tensor<1x14x14x144xf32>
    %535 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%533, %527 : tensor<1x14x14x144xf32>, tensor<1x14x14x144xf32>) outs(%534 : tensor<1x14x14x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x144xf32>
    %cst_319 = arith.constant 0.000000e+00 : f32
    %536 = linalg.init_tensor [1, 144] : tensor<1x144xf32>
    %537 = linalg.fill ins(%cst_319 : f32) outs(%536 : tensor<1x144xf32>) -> tensor<1x144xf32>
    %538 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%535 : tensor<1x14x14x144xf32>) outs(%537 : tensor<1x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x144xf32>
    %539 = linalg.init_tensor [1, 144] : tensor<1x144xf32>
    %540 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%538, %cst_214 : tensor<1x144xf32>, tensor<1x144xf32>) outs(%539 : tensor<1x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x144xf32>
    %541 = tensor.expand_shape %540 [[0], [1, 2, 3]] : tensor<1x144xf32> into tensor<1x1x1x144xf32>
    %542 = linalg.init_tensor [1, 1, 1, 40] : tensor<1x1x1x40xf32>
    %cst_320 = arith.constant 0.000000e+00 : f32
    %543 = linalg.fill ins(%cst_320 : f32) outs(%542 : tensor<1x1x1x40xf32>) -> tensor<1x1x1x40xf32>
    %544 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%541, %cst_253 : tensor<1x1x1x144xf32>, tensor<1x1x144x40xf32>) outs(%543 : tensor<1x1x1x40xf32>) -> tensor<1x1x1x40xf32>
    %545 = linalg.init_tensor [1, 1, 1, 40] : tensor<1x1x1x40xf32>
    %546 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%544, %cst_109 : tensor<1x1x1x40xf32>, tensor<1x1x1x40xf32>) outs(%545 : tensor<1x1x1x40xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x40xf32>
    %547 = linalg.init_tensor [1, 1, 1, 40] : tensor<1x1x1x40xf32>
    %548 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%546, %cst_215 : tensor<1x1x1x40xf32>, tensor<1x1x1x40xf32>) outs(%547 : tensor<1x1x1x40xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x40xf32>
    %549 = linalg.init_tensor [1, 1, 1, 144] : tensor<1x1x1x144xf32>
    %cst_321 = arith.constant 0.000000e+00 : f32
    %550 = linalg.fill ins(%cst_321 : f32) outs(%549 : tensor<1x1x1x144xf32>) -> tensor<1x1x1x144xf32>
    %551 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%548, %cst_252 : tensor<1x1x1x40xf32>, tensor<1x1x40x144xf32>) outs(%550 : tensor<1x1x1x144xf32>) -> tensor<1x1x1x144xf32>
    %552 = linalg.init_tensor [1, 1, 1, 144] : tensor<1x1x1x144xf32>
    %553 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%551, %cst_110 : tensor<1x1x1x144xf32>, tensor<1x1x1x144xf32>) outs(%552 : tensor<1x1x1x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x144xf32>
    %554 = linalg.init_tensor [1, 1, 1, 144] : tensor<1x1x1x144xf32>
    %555 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%553, %cst_178 : tensor<1x1x1x144xf32>, tensor<1x1x1x144xf32>) outs(%554 : tensor<1x1x1x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x144xf32>
    %556 = linalg.init_tensor [1, 1, 1, 144] : tensor<1x1x1x144xf32>
    %557 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %555, %cst_220 : tensor<f32>, tensor<1x1x1x144xf32>, tensor<f32>) outs(%556 : tensor<1x1x1x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      %885 = arith.minf %884, %arg3 : f32
      linalg.yield %885 : f32
    } -> tensor<1x1x1x144xf32>
    %558 = linalg.init_tensor [1, 1, 1, 144] : tensor<1x1x1x144xf32>
    %559 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%557, %cst_195 : tensor<1x1x1x144xf32>, tensor<1x1x1x144xf32>) outs(%558 : tensor<1x1x1x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x144xf32>
    %560 = linalg.init_tensor [1, 14, 14, 144] : tensor<1x14x14x144xf32>
    %561 = linalg.generic {indexing_maps = [#map5, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%559 : tensor<1x1x1x144xf32>) outs(%560 : tensor<1x14x14x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<1x14x14x144xf32>
    %562 = linalg.init_tensor [1, 14, 14, 144] : tensor<1x14x14x144xf32>
    %563 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%535, %561 : tensor<1x14x14x144xf32>, tensor<1x14x14x144xf32>) outs(%562 : tensor<1x14x14x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x144xf32>
    %564 = linalg.init_tensor [1, 14, 14, 48] : tensor<1x14x14x48xf32>
    %cst_322 = arith.constant 0.000000e+00 : f32
    %565 = linalg.fill ins(%cst_322 : f32) outs(%564 : tensor<1x14x14x48xf32>) -> tensor<1x14x14x48xf32>
    %566 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%563, %cst_251 : tensor<1x14x14x144xf32>, tensor<1x1x144x48xf32>) outs(%565 : tensor<1x14x14x48xf32>) -> tensor<1x14x14x48xf32>
    %567 = linalg.init_tensor [1, 14, 14, 48] : tensor<1x14x14x48xf32>
    %568 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%566, %cst_114 : tensor<1x14x14x48xf32>, tensor<1x14x14x48xf32>) outs(%567 : tensor<1x14x14x48xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.subf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x48xf32>
    %569 = linalg.init_tensor [1, 14, 14, 48] : tensor<1x14x14x48xf32>
    %570 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%568, %cst_112 : tensor<1x14x14x48xf32>, tensor<1x14x14x48xf32>) outs(%569 : tensor<1x14x14x48xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x48xf32>
    %571 = linalg.init_tensor [1, 14, 14, 48] : tensor<1x14x14x48xf32>
    %572 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%570, %cst_111 : tensor<1x14x14x48xf32>, tensor<1x14x14x48xf32>) outs(%571 : tensor<1x14x14x48xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x48xf32>
    %573 = linalg.init_tensor [1, 14, 14, 48] : tensor<1x14x14x48xf32>
    %574 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%572, %cst_113 : tensor<1x14x14x48xf32>, tensor<1x14x14x48xf32>) outs(%573 : tensor<1x14x14x48xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x48xf32>
    %575 = linalg.init_tensor [1, 14, 14, 48] : tensor<1x14x14x48xf32>
    %576 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%493, %574 : tensor<1x14x14x48xf32>, tensor<1x14x14x48xf32>) outs(%575 : tensor<1x14x14x48xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x48xf32>
    %577 = linalg.init_tensor [1, 14, 14, 288] : tensor<1x14x14x288xf32>
    %cst_323 = arith.constant 0.000000e+00 : f32
    %578 = linalg.fill ins(%cst_323 : f32) outs(%577 : tensor<1x14x14x288xf32>) -> tensor<1x14x14x288xf32>
    %579 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%576, %cst_254 : tensor<1x14x14x48xf32>, tensor<1x1x48x288xf32>) outs(%578 : tensor<1x14x14x288xf32>) -> tensor<1x14x14x288xf32>
    %580 = linalg.init_tensor [1, 14, 14, 288] : tensor<1x14x14x288xf32>
    %581 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%579, %cst_118 : tensor<1x14x14x288xf32>, tensor<1x14x14x288xf32>) outs(%580 : tensor<1x14x14x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.subf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x288xf32>
    %582 = linalg.init_tensor [1, 14, 14, 288] : tensor<1x14x14x288xf32>
    %583 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%581, %cst_116 : tensor<1x14x14x288xf32>, tensor<1x14x14x288xf32>) outs(%582 : tensor<1x14x14x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x288xf32>
    %584 = linalg.init_tensor [1, 14, 14, 288] : tensor<1x14x14x288xf32>
    %585 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%583, %cst_115 : tensor<1x14x14x288xf32>, tensor<1x14x14x288xf32>) outs(%584 : tensor<1x14x14x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x288xf32>
    %586 = linalg.init_tensor [1, 14, 14, 288] : tensor<1x14x14x288xf32>
    %587 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%585, %cst_117 : tensor<1x14x14x288xf32>, tensor<1x14x14x288xf32>) outs(%586 : tensor<1x14x14x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x288xf32>
    %588 = linalg.init_tensor [1, 14, 14, 288] : tensor<1x14x14x288xf32>
    %589 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%587, %cst_179 : tensor<1x14x14x288xf32>, tensor<1x14x14x288xf32>) outs(%588 : tensor<1x14x14x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x288xf32>
    %590 = linalg.init_tensor [1, 14, 14, 288] : tensor<1x14x14x288xf32>
    %591 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %589, %cst_220 : tensor<f32>, tensor<1x14x14x288xf32>, tensor<f32>) outs(%590 : tensor<1x14x14x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      %885 = arith.minf %884, %arg3 : f32
      linalg.yield %885 : f32
    } -> tensor<1x14x14x288xf32>
    %592 = linalg.init_tensor [1, 14, 14, 288] : tensor<1x14x14x288xf32>
    %593 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%591, %cst_196 : tensor<1x14x14x288xf32>, tensor<1x14x14x288xf32>) outs(%592 : tensor<1x14x14x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x288xf32>
    %594 = linalg.init_tensor [1, 14, 14, 288] : tensor<1x14x14x288xf32>
    %595 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%593, %587 : tensor<1x14x14x288xf32>, tensor<1x14x14x288xf32>) outs(%594 : tensor<1x14x14x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x14x14x288xf32>
    %cst_324 = arith.constant 0.000000e+00 : f32
    %596 = linalg.init_tensor [1, 17, 17, 288] : tensor<1x17x17x288xf32>
    %597 = linalg.fill ins(%cst_324 : f32) outs(%596 : tensor<1x17x17x288xf32>) -> tensor<1x17x17x288xf32>
    %598 = tensor.insert_slice %595 into %597[0, 1, 1, 0] [1, 14, 14, 288] [1, 1, 1, 1] : tensor<1x14x14x288xf32> into tensor<1x17x17x288xf32>
    %599 = linalg.init_tensor [1, 7, 7, 288] : tensor<1x7x7x288xf32>
    %cst_325 = arith.constant 0.000000e+00 : f32
    %600 = linalg.fill ins(%cst_325 : f32) outs(%599 : tensor<1x7x7x288xf32>) -> tensor<1x7x7x288xf32>
    %601 = tensor.collapse_shape %cst_119 [[0], [1], [2, 3]] : tensor<5x5x1x288xf32> into tensor<5x5x288xf32>
    %602 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%598, %601 : tensor<1x17x17x288xf32>, tensor<5x5x288xf32>) outs(%600 : tensor<1x7x7x288xf32>) -> tensor<1x7x7x288xf32>
    %603 = linalg.init_tensor [1, 7, 7, 288] : tensor<1x7x7x288xf32>
    %604 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%602, %cst_123 : tensor<1x7x7x288xf32>, tensor<1x7x7x288xf32>) outs(%603 : tensor<1x7x7x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.subf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x288xf32>
    %605 = linalg.init_tensor [1, 7, 7, 288] : tensor<1x7x7x288xf32>
    %606 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%604, %cst_121 : tensor<1x7x7x288xf32>, tensor<1x7x7x288xf32>) outs(%605 : tensor<1x7x7x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x288xf32>
    %607 = linalg.init_tensor [1, 7, 7, 288] : tensor<1x7x7x288xf32>
    %608 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%606, %cst_120 : tensor<1x7x7x288xf32>, tensor<1x7x7x288xf32>) outs(%607 : tensor<1x7x7x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x288xf32>
    %609 = linalg.init_tensor [1, 7, 7, 288] : tensor<1x7x7x288xf32>
    %610 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%608, %cst_122 : tensor<1x7x7x288xf32>, tensor<1x7x7x288xf32>) outs(%609 : tensor<1x7x7x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x288xf32>
    %611 = linalg.init_tensor [1, 7, 7, 288] : tensor<1x7x7x288xf32>
    %612 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%610, %cst_180 : tensor<1x7x7x288xf32>, tensor<1x7x7x288xf32>) outs(%611 : tensor<1x7x7x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x288xf32>
    %613 = linalg.init_tensor [1, 7, 7, 288] : tensor<1x7x7x288xf32>
    %614 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %612, %cst_220 : tensor<f32>, tensor<1x7x7x288xf32>, tensor<f32>) outs(%613 : tensor<1x7x7x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      %885 = arith.minf %884, %arg3 : f32
      linalg.yield %885 : f32
    } -> tensor<1x7x7x288xf32>
    %615 = linalg.init_tensor [1, 7, 7, 288] : tensor<1x7x7x288xf32>
    %616 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%614, %cst_197 : tensor<1x7x7x288xf32>, tensor<1x7x7x288xf32>) outs(%615 : tensor<1x7x7x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x288xf32>
    %617 = linalg.init_tensor [1, 7, 7, 288] : tensor<1x7x7x288xf32>
    %618 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%616, %610 : tensor<1x7x7x288xf32>, tensor<1x7x7x288xf32>) outs(%617 : tensor<1x7x7x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x288xf32>
    %cst_326 = arith.constant 0.000000e+00 : f32
    %619 = linalg.init_tensor [1, 288] : tensor<1x288xf32>
    %620 = linalg.fill ins(%cst_326 : f32) outs(%619 : tensor<1x288xf32>) -> tensor<1x288xf32>
    %621 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%618 : tensor<1x7x7x288xf32>) outs(%620 : tensor<1x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x288xf32>
    %622 = linalg.init_tensor [1, 288] : tensor<1x288xf32>
    %623 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%621, %cst_216 : tensor<1x288xf32>, tensor<1x288xf32>) outs(%622 : tensor<1x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x288xf32>
    %624 = tensor.expand_shape %623 [[0], [1, 2, 3]] : tensor<1x288xf32> into tensor<1x1x1x288xf32>
    %625 = linalg.init_tensor [1, 1, 1, 72] : tensor<1x1x1x72xf32>
    %cst_327 = arith.constant 0.000000e+00 : f32
    %626 = linalg.fill ins(%cst_327 : f32) outs(%625 : tensor<1x1x1x72xf32>) -> tensor<1x1x1x72xf32>
    %627 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%624, %cst_257 : tensor<1x1x1x288xf32>, tensor<1x1x288x72xf32>) outs(%626 : tensor<1x1x1x72xf32>) -> tensor<1x1x1x72xf32>
    %628 = linalg.init_tensor [1, 1, 1, 72] : tensor<1x1x1x72xf32>
    %629 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%627, %cst_124 : tensor<1x1x1x72xf32>, tensor<1x1x1x72xf32>) outs(%628 : tensor<1x1x1x72xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x72xf32>
    %630 = linalg.init_tensor [1, 1, 1, 72] : tensor<1x1x1x72xf32>
    %631 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%629, %cst_217 : tensor<1x1x1x72xf32>, tensor<1x1x1x72xf32>) outs(%630 : tensor<1x1x1x72xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x72xf32>
    %632 = linalg.init_tensor [1, 1, 1, 288] : tensor<1x1x1x288xf32>
    %cst_328 = arith.constant 0.000000e+00 : f32
    %633 = linalg.fill ins(%cst_328 : f32) outs(%632 : tensor<1x1x1x288xf32>) -> tensor<1x1x1x288xf32>
    %634 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%631, %cst_256 : tensor<1x1x1x72xf32>, tensor<1x1x72x288xf32>) outs(%633 : tensor<1x1x1x288xf32>) -> tensor<1x1x1x288xf32>
    %635 = linalg.init_tensor [1, 1, 1, 288] : tensor<1x1x1x288xf32>
    %636 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%634, %cst_125 : tensor<1x1x1x288xf32>, tensor<1x1x1x288xf32>) outs(%635 : tensor<1x1x1x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x288xf32>
    %637 = linalg.init_tensor [1, 1, 1, 288] : tensor<1x1x1x288xf32>
    %638 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%636, %cst_181 : tensor<1x1x1x288xf32>, tensor<1x1x1x288xf32>) outs(%637 : tensor<1x1x1x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x288xf32>
    %639 = linalg.init_tensor [1, 1, 1, 288] : tensor<1x1x1x288xf32>
    %640 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %638, %cst_220 : tensor<f32>, tensor<1x1x1x288xf32>, tensor<f32>) outs(%639 : tensor<1x1x1x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      %885 = arith.minf %884, %arg3 : f32
      linalg.yield %885 : f32
    } -> tensor<1x1x1x288xf32>
    %641 = linalg.init_tensor [1, 1, 1, 288] : tensor<1x1x1x288xf32>
    %642 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%640, %cst_198 : tensor<1x1x1x288xf32>, tensor<1x1x1x288xf32>) outs(%641 : tensor<1x1x1x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x288xf32>
    %643 = linalg.init_tensor [1, 7, 7, 288] : tensor<1x7x7x288xf32>
    %644 = linalg.generic {indexing_maps = [#map5, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%642 : tensor<1x1x1x288xf32>) outs(%643 : tensor<1x7x7x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<1x7x7x288xf32>
    %645 = linalg.init_tensor [1, 7, 7, 288] : tensor<1x7x7x288xf32>
    %646 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%618, %644 : tensor<1x7x7x288xf32>, tensor<1x7x7x288xf32>) outs(%645 : tensor<1x7x7x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x288xf32>
    %647 = linalg.init_tensor [1, 7, 7, 96] : tensor<1x7x7x96xf32>
    %cst_329 = arith.constant 0.000000e+00 : f32
    %648 = linalg.fill ins(%cst_329 : f32) outs(%647 : tensor<1x7x7x96xf32>) -> tensor<1x7x7x96xf32>
    %649 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%646, %cst_255 : tensor<1x7x7x288xf32>, tensor<1x1x288x96xf32>) outs(%648 : tensor<1x7x7x96xf32>) -> tensor<1x7x7x96xf32>
    %650 = linalg.init_tensor [1, 7, 7, 96] : tensor<1x7x7x96xf32>
    %651 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%649, %cst_129 : tensor<1x7x7x96xf32>, tensor<1x7x7x96xf32>) outs(%650 : tensor<1x7x7x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.subf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x96xf32>
    %652 = linalg.init_tensor [1, 7, 7, 96] : tensor<1x7x7x96xf32>
    %653 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%651, %cst_127 : tensor<1x7x7x96xf32>, tensor<1x7x7x96xf32>) outs(%652 : tensor<1x7x7x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x96xf32>
    %654 = linalg.init_tensor [1, 7, 7, 96] : tensor<1x7x7x96xf32>
    %655 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%653, %cst_126 : tensor<1x7x7x96xf32>, tensor<1x7x7x96xf32>) outs(%654 : tensor<1x7x7x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x96xf32>
    %656 = linalg.init_tensor [1, 7, 7, 96] : tensor<1x7x7x96xf32>
    %657 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%655, %cst_128 : tensor<1x7x7x96xf32>, tensor<1x7x7x96xf32>) outs(%656 : tensor<1x7x7x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x96xf32>
    %658 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %cst_330 = arith.constant 0.000000e+00 : f32
    %659 = linalg.fill ins(%cst_330 : f32) outs(%658 : tensor<1x7x7x576xf32>) -> tensor<1x7x7x576xf32>
    %660 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%657, %cst_258 : tensor<1x7x7x96xf32>, tensor<1x1x96x576xf32>) outs(%659 : tensor<1x7x7x576xf32>) -> tensor<1x7x7x576xf32>
    %661 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %662 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%660, %cst_133 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%661 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.subf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x576xf32>
    %663 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %664 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%662, %cst_131 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%663 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x576xf32>
    %665 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %666 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%664, %cst_130 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%665 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x576xf32>
    %667 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %668 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%666, %cst_132 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%667 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x576xf32>
    %669 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %670 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%668, %cst_183 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%669 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x576xf32>
    %671 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %672 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %670, %cst_220 : tensor<f32>, tensor<1x7x7x576xf32>, tensor<f32>) outs(%671 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      %885 = arith.minf %884, %arg3 : f32
      linalg.yield %885 : f32
    } -> tensor<1x7x7x576xf32>
    %673 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %674 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%672, %cst_200 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%673 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x576xf32>
    %675 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %676 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%674, %668 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%675 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x576xf32>
    %cst_331 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_332 = arith.constant 0.000000e+00 : f32
    %677 = linalg.init_tensor [1, 11, 11, 576] : tensor<1x11x11x576xf32>
    %678 = linalg.fill ins(%cst_332 : f32) outs(%677 : tensor<1x11x11x576xf32>) -> tensor<1x11x11x576xf32>
    %679 = tensor.insert_slice %676 into %678[0, 2, 2, 0] [1, 7, 7, 576] [1, 1, 1, 1] : tensor<1x7x7x576xf32> into tensor<1x11x11x576xf32>
    %680 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %cst_333 = arith.constant 0.000000e+00 : f32
    %681 = linalg.fill ins(%cst_333 : f32) outs(%680 : tensor<1x7x7x576xf32>) -> tensor<1x7x7x576xf32>
    %682 = tensor.collapse_shape %cst_134 [[0], [1], [2, 3]] : tensor<5x5x1x576xf32> into tensor<5x5x576xf32>
    %683 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%679, %682 : tensor<1x11x11x576xf32>, tensor<5x5x576xf32>) outs(%681 : tensor<1x7x7x576xf32>) -> tensor<1x7x7x576xf32>
    %684 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %685 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%683, %cst_138 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%684 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.subf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x576xf32>
    %686 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %687 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%685, %cst_136 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%686 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x576xf32>
    %688 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %689 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%687, %cst_135 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%688 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x576xf32>
    %690 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %691 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%689, %cst_137 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%690 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x576xf32>
    %692 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %693 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%691, %cst_183 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%692 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x576xf32>
    %694 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %695 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %693, %cst_220 : tensor<f32>, tensor<1x7x7x576xf32>, tensor<f32>) outs(%694 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      %885 = arith.minf %884, %arg3 : f32
      linalg.yield %885 : f32
    } -> tensor<1x7x7x576xf32>
    %696 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %697 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%695, %cst_200 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%696 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x576xf32>
    %698 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %699 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%697, %691 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%698 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x576xf32>
    %cst_334 = arith.constant 0.000000e+00 : f32
    %700 = linalg.init_tensor [1, 576] : tensor<1x576xf32>
    %701 = linalg.fill ins(%cst_334 : f32) outs(%700 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %702 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%699 : tensor<1x7x7x576xf32>) outs(%701 : tensor<1x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x576xf32>
    %703 = linalg.init_tensor [1, 576] : tensor<1x576xf32>
    %704 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%702, %cst_219 : tensor<1x576xf32>, tensor<1x576xf32>) outs(%703 : tensor<1x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x576xf32>
    %705 = tensor.expand_shape %704 [[0], [1, 2, 3]] : tensor<1x576xf32> into tensor<1x1x1x576xf32>
    %706 = linalg.init_tensor [1, 1, 1, 144] : tensor<1x1x1x144xf32>
    %cst_335 = arith.constant 0.000000e+00 : f32
    %707 = linalg.fill ins(%cst_335 : f32) outs(%706 : tensor<1x1x1x144xf32>) -> tensor<1x1x1x144xf32>
    %708 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%705, %cst_261 : tensor<1x1x1x576xf32>, tensor<1x1x576x144xf32>) outs(%707 : tensor<1x1x1x144xf32>) -> tensor<1x1x1x144xf32>
    %709 = linalg.init_tensor [1, 1, 1, 144] : tensor<1x1x1x144xf32>
    %710 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%708, %cst_139 : tensor<1x1x1x144xf32>, tensor<1x1x1x144xf32>) outs(%709 : tensor<1x1x1x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x144xf32>
    %711 = linalg.init_tensor [1, 1, 1, 144] : tensor<1x1x1x144xf32>
    %712 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%710, %cst_218 : tensor<1x1x1x144xf32>, tensor<1x1x1x144xf32>) outs(%711 : tensor<1x1x1x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x144xf32>
    %713 = linalg.init_tensor [1, 1, 1, 576] : tensor<1x1x1x576xf32>
    %cst_336 = arith.constant 0.000000e+00 : f32
    %714 = linalg.fill ins(%cst_336 : f32) outs(%713 : tensor<1x1x1x576xf32>) -> tensor<1x1x1x576xf32>
    %715 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%712, %cst_260 : tensor<1x1x1x144xf32>, tensor<1x1x144x576xf32>) outs(%714 : tensor<1x1x1x576xf32>) -> tensor<1x1x1x576xf32>
    %716 = linalg.init_tensor [1, 1, 1, 576] : tensor<1x1x1x576xf32>
    %717 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%715, %cst_140 : tensor<1x1x1x576xf32>, tensor<1x1x1x576xf32>) outs(%716 : tensor<1x1x1x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x576xf32>
    %718 = linalg.init_tensor [1, 1, 1, 576] : tensor<1x1x1x576xf32>
    %719 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%717, %cst_182 : tensor<1x1x1x576xf32>, tensor<1x1x1x576xf32>) outs(%718 : tensor<1x1x1x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x576xf32>
    %720 = linalg.init_tensor [1, 1, 1, 576] : tensor<1x1x1x576xf32>
    %721 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %719, %cst_220 : tensor<f32>, tensor<1x1x1x576xf32>, tensor<f32>) outs(%720 : tensor<1x1x1x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      %885 = arith.minf %884, %arg3 : f32
      linalg.yield %885 : f32
    } -> tensor<1x1x1x576xf32>
    %722 = linalg.init_tensor [1, 1, 1, 576] : tensor<1x1x1x576xf32>
    %723 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%721, %cst_199 : tensor<1x1x1x576xf32>, tensor<1x1x1x576xf32>) outs(%722 : tensor<1x1x1x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x576xf32>
    %724 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %725 = linalg.generic {indexing_maps = [#map5, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%723 : tensor<1x1x1x576xf32>) outs(%724 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<1x7x7x576xf32>
    %726 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %727 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%699, %725 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%726 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x576xf32>
    %728 = linalg.init_tensor [1, 7, 7, 96] : tensor<1x7x7x96xf32>
    %cst_337 = arith.constant 0.000000e+00 : f32
    %729 = linalg.fill ins(%cst_337 : f32) outs(%728 : tensor<1x7x7x96xf32>) -> tensor<1x7x7x96xf32>
    %730 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%727, %cst_259 : tensor<1x7x7x576xf32>, tensor<1x1x576x96xf32>) outs(%729 : tensor<1x7x7x96xf32>) -> tensor<1x7x7x96xf32>
    %731 = linalg.init_tensor [1, 7, 7, 96] : tensor<1x7x7x96xf32>
    %732 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%730, %cst_144 : tensor<1x7x7x96xf32>, tensor<1x7x7x96xf32>) outs(%731 : tensor<1x7x7x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.subf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x96xf32>
    %733 = linalg.init_tensor [1, 7, 7, 96] : tensor<1x7x7x96xf32>
    %734 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%732, %cst_142 : tensor<1x7x7x96xf32>, tensor<1x7x7x96xf32>) outs(%733 : tensor<1x7x7x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x96xf32>
    %735 = linalg.init_tensor [1, 7, 7, 96] : tensor<1x7x7x96xf32>
    %736 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%734, %cst_141 : tensor<1x7x7x96xf32>, tensor<1x7x7x96xf32>) outs(%735 : tensor<1x7x7x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x96xf32>
    %737 = linalg.init_tensor [1, 7, 7, 96] : tensor<1x7x7x96xf32>
    %738 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%736, %cst_143 : tensor<1x7x7x96xf32>, tensor<1x7x7x96xf32>) outs(%737 : tensor<1x7x7x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x96xf32>
    %739 = linalg.init_tensor [1, 7, 7, 96] : tensor<1x7x7x96xf32>
    %740 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%657, %738 : tensor<1x7x7x96xf32>, tensor<1x7x7x96xf32>) outs(%739 : tensor<1x7x7x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x96xf32>
    %741 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %cst_338 = arith.constant 0.000000e+00 : f32
    %742 = linalg.fill ins(%cst_338 : f32) outs(%741 : tensor<1x7x7x576xf32>) -> tensor<1x7x7x576xf32>
    %743 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%740, %cst_226 : tensor<1x7x7x96xf32>, tensor<1x1x96x576xf32>) outs(%742 : tensor<1x7x7x576xf32>) -> tensor<1x7x7x576xf32>
    %744 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %745 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%743, %cst_148 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%744 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.subf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x576xf32>
    %746 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %747 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%745, %cst_146 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%746 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x576xf32>
    %748 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %749 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%747, %cst_145 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%748 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x576xf32>
    %750 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %751 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%749, %cst_147 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%750 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x576xf32>
    %752 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %753 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%751, %cst_183 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%752 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x576xf32>
    %754 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %755 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %753, %cst_220 : tensor<f32>, tensor<1x7x7x576xf32>, tensor<f32>) outs(%754 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      %885 = arith.minf %884, %arg3 : f32
      linalg.yield %885 : f32
    } -> tensor<1x7x7x576xf32>
    %756 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %757 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%755, %cst_200 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%756 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x576xf32>
    %758 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %759 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%757, %751 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%758 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x576xf32>
    %cst_339 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_340 = arith.constant 0.000000e+00 : f32
    %760 = linalg.init_tensor [1, 11, 11, 576] : tensor<1x11x11x576xf32>
    %761 = linalg.fill ins(%cst_340 : f32) outs(%760 : tensor<1x11x11x576xf32>) -> tensor<1x11x11x576xf32>
    %762 = tensor.insert_slice %759 into %761[0, 2, 2, 0] [1, 7, 7, 576] [1, 1, 1, 1] : tensor<1x7x7x576xf32> into tensor<1x11x11x576xf32>
    %763 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %cst_341 = arith.constant 0.000000e+00 : f32
    %764 = linalg.fill ins(%cst_341 : f32) outs(%763 : tensor<1x7x7x576xf32>) -> tensor<1x7x7x576xf32>
    %765 = tensor.collapse_shape %cst_149 [[0], [1], [2, 3]] : tensor<5x5x1x576xf32> into tensor<5x5x576xf32>
    %766 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%762, %765 : tensor<1x11x11x576xf32>, tensor<5x5x576xf32>) outs(%764 : tensor<1x7x7x576xf32>) -> tensor<1x7x7x576xf32>
    %767 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %768 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%766, %cst_153 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%767 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.subf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x576xf32>
    %769 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %770 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%768, %cst_151 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%769 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x576xf32>
    %771 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %772 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%770, %cst_150 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%771 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x576xf32>
    %773 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %774 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%772, %cst_152 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%773 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x576xf32>
    %775 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %776 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%774, %cst_183 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%775 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x576xf32>
    %777 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %778 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %776, %cst_220 : tensor<f32>, tensor<1x7x7x576xf32>, tensor<f32>) outs(%777 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      %885 = arith.minf %884, %arg3 : f32
      linalg.yield %885 : f32
    } -> tensor<1x7x7x576xf32>
    %779 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %780 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%778, %cst_200 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%779 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x576xf32>
    %781 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %782 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%780, %774 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%781 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x576xf32>
    %cst_342 = arith.constant 0.000000e+00 : f32
    %783 = linalg.init_tensor [1, 576] : tensor<1x576xf32>
    %784 = linalg.fill ins(%cst_342 : f32) outs(%783 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %785 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%782 : tensor<1x7x7x576xf32>) outs(%784 : tensor<1x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x576xf32>
    %786 = linalg.init_tensor [1, 576] : tensor<1x576xf32>
    %787 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%785, %cst_219 : tensor<1x576xf32>, tensor<1x576xf32>) outs(%786 : tensor<1x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x576xf32>
    %788 = tensor.expand_shape %787 [[0], [1, 2, 3]] : tensor<1x576xf32> into tensor<1x1x1x576xf32>
    %789 = linalg.init_tensor [1, 1, 1, 144] : tensor<1x1x1x144xf32>
    %cst_343 = arith.constant 0.000000e+00 : f32
    %790 = linalg.fill ins(%cst_343 : f32) outs(%789 : tensor<1x1x1x144xf32>) -> tensor<1x1x1x144xf32>
    %791 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%788, %cst_229 : tensor<1x1x1x576xf32>, tensor<1x1x576x144xf32>) outs(%790 : tensor<1x1x1x144xf32>) -> tensor<1x1x1x144xf32>
    %792 = linalg.init_tensor [1, 1, 1, 144] : tensor<1x1x1x144xf32>
    %793 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%791, %cst_154 : tensor<1x1x1x144xf32>, tensor<1x1x1x144xf32>) outs(%792 : tensor<1x1x1x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x144xf32>
    %794 = linalg.init_tensor [1, 1, 1, 144] : tensor<1x1x1x144xf32>
    %795 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%793, %cst_218 : tensor<1x1x1x144xf32>, tensor<1x1x1x144xf32>) outs(%794 : tensor<1x1x1x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x144xf32>
    %796 = linalg.init_tensor [1, 1, 1, 576] : tensor<1x1x1x576xf32>
    %cst_344 = arith.constant 0.000000e+00 : f32
    %797 = linalg.fill ins(%cst_344 : f32) outs(%796 : tensor<1x1x1x576xf32>) -> tensor<1x1x1x576xf32>
    %798 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%795, %cst_228 : tensor<1x1x1x144xf32>, tensor<1x1x144x576xf32>) outs(%797 : tensor<1x1x1x576xf32>) -> tensor<1x1x1x576xf32>
    %799 = linalg.init_tensor [1, 1, 1, 576] : tensor<1x1x1x576xf32>
    %800 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%798, %cst_155 : tensor<1x1x1x576xf32>, tensor<1x1x1x576xf32>) outs(%799 : tensor<1x1x1x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x576xf32>
    %801 = linalg.init_tensor [1, 1, 1, 576] : tensor<1x1x1x576xf32>
    %802 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%800, %cst_182 : tensor<1x1x1x576xf32>, tensor<1x1x1x576xf32>) outs(%801 : tensor<1x1x1x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x576xf32>
    %803 = linalg.init_tensor [1, 1, 1, 576] : tensor<1x1x1x576xf32>
    %804 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %802, %cst_220 : tensor<f32>, tensor<1x1x1x576xf32>, tensor<f32>) outs(%803 : tensor<1x1x1x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      %885 = arith.minf %884, %arg3 : f32
      linalg.yield %885 : f32
    } -> tensor<1x1x1x576xf32>
    %805 = linalg.init_tensor [1, 1, 1, 576] : tensor<1x1x1x576xf32>
    %806 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%804, %cst_199 : tensor<1x1x1x576xf32>, tensor<1x1x1x576xf32>) outs(%805 : tensor<1x1x1x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x576xf32>
    %807 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %808 = linalg.generic {indexing_maps = [#map5, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%806 : tensor<1x1x1x576xf32>) outs(%807 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<1x7x7x576xf32>
    %809 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %810 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%782, %808 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%809 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x576xf32>
    %811 = linalg.init_tensor [1, 7, 7, 96] : tensor<1x7x7x96xf32>
    %cst_345 = arith.constant 0.000000e+00 : f32
    %812 = linalg.fill ins(%cst_345 : f32) outs(%811 : tensor<1x7x7x96xf32>) -> tensor<1x7x7x96xf32>
    %813 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%810, %cst_227 : tensor<1x7x7x576xf32>, tensor<1x1x576x96xf32>) outs(%812 : tensor<1x7x7x96xf32>) -> tensor<1x7x7x96xf32>
    %814 = linalg.init_tensor [1, 7, 7, 96] : tensor<1x7x7x96xf32>
    %815 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%813, %cst_159 : tensor<1x7x7x96xf32>, tensor<1x7x7x96xf32>) outs(%814 : tensor<1x7x7x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.subf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x96xf32>
    %816 = linalg.init_tensor [1, 7, 7, 96] : tensor<1x7x7x96xf32>
    %817 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%815, %cst_157 : tensor<1x7x7x96xf32>, tensor<1x7x7x96xf32>) outs(%816 : tensor<1x7x7x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x96xf32>
    %818 = linalg.init_tensor [1, 7, 7, 96] : tensor<1x7x7x96xf32>
    %819 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%817, %cst_156 : tensor<1x7x7x96xf32>, tensor<1x7x7x96xf32>) outs(%818 : tensor<1x7x7x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x96xf32>
    %820 = linalg.init_tensor [1, 7, 7, 96] : tensor<1x7x7x96xf32>
    %821 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%819, %cst_158 : tensor<1x7x7x96xf32>, tensor<1x7x7x96xf32>) outs(%820 : tensor<1x7x7x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x96xf32>
    %822 = linalg.init_tensor [1, 7, 7, 96] : tensor<1x7x7x96xf32>
    %823 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%740, %821 : tensor<1x7x7x96xf32>, tensor<1x7x7x96xf32>) outs(%822 : tensor<1x7x7x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x96xf32>
    %824 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %cst_346 = arith.constant 0.000000e+00 : f32
    %825 = linalg.fill ins(%cst_346 : f32) outs(%824 : tensor<1x7x7x576xf32>) -> tensor<1x7x7x576xf32>
    %826 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%823, %cst_223 : tensor<1x7x7x96xf32>, tensor<1x1x96x576xf32>) outs(%825 : tensor<1x7x7x576xf32>) -> tensor<1x7x7x576xf32>
    %827 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %828 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%826, %cst_163 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%827 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.subf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x576xf32>
    %829 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %830 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%828, %cst_161 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%829 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x576xf32>
    %831 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %832 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%830, %cst_160 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%831 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x576xf32>
    %833 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %834 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%832, %cst_162 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%833 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x576xf32>
    %835 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %836 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%834, %cst_183 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%835 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x576xf32>
    %837 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %838 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %836, %cst_220 : tensor<f32>, tensor<1x7x7x576xf32>, tensor<f32>) outs(%837 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      %885 = arith.minf %884, %arg3 : f32
      linalg.yield %885 : f32
    } -> tensor<1x7x7x576xf32>
    %839 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %840 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%838, %cst_200 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%839 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x576xf32>
    %841 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %842 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%840, %834 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%841 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x7x7x576xf32>
    %cst_347 = arith.constant 0.000000e+00 : f32
    %843 = linalg.init_tensor [1, 576] : tensor<1x576xf32>
    %844 = linalg.fill ins(%cst_347 : f32) outs(%843 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %845 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%842 : tensor<1x7x7x576xf32>) outs(%844 : tensor<1x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x576xf32>
    %846 = linalg.init_tensor [1, 576] : tensor<1x576xf32>
    %847 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%845, %cst_219 : tensor<1x576xf32>, tensor<1x576xf32>) outs(%846 : tensor<1x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x576xf32>
    %848 = tensor.expand_shape %847 [[0], [1, 2, 3]] : tensor<1x576xf32> into tensor<1x1x1x576xf32>
    %849 = linalg.init_tensor [1, 1, 1, 1024] : tensor<1x1x1x1024xf32>
    %cst_348 = arith.constant 0.000000e+00 : f32
    %850 = linalg.fill ins(%cst_348 : f32) outs(%849 : tensor<1x1x1x1024xf32>) -> tensor<1x1x1x1024xf32>
    %851 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%848, %cst_224 : tensor<1x1x1x576xf32>, tensor<1x1x576x1024xf32>) outs(%850 : tensor<1x1x1x1024xf32>) -> tensor<1x1x1x1024xf32>
    %852 = linalg.init_tensor [1, 1, 1, 1024] : tensor<1x1x1x1024xf32>
    %853 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%851, %cst_164 : tensor<1x1x1x1024xf32>, tensor<1x1x1x1024xf32>) outs(%852 : tensor<1x1x1x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x1024xf32>
    %854 = linalg.init_tensor [1, 1, 1, 1024] : tensor<1x1x1x1024xf32>
    %855 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%853, %cst_184 : tensor<1x1x1x1024xf32>, tensor<1x1x1x1024xf32>) outs(%854 : tensor<1x1x1x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x1024xf32>
    %856 = linalg.init_tensor [1, 1, 1, 1024] : tensor<1x1x1x1024xf32>
    %857 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %855, %cst_220 : tensor<f32>, tensor<1x1x1x1024xf32>, tensor<f32>) outs(%856 : tensor<1x1x1x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      %885 = arith.minf %884, %arg3 : f32
      linalg.yield %885 : f32
    } -> tensor<1x1x1x1024xf32>
    %858 = linalg.init_tensor [1, 1, 1, 1024] : tensor<1x1x1x1024xf32>
    %859 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%857, %cst_201 : tensor<1x1x1x1024xf32>, tensor<1x1x1x1024xf32>) outs(%858 : tensor<1x1x1x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x1024xf32>
    %860 = linalg.init_tensor [1, 1, 1, 1024] : tensor<1x1x1x1024xf32>
    %861 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%859, %853 : tensor<1x1x1x1024xf32>, tensor<1x1x1x1024xf32>) outs(%860 : tensor<1x1x1x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x1024xf32>
    %862 = linalg.init_tensor [1, 1, 1, 1000] : tensor<1x1x1x1000xf32>
    %cst_349 = arith.constant 0.000000e+00 : f32
    %863 = linalg.fill ins(%cst_349 : f32) outs(%862 : tensor<1x1x1x1000xf32>) -> tensor<1x1x1x1000xf32>
    %864 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%861, %cst_265 : tensor<1x1x1x1024xf32>, tensor<1x1x1024x1000xf32>) outs(%863 : tensor<1x1x1x1000xf32>) -> tensor<1x1x1x1000xf32>
    %865 = linalg.init_tensor [1, 1, 1, 1000] : tensor<1x1x1x1000xf32>
    %866 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%864, %cst_165 : tensor<1x1x1x1000xf32>, tensor<1x1x1x1000xf32>) outs(%865 : tensor<1x1x1x1000xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x1x1000xf32>
    %867 = tensor.collapse_shape %866 [[0], [1, 2, 3]] : tensor<1x1x1x1000xf32> into tensor<1x1000xf32>
    %cst_350 = arith.constant 0xFF800000 : f32
    %868 = linalg.init_tensor [1] : tensor<1xf32>
    %869 = linalg.fill ins(%cst_350 : f32) outs(%868 : tensor<1xf32>) -> tensor<1xf32>
    %870 = linalg.generic {indexing_maps = [#map4, #map6], iterator_types = ["parallel", "reduction"]} ins(%867 : tensor<1x1000xf32>) outs(%869 : tensor<1xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %884 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1xf32>
    %871 = linalg.init_tensor [1, 1000] : tensor<1x1000xf32>
    %872 = linalg.generic {indexing_maps = [#map6, #map4], iterator_types = ["parallel", "parallel"]} ins(%870 : tensor<1xf32>) outs(%871 : tensor<1x1000xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<1x1000xf32>
    %873 = linalg.init_tensor [1, 1000] : tensor<1x1000xf32>
    %874 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%867, %872 : tensor<1x1000xf32>, tensor<1x1000xf32>) outs(%873 : tensor<1x1000xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.subf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1000xf32>
    %875 = linalg.init_tensor [1, 1000] : tensor<1x1000xf32>
    %876 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%874 : tensor<1x1000xf32>) outs(%875 : tensor<1x1000xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %884 = math.exp %arg1 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1000xf32>
    %cst_351 = arith.constant 0.000000e+00 : f32
    %877 = linalg.init_tensor [1] : tensor<1xf32>
    %878 = linalg.fill ins(%cst_351 : f32) outs(%877 : tensor<1xf32>) -> tensor<1xf32>
    %879 = linalg.generic {indexing_maps = [#map4, #map6], iterator_types = ["parallel", "reduction"]} ins(%876 : tensor<1x1000xf32>) outs(%878 : tensor<1xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %884 = arith.addf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1xf32>
    %880 = linalg.init_tensor [1, 1000] : tensor<1x1000xf32>
    %881 = linalg.generic {indexing_maps = [#map6, #map4], iterator_types = ["parallel", "parallel"]} ins(%879 : tensor<1xf32>) outs(%880 : tensor<1x1000xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<1x1000xf32>
    %882 = linalg.init_tensor [1, 1000] : tensor<1x1000xf32>
    %883 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%876, %881 : tensor<1x1000xf32>, tensor<1x1000xf32>) outs(%882 : tensor<1x1000xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %884 = arith.divf %arg1, %arg2 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1000xf32>
    return %883 : tensor<1x1000xf32>
  }
}

