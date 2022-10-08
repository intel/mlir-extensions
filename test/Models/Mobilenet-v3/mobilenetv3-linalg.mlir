#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> ()>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, 0, 0, d3)>
#map6 = affine_map<(d0, d1) -> (d0)>
module {

  func.func @main() {

    %0 = arith.constant dense<1.000000e+00> : tensor<1x224x224x3xf32>
    %2 = call @predict(%0) : (tensor<1x224x224x3xf32>) -> tensor<1x1000xf32>
    %unranked = tensor.cast %2 : tensor<1x1000xf32> to tensor<*xf32>
    call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
    return
  }

  func.func private @printMemrefF32(%ptr : tensor<*xf32>)


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
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x224x224x3xf32>
    %2 = linalg.init_tensor [1, 224, 224, 3] : tensor<1x224x224x3xf32>
    %3 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1, %cst_167 : tensor<1x224x224x3xf32>, tensor<1x224x224x3xf32>) outs(%2 : tensor<1x224x224x3xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x224x224x3xf32>
    %4 = linalg.init_tensor [1, 112, 112, 16] : tensor<1x112x112x16xf32>
    %cst_266 = arith.constant 0.000000e+00 : f32
    %5 = linalg.fill ins(%cst_266 : f32) outs(%4 : tensor<1x112x112x16xf32>) -> tensor<1x112x112x16xf32>
    %cst_267 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_268 = arith.constant 0.000000e+00 : f32
    %6 = tensor.pad %3 low[0, 0, 0, 0] high[0, 1, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_268 : f32
    } : tensor<1x224x224x3xf32> to tensor<1x225x225x3xf32>
    %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%6, %cst_225 : tensor<1x225x225x3xf32>, tensor<3x3x3x16xf32>) outs(%5 : tensor<1x112x112x16xf32>) -> tensor<1x112x112x16xf32>
    %8 = linalg.init_tensor [1, 112, 112, 16] : tensor<1x112x112x16xf32>
    %9 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%7, %cst_2 : tensor<1x112x112x16xf32>, tensor<1x112x112x16xf32>) outs(%8 : tensor<1x112x112x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.subf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x112x112x16xf32>
    %10 = linalg.init_tensor [1, 112, 112, 16] : tensor<1x112x112x16xf32>
    %11 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%9, %cst_0 : tensor<1x112x112x16xf32>, tensor<1x112x112x16xf32>) outs(%10 : tensor<1x112x112x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x112x112x16xf32>
    %12 = linalg.init_tensor [1, 112, 112, 16] : tensor<1x112x112x16xf32>
    %13 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%11, %cst : tensor<1x112x112x16xf32>, tensor<1x112x112x16xf32>) outs(%12 : tensor<1x112x112x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x112x112x16xf32>
    %14 = linalg.init_tensor [1, 112, 112, 16] : tensor<1x112x112x16xf32>
    %15 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%13, %cst_1 : tensor<1x112x112x16xf32>, tensor<1x112x112x16xf32>) outs(%14 : tensor<1x112x112x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x112x112x16xf32>
    %16 = linalg.init_tensor [1, 112, 112, 16] : tensor<1x112x112x16xf32>
    %17 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%15, %cst_168 : tensor<1x112x112x16xf32>, tensor<1x112x112x16xf32>) outs(%16 : tensor<1x112x112x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x112x112x16xf32>
    %18 = linalg.init_tensor [1, 112, 112, 16] : tensor<1x112x112x16xf32>
    %19 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %17, %cst_220 : tensor<f32>, tensor<1x112x112x16xf32>, tensor<f32>) outs(%18 : tensor<1x112x112x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      %861 = arith.minf %860, %arg3 : f32
      linalg.yield %861 : f32
    } -> tensor<1x112x112x16xf32>
    %20 = linalg.init_tensor [1, 112, 112, 16] : tensor<1x112x112x16xf32>
    %21 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%19, %cst_185 : tensor<1x112x112x16xf32>, tensor<1x112x112x16xf32>) outs(%20 : tensor<1x112x112x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x112x112x16xf32>
    %22 = linalg.init_tensor [1, 112, 112, 16] : tensor<1x112x112x16xf32>
    %23 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%21, %15 : tensor<1x112x112x16xf32>, tensor<1x112x112x16xf32>) outs(%22 : tensor<1x112x112x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x112x112x16xf32>
    %cst_269 = arith.constant 0.000000e+00 : f32
    %24 = tensor.pad %23 low[0, 0, 0, 0] high[0, 1, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_269 : f32
    } : tensor<1x112x112x16xf32> to tensor<1x113x113x16xf32>
    %25 = linalg.init_tensor [1, 56, 56, 16] : tensor<1x56x56x16xf32>
    %cst_270 = arith.constant 0.000000e+00 : f32
    %26 = linalg.fill ins(%cst_270 : f32) outs(%25 : tensor<1x56x56x16xf32>) -> tensor<1x56x56x16xf32>
    %27 = tensor.collapse_shape %cst_3 [[0], [1], [2, 3]] : tensor<3x3x1x16xf32> into tensor<3x3x16xf32>
    %28 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%24, %27 : tensor<1x113x113x16xf32>, tensor<3x3x16xf32>) outs(%26 : tensor<1x56x56x16xf32>) -> tensor<1x56x56x16xf32>
    %29 = linalg.init_tensor [1, 56, 56, 16] : tensor<1x56x56x16xf32>
    %30 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%28, %cst_7 : tensor<1x56x56x16xf32>, tensor<1x56x56x16xf32>) outs(%29 : tensor<1x56x56x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.subf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x56x56x16xf32>
    %31 = linalg.init_tensor [1, 56, 56, 16] : tensor<1x56x56x16xf32>
    %32 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%30, %cst_5 : tensor<1x56x56x16xf32>, tensor<1x56x56x16xf32>) outs(%31 : tensor<1x56x56x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x56x56x16xf32>
    %33 = linalg.init_tensor [1, 56, 56, 16] : tensor<1x56x56x16xf32>
    %34 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%32, %cst_4 : tensor<1x56x56x16xf32>, tensor<1x56x56x16xf32>) outs(%33 : tensor<1x56x56x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x56x56x16xf32>
    %35 = linalg.init_tensor [1, 56, 56, 16] : tensor<1x56x56x16xf32>
    %36 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%34, %cst_6 : tensor<1x56x56x16xf32>, tensor<1x56x56x16xf32>) outs(%35 : tensor<1x56x56x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x56x56x16xf32>
    %37 = linalg.init_tensor [1, 56, 56, 16] : tensor<1x56x56x16xf32>
    %38 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%36, %cst_202 : tensor<1x56x56x16xf32>, tensor<1x56x56x16xf32>) outs(%37 : tensor<1x56x56x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x56x56x16xf32>
    %cst_271 = arith.constant 0.000000e+00 : f32
    %39 = linalg.init_tensor [1, 16] : tensor<1x16xf32>
    %40 = linalg.fill ins(%cst_271 : f32) outs(%39 : tensor<1x16xf32>) -> tensor<1x16xf32>
    %41 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%38 : tensor<1x56x56x16xf32>) outs(%40 : tensor<1x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x16xf32>
    %42 = linalg.init_tensor [1, 16] : tensor<1x16xf32>
    %43 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%41, %cst_203 : tensor<1x16xf32>, tensor<1x16xf32>) outs(%42 : tensor<1x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x16xf32>
    %44 = tensor.expand_shape %43 [[0], [1, 2, 3]] : tensor<1x16xf32> into tensor<1x1x1x16xf32>
    %45 = linalg.init_tensor [1, 1, 1, 8] : tensor<1x1x1x8xf32>
    %cst_272 = arith.constant 0.000000e+00 : f32
    %46 = linalg.fill ins(%cst_272 : f32) outs(%45 : tensor<1x1x1x8xf32>) -> tensor<1x1x1x8xf32>
    %47 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%44, %cst_264 : tensor<1x1x1x16xf32>, tensor<1x1x16x8xf32>) outs(%46 : tensor<1x1x1x8xf32>) -> tensor<1x1x1x8xf32>
    %48 = linalg.init_tensor [1, 1, 1, 8] : tensor<1x1x1x8xf32>
    %49 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%47, %cst_8 : tensor<1x1x1x8xf32>, tensor<1x1x1x8xf32>) outs(%48 : tensor<1x1x1x8xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x8xf32>
    %50 = linalg.init_tensor [1, 1, 1, 8] : tensor<1x1x1x8xf32>
    %51 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%49, %cst_204 : tensor<1x1x1x8xf32>, tensor<1x1x1x8xf32>) outs(%50 : tensor<1x1x1x8xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x8xf32>
    %52 = linalg.init_tensor [1, 1, 1, 16] : tensor<1x1x1x16xf32>
    %cst_273 = arith.constant 0.000000e+00 : f32
    %53 = linalg.fill ins(%cst_273 : f32) outs(%52 : tensor<1x1x1x16xf32>) -> tensor<1x1x1x16xf32>
    %54 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%51, %cst_263 : tensor<1x1x1x8xf32>, tensor<1x1x8x16xf32>) outs(%53 : tensor<1x1x1x16xf32>) -> tensor<1x1x1x16xf32>
    %55 = linalg.init_tensor [1, 1, 1, 16] : tensor<1x1x1x16xf32>
    %56 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%54, %cst_9 : tensor<1x1x1x16xf32>, tensor<1x1x1x16xf32>) outs(%55 : tensor<1x1x1x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x16xf32>
    %57 = linalg.init_tensor [1, 1, 1, 16] : tensor<1x1x1x16xf32>
    %58 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%56, %cst_169 : tensor<1x1x1x16xf32>, tensor<1x1x1x16xf32>) outs(%57 : tensor<1x1x1x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x16xf32>
    %59 = linalg.init_tensor [1, 1, 1, 16] : tensor<1x1x1x16xf32>
    %60 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %58, %cst_220 : tensor<f32>, tensor<1x1x1x16xf32>, tensor<f32>) outs(%59 : tensor<1x1x1x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      %861 = arith.minf %860, %arg3 : f32
      linalg.yield %861 : f32
    } -> tensor<1x1x1x16xf32>
    %61 = linalg.init_tensor [1, 1, 1, 16] : tensor<1x1x1x16xf32>
    %62 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%60, %cst_186 : tensor<1x1x1x16xf32>, tensor<1x1x1x16xf32>) outs(%61 : tensor<1x1x1x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x16xf32>
    %63 = linalg.init_tensor [1, 56, 56, 16] : tensor<1x56x56x16xf32>
    %64 = linalg.generic {indexing_maps = [#map5, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%62 : tensor<1x1x1x16xf32>) outs(%63 : tensor<1x56x56x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<1x56x56x16xf32>
    %65 = linalg.init_tensor [1, 56, 56, 16] : tensor<1x56x56x16xf32>
    %66 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%38, %64 : tensor<1x56x56x16xf32>, tensor<1x56x56x16xf32>) outs(%65 : tensor<1x56x56x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x56x56x16xf32>
    %67 = linalg.init_tensor [1, 56, 56, 16] : tensor<1x56x56x16xf32>
    %cst_274 = arith.constant 0.000000e+00 : f32
    %68 = linalg.fill ins(%cst_274 : f32) outs(%67 : tensor<1x56x56x16xf32>) -> tensor<1x56x56x16xf32>
    %69 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%66, %cst_262 : tensor<1x56x56x16xf32>, tensor<1x1x16x16xf32>) outs(%68 : tensor<1x56x56x16xf32>) -> tensor<1x56x56x16xf32>
    %70 = linalg.init_tensor [1, 56, 56, 16] : tensor<1x56x56x16xf32>
    %71 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%69, %cst_13 : tensor<1x56x56x16xf32>, tensor<1x56x56x16xf32>) outs(%70 : tensor<1x56x56x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.subf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x56x56x16xf32>
    %72 = linalg.init_tensor [1, 56, 56, 16] : tensor<1x56x56x16xf32>
    %73 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%71, %cst_11 : tensor<1x56x56x16xf32>, tensor<1x56x56x16xf32>) outs(%72 : tensor<1x56x56x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x56x56x16xf32>
    %74 = linalg.init_tensor [1, 56, 56, 16] : tensor<1x56x56x16xf32>
    %75 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%73, %cst_10 : tensor<1x56x56x16xf32>, tensor<1x56x56x16xf32>) outs(%74 : tensor<1x56x56x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x56x56x16xf32>
    %76 = linalg.init_tensor [1, 56, 56, 16] : tensor<1x56x56x16xf32>
    %77 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%75, %cst_12 : tensor<1x56x56x16xf32>, tensor<1x56x56x16xf32>) outs(%76 : tensor<1x56x56x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x56x56x16xf32>
    %78 = linalg.init_tensor [1, 56, 56, 72] : tensor<1x56x56x72xf32>
    %cst_275 = arith.constant 0.000000e+00 : f32
    %79 = linalg.fill ins(%cst_275 : f32) outs(%78 : tensor<1x56x56x72xf32>) -> tensor<1x56x56x72xf32>
    %80 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%77, %cst_230 : tensor<1x56x56x16xf32>, tensor<1x1x16x72xf32>) outs(%79 : tensor<1x56x56x72xf32>) -> tensor<1x56x56x72xf32>
    %81 = linalg.init_tensor [1, 56, 56, 72] : tensor<1x56x56x72xf32>
    %82 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%80, %cst_17 : tensor<1x56x56x72xf32>, tensor<1x56x56x72xf32>) outs(%81 : tensor<1x56x56x72xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.subf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x56x56x72xf32>
    %83 = linalg.init_tensor [1, 56, 56, 72] : tensor<1x56x56x72xf32>
    %84 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%82, %cst_15 : tensor<1x56x56x72xf32>, tensor<1x56x56x72xf32>) outs(%83 : tensor<1x56x56x72xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x56x56x72xf32>
    %85 = linalg.init_tensor [1, 56, 56, 72] : tensor<1x56x56x72xf32>
    %86 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%84, %cst_14 : tensor<1x56x56x72xf32>, tensor<1x56x56x72xf32>) outs(%85 : tensor<1x56x56x72xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x56x56x72xf32>
    %87 = linalg.init_tensor [1, 56, 56, 72] : tensor<1x56x56x72xf32>
    %88 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%86, %cst_16 : tensor<1x56x56x72xf32>, tensor<1x56x56x72xf32>) outs(%87 : tensor<1x56x56x72xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x56x56x72xf32>
    %89 = linalg.init_tensor [1, 56, 56, 72] : tensor<1x56x56x72xf32>
    %90 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%88, %cst_205 : tensor<1x56x56x72xf32>, tensor<1x56x56x72xf32>) outs(%89 : tensor<1x56x56x72xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x56x56x72xf32>
    %cst_276 = arith.constant 0.000000e+00 : f32
    %91 = tensor.pad %90 low[0, 0, 0, 0] high[0, 1, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_276 : f32
    } : tensor<1x56x56x72xf32> to tensor<1x57x57x72xf32>
    %92 = linalg.init_tensor [1, 28, 28, 72] : tensor<1x28x28x72xf32>
    %cst_277 = arith.constant 0.000000e+00 : f32
    %93 = linalg.fill ins(%cst_277 : f32) outs(%92 : tensor<1x28x28x72xf32>) -> tensor<1x28x28x72xf32>
    %94 = tensor.collapse_shape %cst_18 [[0], [1], [2, 3]] : tensor<3x3x1x72xf32> into tensor<3x3x72xf32>
    %95 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%91, %94 : tensor<1x57x57x72xf32>, tensor<3x3x72xf32>) outs(%93 : tensor<1x28x28x72xf32>) -> tensor<1x28x28x72xf32>
    %96 = linalg.init_tensor [1, 28, 28, 72] : tensor<1x28x28x72xf32>
    %97 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%95, %cst_22 : tensor<1x28x28x72xf32>, tensor<1x28x28x72xf32>) outs(%96 : tensor<1x28x28x72xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.subf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x28x28x72xf32>
    %98 = linalg.init_tensor [1, 28, 28, 72] : tensor<1x28x28x72xf32>
    %99 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%97, %cst_20 : tensor<1x28x28x72xf32>, tensor<1x28x28x72xf32>) outs(%98 : tensor<1x28x28x72xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x28x28x72xf32>
    %100 = linalg.init_tensor [1, 28, 28, 72] : tensor<1x28x28x72xf32>
    %101 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%99, %cst_19 : tensor<1x28x28x72xf32>, tensor<1x28x28x72xf32>) outs(%100 : tensor<1x28x28x72xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x28x28x72xf32>
    %102 = linalg.init_tensor [1, 28, 28, 72] : tensor<1x28x28x72xf32>
    %103 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%101, %cst_21 : tensor<1x28x28x72xf32>, tensor<1x28x28x72xf32>) outs(%102 : tensor<1x28x28x72xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x28x28x72xf32>
    %104 = linalg.init_tensor [1, 28, 28, 72] : tensor<1x28x28x72xf32>
    %105 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%103, %cst_206 : tensor<1x28x28x72xf32>, tensor<1x28x28x72xf32>) outs(%104 : tensor<1x28x28x72xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x28x28x72xf32>
    %106 = linalg.init_tensor [1, 28, 28, 24] : tensor<1x28x28x24xf32>
    %cst_278 = arith.constant 0.000000e+00 : f32
    %107 = linalg.fill ins(%cst_278 : f32) outs(%106 : tensor<1x28x28x24xf32>) -> tensor<1x28x28x24xf32>
    %108 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%105, %cst_231 : tensor<1x28x28x72xf32>, tensor<1x1x72x24xf32>) outs(%107 : tensor<1x28x28x24xf32>) -> tensor<1x28x28x24xf32>
    %109 = linalg.init_tensor [1, 28, 28, 24] : tensor<1x28x28x24xf32>
    %110 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%108, %cst_26 : tensor<1x28x28x24xf32>, tensor<1x28x28x24xf32>) outs(%109 : tensor<1x28x28x24xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.subf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x28x28x24xf32>
    %111 = linalg.init_tensor [1, 28, 28, 24] : tensor<1x28x28x24xf32>
    %112 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%110, %cst_24 : tensor<1x28x28x24xf32>, tensor<1x28x28x24xf32>) outs(%111 : tensor<1x28x28x24xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x28x28x24xf32>
    %113 = linalg.init_tensor [1, 28, 28, 24] : tensor<1x28x28x24xf32>
    %114 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%112, %cst_23 : tensor<1x28x28x24xf32>, tensor<1x28x28x24xf32>) outs(%113 : tensor<1x28x28x24xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x28x28x24xf32>
    %115 = linalg.init_tensor [1, 28, 28, 24] : tensor<1x28x28x24xf32>
    %116 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%114, %cst_25 : tensor<1x28x28x24xf32>, tensor<1x28x28x24xf32>) outs(%115 : tensor<1x28x28x24xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x28x28x24xf32>
    %117 = linalg.init_tensor [1, 28, 28, 88] : tensor<1x28x28x88xf32>
    %cst_279 = arith.constant 0.000000e+00 : f32
    %118 = linalg.fill ins(%cst_279 : f32) outs(%117 : tensor<1x28x28x88xf32>) -> tensor<1x28x28x88xf32>
    %119 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%116, %cst_232 : tensor<1x28x28x24xf32>, tensor<1x1x24x88xf32>) outs(%118 : tensor<1x28x28x88xf32>) -> tensor<1x28x28x88xf32>
    %120 = linalg.init_tensor [1, 28, 28, 88] : tensor<1x28x28x88xf32>
    %121 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%119, %cst_30 : tensor<1x28x28x88xf32>, tensor<1x28x28x88xf32>) outs(%120 : tensor<1x28x28x88xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.subf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x28x28x88xf32>
    %122 = linalg.init_tensor [1, 28, 28, 88] : tensor<1x28x28x88xf32>
    %123 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%121, %cst_28 : tensor<1x28x28x88xf32>, tensor<1x28x28x88xf32>) outs(%122 : tensor<1x28x28x88xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x28x28x88xf32>
    %124 = linalg.init_tensor [1, 28, 28, 88] : tensor<1x28x28x88xf32>
    %125 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%123, %cst_27 : tensor<1x28x28x88xf32>, tensor<1x28x28x88xf32>) outs(%124 : tensor<1x28x28x88xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x28x28x88xf32>
    %126 = linalg.init_tensor [1, 28, 28, 88] : tensor<1x28x28x88xf32>
    %127 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%125, %cst_29 : tensor<1x28x28x88xf32>, tensor<1x28x28x88xf32>) outs(%126 : tensor<1x28x28x88xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x28x28x88xf32>
    %128 = linalg.init_tensor [1, 28, 28, 88] : tensor<1x28x28x88xf32>
    %129 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%127, %cst_207 : tensor<1x28x28x88xf32>, tensor<1x28x28x88xf32>) outs(%128 : tensor<1x28x28x88xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x28x28x88xf32>
    %cst_280 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_281 = arith.constant 0.000000e+00 : f32
    %130 = tensor.pad %129 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_281 : f32
    } : tensor<1x28x28x88xf32> to tensor<1x30x30x88xf32>
    %131 = linalg.init_tensor [1, 28, 28, 88] : tensor<1x28x28x88xf32>
    %cst_282 = arith.constant 0.000000e+00 : f32
    %132 = linalg.fill ins(%cst_282 : f32) outs(%131 : tensor<1x28x28x88xf32>) -> tensor<1x28x28x88xf32>
    %133 = tensor.collapse_shape %cst_31 [[0], [1], [2, 3]] : tensor<3x3x1x88xf32> into tensor<3x3x88xf32>
    %134 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%130, %133 : tensor<1x30x30x88xf32>, tensor<3x3x88xf32>) outs(%132 : tensor<1x28x28x88xf32>) -> tensor<1x28x28x88xf32>
    %135 = linalg.init_tensor [1, 28, 28, 88] : tensor<1x28x28x88xf32>
    %136 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%134, %cst_35 : tensor<1x28x28x88xf32>, tensor<1x28x28x88xf32>) outs(%135 : tensor<1x28x28x88xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.subf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x28x28x88xf32>
    %137 = linalg.init_tensor [1, 28, 28, 88] : tensor<1x28x28x88xf32>
    %138 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%136, %cst_33 : tensor<1x28x28x88xf32>, tensor<1x28x28x88xf32>) outs(%137 : tensor<1x28x28x88xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x28x28x88xf32>
    %139 = linalg.init_tensor [1, 28, 28, 88] : tensor<1x28x28x88xf32>
    %140 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%138, %cst_32 : tensor<1x28x28x88xf32>, tensor<1x28x28x88xf32>) outs(%139 : tensor<1x28x28x88xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x28x28x88xf32>
    %141 = linalg.init_tensor [1, 28, 28, 88] : tensor<1x28x28x88xf32>
    %142 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%140, %cst_34 : tensor<1x28x28x88xf32>, tensor<1x28x28x88xf32>) outs(%141 : tensor<1x28x28x88xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x28x28x88xf32>
    %143 = linalg.init_tensor [1, 28, 28, 88] : tensor<1x28x28x88xf32>
    %144 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%142, %cst_207 : tensor<1x28x28x88xf32>, tensor<1x28x28x88xf32>) outs(%143 : tensor<1x28x28x88xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x28x28x88xf32>
    %145 = linalg.init_tensor [1, 28, 28, 24] : tensor<1x28x28x24xf32>
    %cst_283 = arith.constant 0.000000e+00 : f32
    %146 = linalg.fill ins(%cst_283 : f32) outs(%145 : tensor<1x28x28x24xf32>) -> tensor<1x28x28x24xf32>
    %147 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%144, %cst_233 : tensor<1x28x28x88xf32>, tensor<1x1x88x24xf32>) outs(%146 : tensor<1x28x28x24xf32>) -> tensor<1x28x28x24xf32>
    %148 = linalg.init_tensor [1, 28, 28, 24] : tensor<1x28x28x24xf32>
    %149 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%147, %cst_39 : tensor<1x28x28x24xf32>, tensor<1x28x28x24xf32>) outs(%148 : tensor<1x28x28x24xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.subf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x28x28x24xf32>
    %150 = linalg.init_tensor [1, 28, 28, 24] : tensor<1x28x28x24xf32>
    %151 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%149, %cst_37 : tensor<1x28x28x24xf32>, tensor<1x28x28x24xf32>) outs(%150 : tensor<1x28x28x24xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x28x28x24xf32>
    %152 = linalg.init_tensor [1, 28, 28, 24] : tensor<1x28x28x24xf32>
    %153 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%151, %cst_36 : tensor<1x28x28x24xf32>, tensor<1x28x28x24xf32>) outs(%152 : tensor<1x28x28x24xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x28x28x24xf32>
    %154 = linalg.init_tensor [1, 28, 28, 24] : tensor<1x28x28x24xf32>
    %155 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%153, %cst_38 : tensor<1x28x28x24xf32>, tensor<1x28x28x24xf32>) outs(%154 : tensor<1x28x28x24xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x28x28x24xf32>
    %156 = linalg.init_tensor [1, 28, 28, 24] : tensor<1x28x28x24xf32>
    %157 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%116, %155 : tensor<1x28x28x24xf32>, tensor<1x28x28x24xf32>) outs(%156 : tensor<1x28x28x24xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x28x28x24xf32>
    %158 = linalg.init_tensor [1, 28, 28, 96] : tensor<1x28x28x96xf32>
    %cst_284 = arith.constant 0.000000e+00 : f32
    %159 = linalg.fill ins(%cst_284 : f32) outs(%158 : tensor<1x28x28x96xf32>) -> tensor<1x28x28x96xf32>
    %160 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%157, %cst_234 : tensor<1x28x28x24xf32>, tensor<1x1x24x96xf32>) outs(%159 : tensor<1x28x28x96xf32>) -> tensor<1x28x28x96xf32>
    %161 = linalg.init_tensor [1, 28, 28, 96] : tensor<1x28x28x96xf32>
    %162 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%160, %cst_43 : tensor<1x28x28x96xf32>, tensor<1x28x28x96xf32>) outs(%161 : tensor<1x28x28x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.subf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x28x28x96xf32>
    %163 = linalg.init_tensor [1, 28, 28, 96] : tensor<1x28x28x96xf32>
    %164 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%162, %cst_41 : tensor<1x28x28x96xf32>, tensor<1x28x28x96xf32>) outs(%163 : tensor<1x28x28x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x28x28x96xf32>
    %165 = linalg.init_tensor [1, 28, 28, 96] : tensor<1x28x28x96xf32>
    %166 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%164, %cst_40 : tensor<1x28x28x96xf32>, tensor<1x28x28x96xf32>) outs(%165 : tensor<1x28x28x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x28x28x96xf32>
    %167 = linalg.init_tensor [1, 28, 28, 96] : tensor<1x28x28x96xf32>
    %168 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%166, %cst_42 : tensor<1x28x28x96xf32>, tensor<1x28x28x96xf32>) outs(%167 : tensor<1x28x28x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x28x28x96xf32>
    %169 = linalg.init_tensor [1, 28, 28, 96] : tensor<1x28x28x96xf32>
    %170 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%168, %cst_170 : tensor<1x28x28x96xf32>, tensor<1x28x28x96xf32>) outs(%169 : tensor<1x28x28x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x28x28x96xf32>
    %171 = linalg.init_tensor [1, 28, 28, 96] : tensor<1x28x28x96xf32>
    %172 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %170, %cst_220 : tensor<f32>, tensor<1x28x28x96xf32>, tensor<f32>) outs(%171 : tensor<1x28x28x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      %861 = arith.minf %860, %arg3 : f32
      linalg.yield %861 : f32
    } -> tensor<1x28x28x96xf32>
    %173 = linalg.init_tensor [1, 28, 28, 96] : tensor<1x28x28x96xf32>
    %174 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%172, %cst_187 : tensor<1x28x28x96xf32>, tensor<1x28x28x96xf32>) outs(%173 : tensor<1x28x28x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x28x28x96xf32>
    %175 = linalg.init_tensor [1, 28, 28, 96] : tensor<1x28x28x96xf32>
    %176 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%174, %168 : tensor<1x28x28x96xf32>, tensor<1x28x28x96xf32>) outs(%175 : tensor<1x28x28x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x28x28x96xf32>
    %cst_285 = arith.constant 0.000000e+00 : f32
    %177 = tensor.pad %176 low[0, 1, 1, 0] high[0, 2, 2, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_285 : f32
    } : tensor<1x28x28x96xf32> to tensor<1x31x31x96xf32>
    %178 = linalg.init_tensor [1, 14, 14, 96] : tensor<1x14x14x96xf32>
    %cst_286 = arith.constant 0.000000e+00 : f32
    %179 = linalg.fill ins(%cst_286 : f32) outs(%178 : tensor<1x14x14x96xf32>) -> tensor<1x14x14x96xf32>
    %180 = tensor.collapse_shape %cst_44 [[0], [1], [2, 3]] : tensor<5x5x1x96xf32> into tensor<5x5x96xf32>
    %181 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%177, %180 : tensor<1x31x31x96xf32>, tensor<5x5x96xf32>) outs(%179 : tensor<1x14x14x96xf32>) -> tensor<1x14x14x96xf32>
    %182 = linalg.init_tensor [1, 14, 14, 96] : tensor<1x14x14x96xf32>
    %183 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%181, %cst_48 : tensor<1x14x14x96xf32>, tensor<1x14x14x96xf32>) outs(%182 : tensor<1x14x14x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.subf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x96xf32>
    %184 = linalg.init_tensor [1, 14, 14, 96] : tensor<1x14x14x96xf32>
    %185 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%183, %cst_46 : tensor<1x14x14x96xf32>, tensor<1x14x14x96xf32>) outs(%184 : tensor<1x14x14x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x96xf32>
    %186 = linalg.init_tensor [1, 14, 14, 96] : tensor<1x14x14x96xf32>
    %187 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%185, %cst_45 : tensor<1x14x14x96xf32>, tensor<1x14x14x96xf32>) outs(%186 : tensor<1x14x14x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x96xf32>
    %188 = linalg.init_tensor [1, 14, 14, 96] : tensor<1x14x14x96xf32>
    %189 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%187, %cst_47 : tensor<1x14x14x96xf32>, tensor<1x14x14x96xf32>) outs(%188 : tensor<1x14x14x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x96xf32>
    %190 = linalg.init_tensor [1, 14, 14, 96] : tensor<1x14x14x96xf32>
    %191 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%189, %cst_171 : tensor<1x14x14x96xf32>, tensor<1x14x14x96xf32>) outs(%190 : tensor<1x14x14x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x96xf32>
    %192 = linalg.init_tensor [1, 14, 14, 96] : tensor<1x14x14x96xf32>
    %193 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %191, %cst_220 : tensor<f32>, tensor<1x14x14x96xf32>, tensor<f32>) outs(%192 : tensor<1x14x14x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      %861 = arith.minf %860, %arg3 : f32
      linalg.yield %861 : f32
    } -> tensor<1x14x14x96xf32>
    %194 = linalg.init_tensor [1, 14, 14, 96] : tensor<1x14x14x96xf32>
    %195 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%193, %cst_188 : tensor<1x14x14x96xf32>, tensor<1x14x14x96xf32>) outs(%194 : tensor<1x14x14x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x96xf32>
    %196 = linalg.init_tensor [1, 14, 14, 96] : tensor<1x14x14x96xf32>
    %197 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%195, %189 : tensor<1x14x14x96xf32>, tensor<1x14x14x96xf32>) outs(%196 : tensor<1x14x14x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x96xf32>
    %cst_287 = arith.constant 0.000000e+00 : f32
    %198 = linalg.init_tensor [1, 96] : tensor<1x96xf32>
    %199 = linalg.fill ins(%cst_287 : f32) outs(%198 : tensor<1x96xf32>) -> tensor<1x96xf32>
    %200 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%197 : tensor<1x14x14x96xf32>) outs(%199 : tensor<1x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x96xf32>
    %201 = linalg.init_tensor [1, 96] : tensor<1x96xf32>
    %202 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%200, %cst_208 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%201 : tensor<1x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x96xf32>
    %203 = tensor.expand_shape %202 [[0], [1, 2, 3]] : tensor<1x96xf32> into tensor<1x1x1x96xf32>
    %204 = linalg.init_tensor [1, 1, 1, 24] : tensor<1x1x1x24xf32>
    %cst_288 = arith.constant 0.000000e+00 : f32
    %205 = linalg.fill ins(%cst_288 : f32) outs(%204 : tensor<1x1x1x24xf32>) -> tensor<1x1x1x24xf32>
    %206 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%203, %cst_237 : tensor<1x1x1x96xf32>, tensor<1x1x96x24xf32>) outs(%205 : tensor<1x1x1x24xf32>) -> tensor<1x1x1x24xf32>
    %207 = linalg.init_tensor [1, 1, 1, 24] : tensor<1x1x1x24xf32>
    %208 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%206, %cst_49 : tensor<1x1x1x24xf32>, tensor<1x1x1x24xf32>) outs(%207 : tensor<1x1x1x24xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x24xf32>
    %209 = linalg.init_tensor [1, 1, 1, 24] : tensor<1x1x1x24xf32>
    %210 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%208, %cst_209 : tensor<1x1x1x24xf32>, tensor<1x1x1x24xf32>) outs(%209 : tensor<1x1x1x24xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x24xf32>
    %211 = linalg.init_tensor [1, 1, 1, 96] : tensor<1x1x1x96xf32>
    %cst_289 = arith.constant 0.000000e+00 : f32
    %212 = linalg.fill ins(%cst_289 : f32) outs(%211 : tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32>
    %213 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%210, %cst_236 : tensor<1x1x1x24xf32>, tensor<1x1x24x96xf32>) outs(%212 : tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32>
    %214 = linalg.init_tensor [1, 1, 1, 96] : tensor<1x1x1x96xf32>
    %215 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%213, %cst_50 : tensor<1x1x1x96xf32>, tensor<1x1x1x96xf32>) outs(%214 : tensor<1x1x1x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x96xf32>
    %216 = linalg.init_tensor [1, 1, 1, 96] : tensor<1x1x1x96xf32>
    %217 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%215, %cst_172 : tensor<1x1x1x96xf32>, tensor<1x1x1x96xf32>) outs(%216 : tensor<1x1x1x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x96xf32>
    %218 = linalg.init_tensor [1, 1, 1, 96] : tensor<1x1x1x96xf32>
    %219 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %217, %cst_220 : tensor<f32>, tensor<1x1x1x96xf32>, tensor<f32>) outs(%218 : tensor<1x1x1x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      %861 = arith.minf %860, %arg3 : f32
      linalg.yield %861 : f32
    } -> tensor<1x1x1x96xf32>
    %220 = linalg.init_tensor [1, 1, 1, 96] : tensor<1x1x1x96xf32>
    %221 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%219, %cst_189 : tensor<1x1x1x96xf32>, tensor<1x1x1x96xf32>) outs(%220 : tensor<1x1x1x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x96xf32>
    %222 = linalg.init_tensor [1, 14, 14, 96] : tensor<1x14x14x96xf32>
    %223 = linalg.generic {indexing_maps = [#map5, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%221 : tensor<1x1x1x96xf32>) outs(%222 : tensor<1x14x14x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<1x14x14x96xf32>
    %224 = linalg.init_tensor [1, 14, 14, 96] : tensor<1x14x14x96xf32>
    %225 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%197, %223 : tensor<1x14x14x96xf32>, tensor<1x14x14x96xf32>) outs(%224 : tensor<1x14x14x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x96xf32>
    %226 = linalg.init_tensor [1, 14, 14, 40] : tensor<1x14x14x40xf32>
    %cst_290 = arith.constant 0.000000e+00 : f32
    %227 = linalg.fill ins(%cst_290 : f32) outs(%226 : tensor<1x14x14x40xf32>) -> tensor<1x14x14x40xf32>
    %228 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%225, %cst_235 : tensor<1x14x14x96xf32>, tensor<1x1x96x40xf32>) outs(%227 : tensor<1x14x14x40xf32>) -> tensor<1x14x14x40xf32>
    %229 = linalg.init_tensor [1, 14, 14, 40] : tensor<1x14x14x40xf32>
    %230 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%228, %cst_54 : tensor<1x14x14x40xf32>, tensor<1x14x14x40xf32>) outs(%229 : tensor<1x14x14x40xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.subf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x40xf32>
    %231 = linalg.init_tensor [1, 14, 14, 40] : tensor<1x14x14x40xf32>
    %232 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%230, %cst_52 : tensor<1x14x14x40xf32>, tensor<1x14x14x40xf32>) outs(%231 : tensor<1x14x14x40xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x40xf32>
    %233 = linalg.init_tensor [1, 14, 14, 40] : tensor<1x14x14x40xf32>
    %234 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%232, %cst_51 : tensor<1x14x14x40xf32>, tensor<1x14x14x40xf32>) outs(%233 : tensor<1x14x14x40xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x40xf32>
    %235 = linalg.init_tensor [1, 14, 14, 40] : tensor<1x14x14x40xf32>
    %236 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%234, %cst_53 : tensor<1x14x14x40xf32>, tensor<1x14x14x40xf32>) outs(%235 : tensor<1x14x14x40xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x40xf32>
    %237 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %cst_291 = arith.constant 0.000000e+00 : f32
    %238 = linalg.fill ins(%cst_291 : f32) outs(%237 : tensor<1x14x14x240xf32>) -> tensor<1x14x14x240xf32>
    %239 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%236, %cst_238 : tensor<1x14x14x40xf32>, tensor<1x1x40x240xf32>) outs(%238 : tensor<1x14x14x240xf32>) -> tensor<1x14x14x240xf32>
    %240 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %241 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%239, %cst_58 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%240 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.subf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x240xf32>
    %242 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %243 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%241, %cst_56 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%242 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x240xf32>
    %244 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %245 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%243, %cst_55 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%244 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x240xf32>
    %246 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %247 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%245, %cst_57 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%246 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x240xf32>
    %248 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %249 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%247, %cst_173 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%248 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x240xf32>
    %250 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %251 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %249, %cst_220 : tensor<f32>, tensor<1x14x14x240xf32>, tensor<f32>) outs(%250 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      %861 = arith.minf %860, %arg3 : f32
      linalg.yield %861 : f32
    } -> tensor<1x14x14x240xf32>
    %252 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %253 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%251, %cst_190 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%252 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x240xf32>
    %254 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %255 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%253, %247 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%254 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x240xf32>
    %cst_292 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_293 = arith.constant 0.000000e+00 : f32
    %256 = tensor.pad %255 low[0, 2, 2, 0] high[0, 2, 2, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_293 : f32
    } : tensor<1x14x14x240xf32> to tensor<1x18x18x240xf32>
    %257 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %cst_294 = arith.constant 0.000000e+00 : f32
    %258 = linalg.fill ins(%cst_294 : f32) outs(%257 : tensor<1x14x14x240xf32>) -> tensor<1x14x14x240xf32>
    %259 = tensor.collapse_shape %cst_59 [[0], [1], [2, 3]] : tensor<5x5x1x240xf32> into tensor<5x5x240xf32>
    %260 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%256, %259 : tensor<1x18x18x240xf32>, tensor<5x5x240xf32>) outs(%258 : tensor<1x14x14x240xf32>) -> tensor<1x14x14x240xf32>
    %261 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %262 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%260, %cst_63 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%261 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.subf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x240xf32>
    %263 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %264 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%262, %cst_61 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%263 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x240xf32>
    %265 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %266 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%264, %cst_60 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%265 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x240xf32>
    %267 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %268 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%266, %cst_62 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%267 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x240xf32>
    %269 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %270 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%268, %cst_173 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%269 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x240xf32>
    %271 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %272 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %270, %cst_220 : tensor<f32>, tensor<1x14x14x240xf32>, tensor<f32>) outs(%271 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      %861 = arith.minf %860, %arg3 : f32
      linalg.yield %861 : f32
    } -> tensor<1x14x14x240xf32>
    %273 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %274 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%272, %cst_190 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%273 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x240xf32>
    %275 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %276 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%274, %268 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%275 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x240xf32>
    %cst_295 = arith.constant 0.000000e+00 : f32
    %277 = linalg.init_tensor [1, 240] : tensor<1x240xf32>
    %278 = linalg.fill ins(%cst_295 : f32) outs(%277 : tensor<1x240xf32>) -> tensor<1x240xf32>
    %279 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%276 : tensor<1x14x14x240xf32>) outs(%278 : tensor<1x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x240xf32>
    %280 = linalg.init_tensor [1, 240] : tensor<1x240xf32>
    %281 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%279, %cst_210 : tensor<1x240xf32>, tensor<1x240xf32>) outs(%280 : tensor<1x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x240xf32>
    %282 = tensor.expand_shape %281 [[0], [1, 2, 3]] : tensor<1x240xf32> into tensor<1x1x1x240xf32>
    %283 = linalg.init_tensor [1, 1, 1, 64] : tensor<1x1x1x64xf32>
    %cst_296 = arith.constant 0.000000e+00 : f32
    %284 = linalg.fill ins(%cst_296 : f32) outs(%283 : tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xf32>
    %285 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%282, %cst_241 : tensor<1x1x1x240xf32>, tensor<1x1x240x64xf32>) outs(%284 : tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xf32>
    %286 = linalg.init_tensor [1, 1, 1, 64] : tensor<1x1x1x64xf32>
    %287 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%285, %cst_64 : tensor<1x1x1x64xf32>, tensor<1x1x1x64xf32>) outs(%286 : tensor<1x1x1x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x64xf32>
    %288 = linalg.init_tensor [1, 1, 1, 64] : tensor<1x1x1x64xf32>
    %289 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%287, %cst_211 : tensor<1x1x1x64xf32>, tensor<1x1x1x64xf32>) outs(%288 : tensor<1x1x1x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x64xf32>
    %290 = linalg.init_tensor [1, 1, 1, 240] : tensor<1x1x1x240xf32>
    %cst_297 = arith.constant 0.000000e+00 : f32
    %291 = linalg.fill ins(%cst_297 : f32) outs(%290 : tensor<1x1x1x240xf32>) -> tensor<1x1x1x240xf32>
    %292 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%289, %cst_240 : tensor<1x1x1x64xf32>, tensor<1x1x64x240xf32>) outs(%291 : tensor<1x1x1x240xf32>) -> tensor<1x1x1x240xf32>
    %293 = linalg.init_tensor [1, 1, 1, 240] : tensor<1x1x1x240xf32>
    %294 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%292, %cst_65 : tensor<1x1x1x240xf32>, tensor<1x1x1x240xf32>) outs(%293 : tensor<1x1x1x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x240xf32>
    %295 = linalg.init_tensor [1, 1, 1, 240] : tensor<1x1x1x240xf32>
    %296 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%294, %cst_174 : tensor<1x1x1x240xf32>, tensor<1x1x1x240xf32>) outs(%295 : tensor<1x1x1x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x240xf32>
    %297 = linalg.init_tensor [1, 1, 1, 240] : tensor<1x1x1x240xf32>
    %298 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %296, %cst_220 : tensor<f32>, tensor<1x1x1x240xf32>, tensor<f32>) outs(%297 : tensor<1x1x1x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      %861 = arith.minf %860, %arg3 : f32
      linalg.yield %861 : f32
    } -> tensor<1x1x1x240xf32>
    %299 = linalg.init_tensor [1, 1, 1, 240] : tensor<1x1x1x240xf32>
    %300 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%298, %cst_191 : tensor<1x1x1x240xf32>, tensor<1x1x1x240xf32>) outs(%299 : tensor<1x1x1x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x240xf32>
    %301 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %302 = linalg.generic {indexing_maps = [#map5, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%300 : tensor<1x1x1x240xf32>) outs(%301 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<1x14x14x240xf32>
    %303 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %304 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%276, %302 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%303 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x240xf32>
    %305 = linalg.init_tensor [1, 14, 14, 40] : tensor<1x14x14x40xf32>
    %cst_298 = arith.constant 0.000000e+00 : f32
    %306 = linalg.fill ins(%cst_298 : f32) outs(%305 : tensor<1x14x14x40xf32>) -> tensor<1x14x14x40xf32>
    %307 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%304, %cst_239 : tensor<1x14x14x240xf32>, tensor<1x1x240x40xf32>) outs(%306 : tensor<1x14x14x40xf32>) -> tensor<1x14x14x40xf32>
    %308 = linalg.init_tensor [1, 14, 14, 40] : tensor<1x14x14x40xf32>
    %309 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%307, %cst_69 : tensor<1x14x14x40xf32>, tensor<1x14x14x40xf32>) outs(%308 : tensor<1x14x14x40xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.subf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x40xf32>
    %310 = linalg.init_tensor [1, 14, 14, 40] : tensor<1x14x14x40xf32>
    %311 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%309, %cst_67 : tensor<1x14x14x40xf32>, tensor<1x14x14x40xf32>) outs(%310 : tensor<1x14x14x40xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x40xf32>
    %312 = linalg.init_tensor [1, 14, 14, 40] : tensor<1x14x14x40xf32>
    %313 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%311, %cst_66 : tensor<1x14x14x40xf32>, tensor<1x14x14x40xf32>) outs(%312 : tensor<1x14x14x40xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x40xf32>
    %314 = linalg.init_tensor [1, 14, 14, 40] : tensor<1x14x14x40xf32>
    %315 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%313, %cst_68 : tensor<1x14x14x40xf32>, tensor<1x14x14x40xf32>) outs(%314 : tensor<1x14x14x40xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x40xf32>
    %316 = linalg.init_tensor [1, 14, 14, 40] : tensor<1x14x14x40xf32>
    %317 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%236, %315 : tensor<1x14x14x40xf32>, tensor<1x14x14x40xf32>) outs(%316 : tensor<1x14x14x40xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x40xf32>
    %318 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %cst_299 = arith.constant 0.000000e+00 : f32
    %319 = linalg.fill ins(%cst_299 : f32) outs(%318 : tensor<1x14x14x240xf32>) -> tensor<1x14x14x240xf32>
    %320 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%317, %cst_242 : tensor<1x14x14x40xf32>, tensor<1x1x40x240xf32>) outs(%319 : tensor<1x14x14x240xf32>) -> tensor<1x14x14x240xf32>
    %321 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %322 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%320, %cst_73 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%321 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.subf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x240xf32>
    %323 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %324 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%322, %cst_71 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%323 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x240xf32>
    %325 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %326 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%324, %cst_70 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%325 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x240xf32>
    %327 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %328 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%326, %cst_72 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%327 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x240xf32>
    %329 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %330 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%328, %cst_173 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%329 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x240xf32>
    %331 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %332 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %330, %cst_220 : tensor<f32>, tensor<1x14x14x240xf32>, tensor<f32>) outs(%331 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      %861 = arith.minf %860, %arg3 : f32
      linalg.yield %861 : f32
    } -> tensor<1x14x14x240xf32>
    %333 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %334 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%332, %cst_190 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%333 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x240xf32>
    %335 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %336 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%334, %328 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%335 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x240xf32>
    %cst_300 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_301 = arith.constant 0.000000e+00 : f32
    %337 = tensor.pad %336 low[0, 2, 2, 0] high[0, 2, 2, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_301 : f32
    } : tensor<1x14x14x240xf32> to tensor<1x18x18x240xf32>
    %338 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %cst_302 = arith.constant 0.000000e+00 : f32
    %339 = linalg.fill ins(%cst_302 : f32) outs(%338 : tensor<1x14x14x240xf32>) -> tensor<1x14x14x240xf32>
    %340 = tensor.collapse_shape %cst_74 [[0], [1], [2, 3]] : tensor<5x5x1x240xf32> into tensor<5x5x240xf32>
    %341 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%337, %340 : tensor<1x18x18x240xf32>, tensor<5x5x240xf32>) outs(%339 : tensor<1x14x14x240xf32>) -> tensor<1x14x14x240xf32>
    %342 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %343 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%341, %cst_78 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%342 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.subf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x240xf32>
    %344 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %345 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%343, %cst_76 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%344 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x240xf32>
    %346 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %347 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%345, %cst_75 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%346 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x240xf32>
    %348 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %349 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%347, %cst_77 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%348 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x240xf32>
    %350 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %351 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%349, %cst_173 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%350 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x240xf32>
    %352 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %353 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %351, %cst_220 : tensor<f32>, tensor<1x14x14x240xf32>, tensor<f32>) outs(%352 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      %861 = arith.minf %860, %arg3 : f32
      linalg.yield %861 : f32
    } -> tensor<1x14x14x240xf32>
    %354 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %355 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%353, %cst_190 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%354 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x240xf32>
    %356 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %357 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%355, %349 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%356 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x240xf32>
    %cst_303 = arith.constant 0.000000e+00 : f32
    %358 = linalg.init_tensor [1, 240] : tensor<1x240xf32>
    %359 = linalg.fill ins(%cst_303 : f32) outs(%358 : tensor<1x240xf32>) -> tensor<1x240xf32>
    %360 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%357 : tensor<1x14x14x240xf32>) outs(%359 : tensor<1x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x240xf32>
    %361 = linalg.init_tensor [1, 240] : tensor<1x240xf32>
    %362 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%360, %cst_210 : tensor<1x240xf32>, tensor<1x240xf32>) outs(%361 : tensor<1x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x240xf32>
    %363 = tensor.expand_shape %362 [[0], [1, 2, 3]] : tensor<1x240xf32> into tensor<1x1x1x240xf32>
    %364 = linalg.init_tensor [1, 1, 1, 64] : tensor<1x1x1x64xf32>
    %cst_304 = arith.constant 0.000000e+00 : f32
    %365 = linalg.fill ins(%cst_304 : f32) outs(%364 : tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xf32>
    %366 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%363, %cst_245 : tensor<1x1x1x240xf32>, tensor<1x1x240x64xf32>) outs(%365 : tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xf32>
    %367 = linalg.init_tensor [1, 1, 1, 64] : tensor<1x1x1x64xf32>
    %368 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%366, %cst_79 : tensor<1x1x1x64xf32>, tensor<1x1x1x64xf32>) outs(%367 : tensor<1x1x1x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x64xf32>
    %369 = linalg.init_tensor [1, 1, 1, 64] : tensor<1x1x1x64xf32>
    %370 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%368, %cst_211 : tensor<1x1x1x64xf32>, tensor<1x1x1x64xf32>) outs(%369 : tensor<1x1x1x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x64xf32>
    %371 = linalg.init_tensor [1, 1, 1, 240] : tensor<1x1x1x240xf32>
    %cst_305 = arith.constant 0.000000e+00 : f32
    %372 = linalg.fill ins(%cst_305 : f32) outs(%371 : tensor<1x1x1x240xf32>) -> tensor<1x1x1x240xf32>
    %373 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%370, %cst_244 : tensor<1x1x1x64xf32>, tensor<1x1x64x240xf32>) outs(%372 : tensor<1x1x1x240xf32>) -> tensor<1x1x1x240xf32>
    %374 = linalg.init_tensor [1, 1, 1, 240] : tensor<1x1x1x240xf32>
    %375 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%373, %cst_80 : tensor<1x1x1x240xf32>, tensor<1x1x1x240xf32>) outs(%374 : tensor<1x1x1x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x240xf32>
    %376 = linalg.init_tensor [1, 1, 1, 240] : tensor<1x1x1x240xf32>
    %377 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%375, %cst_174 : tensor<1x1x1x240xf32>, tensor<1x1x1x240xf32>) outs(%376 : tensor<1x1x1x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x240xf32>
    %378 = linalg.init_tensor [1, 1, 1, 240] : tensor<1x1x1x240xf32>
    %379 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %377, %cst_220 : tensor<f32>, tensor<1x1x1x240xf32>, tensor<f32>) outs(%378 : tensor<1x1x1x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      %861 = arith.minf %860, %arg3 : f32
      linalg.yield %861 : f32
    } -> tensor<1x1x1x240xf32>
    %380 = linalg.init_tensor [1, 1, 1, 240] : tensor<1x1x1x240xf32>
    %381 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%379, %cst_191 : tensor<1x1x1x240xf32>, tensor<1x1x1x240xf32>) outs(%380 : tensor<1x1x1x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x240xf32>
    %382 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %383 = linalg.generic {indexing_maps = [#map5, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%381 : tensor<1x1x1x240xf32>) outs(%382 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<1x14x14x240xf32>
    %384 = linalg.init_tensor [1, 14, 14, 240] : tensor<1x14x14x240xf32>
    %385 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%357, %383 : tensor<1x14x14x240xf32>, tensor<1x14x14x240xf32>) outs(%384 : tensor<1x14x14x240xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x240xf32>
    %386 = linalg.init_tensor [1, 14, 14, 40] : tensor<1x14x14x40xf32>
    %cst_306 = arith.constant 0.000000e+00 : f32
    %387 = linalg.fill ins(%cst_306 : f32) outs(%386 : tensor<1x14x14x40xf32>) -> tensor<1x14x14x40xf32>
    %388 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%385, %cst_243 : tensor<1x14x14x240xf32>, tensor<1x1x240x40xf32>) outs(%387 : tensor<1x14x14x40xf32>) -> tensor<1x14x14x40xf32>
    %389 = linalg.init_tensor [1, 14, 14, 40] : tensor<1x14x14x40xf32>
    %390 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%388, %cst_84 : tensor<1x14x14x40xf32>, tensor<1x14x14x40xf32>) outs(%389 : tensor<1x14x14x40xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.subf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x40xf32>
    %391 = linalg.init_tensor [1, 14, 14, 40] : tensor<1x14x14x40xf32>
    %392 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%390, %cst_82 : tensor<1x14x14x40xf32>, tensor<1x14x14x40xf32>) outs(%391 : tensor<1x14x14x40xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x40xf32>
    %393 = linalg.init_tensor [1, 14, 14, 40] : tensor<1x14x14x40xf32>
    %394 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%392, %cst_81 : tensor<1x14x14x40xf32>, tensor<1x14x14x40xf32>) outs(%393 : tensor<1x14x14x40xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x40xf32>
    %395 = linalg.init_tensor [1, 14, 14, 40] : tensor<1x14x14x40xf32>
    %396 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%394, %cst_83 : tensor<1x14x14x40xf32>, tensor<1x14x14x40xf32>) outs(%395 : tensor<1x14x14x40xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x40xf32>
    %397 = linalg.init_tensor [1, 14, 14, 40] : tensor<1x14x14x40xf32>
    %398 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%317, %396 : tensor<1x14x14x40xf32>, tensor<1x14x14x40xf32>) outs(%397 : tensor<1x14x14x40xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x40xf32>
    %399 = linalg.init_tensor [1, 14, 14, 120] : tensor<1x14x14x120xf32>
    %cst_307 = arith.constant 0.000000e+00 : f32
    %400 = linalg.fill ins(%cst_307 : f32) outs(%399 : tensor<1x14x14x120xf32>) -> tensor<1x14x14x120xf32>
    %401 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%398, %cst_246 : tensor<1x14x14x40xf32>, tensor<1x1x40x120xf32>) outs(%400 : tensor<1x14x14x120xf32>) -> tensor<1x14x14x120xf32>
    %402 = linalg.init_tensor [1, 14, 14, 120] : tensor<1x14x14x120xf32>
    %403 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%401, %cst_88 : tensor<1x14x14x120xf32>, tensor<1x14x14x120xf32>) outs(%402 : tensor<1x14x14x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.subf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x120xf32>
    %404 = linalg.init_tensor [1, 14, 14, 120] : tensor<1x14x14x120xf32>
    %405 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%403, %cst_86 : tensor<1x14x14x120xf32>, tensor<1x14x14x120xf32>) outs(%404 : tensor<1x14x14x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x120xf32>
    %406 = linalg.init_tensor [1, 14, 14, 120] : tensor<1x14x14x120xf32>
    %407 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%405, %cst_85 : tensor<1x14x14x120xf32>, tensor<1x14x14x120xf32>) outs(%406 : tensor<1x14x14x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x120xf32>
    %408 = linalg.init_tensor [1, 14, 14, 120] : tensor<1x14x14x120xf32>
    %409 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%407, %cst_87 : tensor<1x14x14x120xf32>, tensor<1x14x14x120xf32>) outs(%408 : tensor<1x14x14x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x120xf32>
    %410 = linalg.init_tensor [1, 14, 14, 120] : tensor<1x14x14x120xf32>
    %411 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%409, %cst_175 : tensor<1x14x14x120xf32>, tensor<1x14x14x120xf32>) outs(%410 : tensor<1x14x14x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x120xf32>
    %412 = linalg.init_tensor [1, 14, 14, 120] : tensor<1x14x14x120xf32>
    %413 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %411, %cst_220 : tensor<f32>, tensor<1x14x14x120xf32>, tensor<f32>) outs(%412 : tensor<1x14x14x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      %861 = arith.minf %860, %arg3 : f32
      linalg.yield %861 : f32
    } -> tensor<1x14x14x120xf32>
    %414 = linalg.init_tensor [1, 14, 14, 120] : tensor<1x14x14x120xf32>
    %415 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%413, %cst_192 : tensor<1x14x14x120xf32>, tensor<1x14x14x120xf32>) outs(%414 : tensor<1x14x14x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x120xf32>
    %416 = linalg.init_tensor [1, 14, 14, 120] : tensor<1x14x14x120xf32>
    %417 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%415, %409 : tensor<1x14x14x120xf32>, tensor<1x14x14x120xf32>) outs(%416 : tensor<1x14x14x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x120xf32>
    %cst_308 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_309 = arith.constant 0.000000e+00 : f32
    %418 = tensor.pad %417 low[0, 2, 2, 0] high[0, 2, 2, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_309 : f32
    } : tensor<1x14x14x120xf32> to tensor<1x18x18x120xf32>
    %419 = linalg.init_tensor [1, 14, 14, 120] : tensor<1x14x14x120xf32>
    %cst_310 = arith.constant 0.000000e+00 : f32
    %420 = linalg.fill ins(%cst_310 : f32) outs(%419 : tensor<1x14x14x120xf32>) -> tensor<1x14x14x120xf32>
    %421 = tensor.collapse_shape %cst_89 [[0], [1], [2, 3]] : tensor<5x5x1x120xf32> into tensor<5x5x120xf32>
    %422 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%418, %421 : tensor<1x18x18x120xf32>, tensor<5x5x120xf32>) outs(%420 : tensor<1x14x14x120xf32>) -> tensor<1x14x14x120xf32>
    %423 = linalg.init_tensor [1, 14, 14, 120] : tensor<1x14x14x120xf32>
    %424 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%422, %cst_93 : tensor<1x14x14x120xf32>, tensor<1x14x14x120xf32>) outs(%423 : tensor<1x14x14x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.subf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x120xf32>
    %425 = linalg.init_tensor [1, 14, 14, 120] : tensor<1x14x14x120xf32>
    %426 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%424, %cst_91 : tensor<1x14x14x120xf32>, tensor<1x14x14x120xf32>) outs(%425 : tensor<1x14x14x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x120xf32>
    %427 = linalg.init_tensor [1, 14, 14, 120] : tensor<1x14x14x120xf32>
    %428 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%426, %cst_90 : tensor<1x14x14x120xf32>, tensor<1x14x14x120xf32>) outs(%427 : tensor<1x14x14x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x120xf32>
    %429 = linalg.init_tensor [1, 14, 14, 120] : tensor<1x14x14x120xf32>
    %430 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%428, %cst_92 : tensor<1x14x14x120xf32>, tensor<1x14x14x120xf32>) outs(%429 : tensor<1x14x14x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x120xf32>
    %431 = linalg.init_tensor [1, 14, 14, 120] : tensor<1x14x14x120xf32>
    %432 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%430, %cst_175 : tensor<1x14x14x120xf32>, tensor<1x14x14x120xf32>) outs(%431 : tensor<1x14x14x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x120xf32>
    %433 = linalg.init_tensor [1, 14, 14, 120] : tensor<1x14x14x120xf32>
    %434 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %432, %cst_220 : tensor<f32>, tensor<1x14x14x120xf32>, tensor<f32>) outs(%433 : tensor<1x14x14x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      %861 = arith.minf %860, %arg3 : f32
      linalg.yield %861 : f32
    } -> tensor<1x14x14x120xf32>
    %435 = linalg.init_tensor [1, 14, 14, 120] : tensor<1x14x14x120xf32>
    %436 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%434, %cst_192 : tensor<1x14x14x120xf32>, tensor<1x14x14x120xf32>) outs(%435 : tensor<1x14x14x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x120xf32>
    %437 = linalg.init_tensor [1, 14, 14, 120] : tensor<1x14x14x120xf32>
    %438 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%436, %430 : tensor<1x14x14x120xf32>, tensor<1x14x14x120xf32>) outs(%437 : tensor<1x14x14x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x120xf32>
    %cst_311 = arith.constant 0.000000e+00 : f32
    %439 = linalg.init_tensor [1, 120] : tensor<1x120xf32>
    %440 = linalg.fill ins(%cst_311 : f32) outs(%439 : tensor<1x120xf32>) -> tensor<1x120xf32>
    %441 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%438 : tensor<1x14x14x120xf32>) outs(%440 : tensor<1x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x120xf32>
    %442 = linalg.init_tensor [1, 120] : tensor<1x120xf32>
    %443 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%441, %cst_212 : tensor<1x120xf32>, tensor<1x120xf32>) outs(%442 : tensor<1x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x120xf32>
    %444 = tensor.expand_shape %443 [[0], [1, 2, 3]] : tensor<1x120xf32> into tensor<1x1x1x120xf32>
    %445 = linalg.init_tensor [1, 1, 1, 32] : tensor<1x1x1x32xf32>
    %cst_312 = arith.constant 0.000000e+00 : f32
    %446 = linalg.fill ins(%cst_312 : f32) outs(%445 : tensor<1x1x1x32xf32>) -> tensor<1x1x1x32xf32>
    %447 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%444, %cst_249 : tensor<1x1x1x120xf32>, tensor<1x1x120x32xf32>) outs(%446 : tensor<1x1x1x32xf32>) -> tensor<1x1x1x32xf32>
    %448 = linalg.init_tensor [1, 1, 1, 32] : tensor<1x1x1x32xf32>
    %449 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%447, %cst_94 : tensor<1x1x1x32xf32>, tensor<1x1x1x32xf32>) outs(%448 : tensor<1x1x1x32xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x32xf32>
    %450 = linalg.init_tensor [1, 1, 1, 32] : tensor<1x1x1x32xf32>
    %451 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%449, %cst_213 : tensor<1x1x1x32xf32>, tensor<1x1x1x32xf32>) outs(%450 : tensor<1x1x1x32xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x32xf32>
    %452 = linalg.init_tensor [1, 1, 1, 120] : tensor<1x1x1x120xf32>
    %cst_313 = arith.constant 0.000000e+00 : f32
    %453 = linalg.fill ins(%cst_313 : f32) outs(%452 : tensor<1x1x1x120xf32>) -> tensor<1x1x1x120xf32>
    %454 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%451, %cst_248 : tensor<1x1x1x32xf32>, tensor<1x1x32x120xf32>) outs(%453 : tensor<1x1x1x120xf32>) -> tensor<1x1x1x120xf32>
    %455 = linalg.init_tensor [1, 1, 1, 120] : tensor<1x1x1x120xf32>
    %456 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%454, %cst_95 : tensor<1x1x1x120xf32>, tensor<1x1x1x120xf32>) outs(%455 : tensor<1x1x1x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x120xf32>
    %457 = linalg.init_tensor [1, 1, 1, 120] : tensor<1x1x1x120xf32>
    %458 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%456, %cst_176 : tensor<1x1x1x120xf32>, tensor<1x1x1x120xf32>) outs(%457 : tensor<1x1x1x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x120xf32>
    %459 = linalg.init_tensor [1, 1, 1, 120] : tensor<1x1x1x120xf32>
    %460 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %458, %cst_220 : tensor<f32>, tensor<1x1x1x120xf32>, tensor<f32>) outs(%459 : tensor<1x1x1x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      %861 = arith.minf %860, %arg3 : f32
      linalg.yield %861 : f32
    } -> tensor<1x1x1x120xf32>
    %461 = linalg.init_tensor [1, 1, 1, 120] : tensor<1x1x1x120xf32>
    %462 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%460, %cst_193 : tensor<1x1x1x120xf32>, tensor<1x1x1x120xf32>) outs(%461 : tensor<1x1x1x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x120xf32>
    %463 = linalg.init_tensor [1, 14, 14, 120] : tensor<1x14x14x120xf32>
    %464 = linalg.generic {indexing_maps = [#map5, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%462 : tensor<1x1x1x120xf32>) outs(%463 : tensor<1x14x14x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<1x14x14x120xf32>
    %465 = linalg.init_tensor [1, 14, 14, 120] : tensor<1x14x14x120xf32>
    %466 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%438, %464 : tensor<1x14x14x120xf32>, tensor<1x14x14x120xf32>) outs(%465 : tensor<1x14x14x120xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x120xf32>
    %467 = linalg.init_tensor [1, 14, 14, 48] : tensor<1x14x14x48xf32>
    %cst_314 = arith.constant 0.000000e+00 : f32
    %468 = linalg.fill ins(%cst_314 : f32) outs(%467 : tensor<1x14x14x48xf32>) -> tensor<1x14x14x48xf32>
    %469 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%466, %cst_247 : tensor<1x14x14x120xf32>, tensor<1x1x120x48xf32>) outs(%468 : tensor<1x14x14x48xf32>) -> tensor<1x14x14x48xf32>
    %470 = linalg.init_tensor [1, 14, 14, 48] : tensor<1x14x14x48xf32>
    %471 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%469, %cst_99 : tensor<1x14x14x48xf32>, tensor<1x14x14x48xf32>) outs(%470 : tensor<1x14x14x48xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.subf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x48xf32>
    %472 = linalg.init_tensor [1, 14, 14, 48] : tensor<1x14x14x48xf32>
    %473 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%471, %cst_97 : tensor<1x14x14x48xf32>, tensor<1x14x14x48xf32>) outs(%472 : tensor<1x14x14x48xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x48xf32>
    %474 = linalg.init_tensor [1, 14, 14, 48] : tensor<1x14x14x48xf32>
    %475 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%473, %cst_96 : tensor<1x14x14x48xf32>, tensor<1x14x14x48xf32>) outs(%474 : tensor<1x14x14x48xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x48xf32>
    %476 = linalg.init_tensor [1, 14, 14, 48] : tensor<1x14x14x48xf32>
    %477 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%475, %cst_98 : tensor<1x14x14x48xf32>, tensor<1x14x14x48xf32>) outs(%476 : tensor<1x14x14x48xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x48xf32>
    %478 = linalg.init_tensor [1, 14, 14, 144] : tensor<1x14x14x144xf32>
    %cst_315 = arith.constant 0.000000e+00 : f32
    %479 = linalg.fill ins(%cst_315 : f32) outs(%478 : tensor<1x14x14x144xf32>) -> tensor<1x14x14x144xf32>
    %480 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%477, %cst_250 : tensor<1x14x14x48xf32>, tensor<1x1x48x144xf32>) outs(%479 : tensor<1x14x14x144xf32>) -> tensor<1x14x14x144xf32>
    %481 = linalg.init_tensor [1, 14, 14, 144] : tensor<1x14x14x144xf32>
    %482 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%480, %cst_103 : tensor<1x14x14x144xf32>, tensor<1x14x14x144xf32>) outs(%481 : tensor<1x14x14x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.subf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x144xf32>
    %483 = linalg.init_tensor [1, 14, 14, 144] : tensor<1x14x14x144xf32>
    %484 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%482, %cst_101 : tensor<1x14x14x144xf32>, tensor<1x14x14x144xf32>) outs(%483 : tensor<1x14x14x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x144xf32>
    %485 = linalg.init_tensor [1, 14, 14, 144] : tensor<1x14x14x144xf32>
    %486 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%484, %cst_100 : tensor<1x14x14x144xf32>, tensor<1x14x14x144xf32>) outs(%485 : tensor<1x14x14x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x144xf32>
    %487 = linalg.init_tensor [1, 14, 14, 144] : tensor<1x14x14x144xf32>
    %488 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%486, %cst_102 : tensor<1x14x14x144xf32>, tensor<1x14x14x144xf32>) outs(%487 : tensor<1x14x14x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x144xf32>
    %489 = linalg.init_tensor [1, 14, 14, 144] : tensor<1x14x14x144xf32>
    %490 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%488, %cst_177 : tensor<1x14x14x144xf32>, tensor<1x14x14x144xf32>) outs(%489 : tensor<1x14x14x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x144xf32>
    %491 = linalg.init_tensor [1, 14, 14, 144] : tensor<1x14x14x144xf32>
    %492 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %490, %cst_220 : tensor<f32>, tensor<1x14x14x144xf32>, tensor<f32>) outs(%491 : tensor<1x14x14x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      %861 = arith.minf %860, %arg3 : f32
      linalg.yield %861 : f32
    } -> tensor<1x14x14x144xf32>
    %493 = linalg.init_tensor [1, 14, 14, 144] : tensor<1x14x14x144xf32>
    %494 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%492, %cst_194 : tensor<1x14x14x144xf32>, tensor<1x14x14x144xf32>) outs(%493 : tensor<1x14x14x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x144xf32>
    %495 = linalg.init_tensor [1, 14, 14, 144] : tensor<1x14x14x144xf32>
    %496 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%494, %488 : tensor<1x14x14x144xf32>, tensor<1x14x14x144xf32>) outs(%495 : tensor<1x14x14x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x144xf32>
    %cst_316 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_317 = arith.constant 0.000000e+00 : f32
    %497 = tensor.pad %496 low[0, 2, 2, 0] high[0, 2, 2, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_317 : f32
    } : tensor<1x14x14x144xf32> to tensor<1x18x18x144xf32>
    %498 = linalg.init_tensor [1, 14, 14, 144] : tensor<1x14x14x144xf32>
    %cst_318 = arith.constant 0.000000e+00 : f32
    %499 = linalg.fill ins(%cst_318 : f32) outs(%498 : tensor<1x14x14x144xf32>) -> tensor<1x14x14x144xf32>
    %500 = tensor.collapse_shape %cst_104 [[0], [1], [2, 3]] : tensor<5x5x1x144xf32> into tensor<5x5x144xf32>
    %501 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%497, %500 : tensor<1x18x18x144xf32>, tensor<5x5x144xf32>) outs(%499 : tensor<1x14x14x144xf32>) -> tensor<1x14x14x144xf32>
    %502 = linalg.init_tensor [1, 14, 14, 144] : tensor<1x14x14x144xf32>
    %503 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%501, %cst_108 : tensor<1x14x14x144xf32>, tensor<1x14x14x144xf32>) outs(%502 : tensor<1x14x14x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.subf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x144xf32>
    %504 = linalg.init_tensor [1, 14, 14, 144] : tensor<1x14x14x144xf32>
    %505 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%503, %cst_106 : tensor<1x14x14x144xf32>, tensor<1x14x14x144xf32>) outs(%504 : tensor<1x14x14x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x144xf32>
    %506 = linalg.init_tensor [1, 14, 14, 144] : tensor<1x14x14x144xf32>
    %507 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%505, %cst_105 : tensor<1x14x14x144xf32>, tensor<1x14x14x144xf32>) outs(%506 : tensor<1x14x14x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x144xf32>
    %508 = linalg.init_tensor [1, 14, 14, 144] : tensor<1x14x14x144xf32>
    %509 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%507, %cst_107 : tensor<1x14x14x144xf32>, tensor<1x14x14x144xf32>) outs(%508 : tensor<1x14x14x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x144xf32>
    %510 = linalg.init_tensor [1, 14, 14, 144] : tensor<1x14x14x144xf32>
    %511 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%509, %cst_177 : tensor<1x14x14x144xf32>, tensor<1x14x14x144xf32>) outs(%510 : tensor<1x14x14x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x144xf32>
    %512 = linalg.init_tensor [1, 14, 14, 144] : tensor<1x14x14x144xf32>
    %513 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %511, %cst_220 : tensor<f32>, tensor<1x14x14x144xf32>, tensor<f32>) outs(%512 : tensor<1x14x14x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      %861 = arith.minf %860, %arg3 : f32
      linalg.yield %861 : f32
    } -> tensor<1x14x14x144xf32>
    %514 = linalg.init_tensor [1, 14, 14, 144] : tensor<1x14x14x144xf32>
    %515 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%513, %cst_194 : tensor<1x14x14x144xf32>, tensor<1x14x14x144xf32>) outs(%514 : tensor<1x14x14x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x144xf32>
    %516 = linalg.init_tensor [1, 14, 14, 144] : tensor<1x14x14x144xf32>
    %517 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%515, %509 : tensor<1x14x14x144xf32>, tensor<1x14x14x144xf32>) outs(%516 : tensor<1x14x14x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x144xf32>
    %cst_319 = arith.constant 0.000000e+00 : f32
    %518 = linalg.init_tensor [1, 144] : tensor<1x144xf32>
    %519 = linalg.fill ins(%cst_319 : f32) outs(%518 : tensor<1x144xf32>) -> tensor<1x144xf32>
    %520 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%517 : tensor<1x14x14x144xf32>) outs(%519 : tensor<1x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x144xf32>
    %521 = linalg.init_tensor [1, 144] : tensor<1x144xf32>
    %522 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%520, %cst_214 : tensor<1x144xf32>, tensor<1x144xf32>) outs(%521 : tensor<1x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x144xf32>
    %523 = tensor.expand_shape %522 [[0], [1, 2, 3]] : tensor<1x144xf32> into tensor<1x1x1x144xf32>
    %524 = linalg.init_tensor [1, 1, 1, 40] : tensor<1x1x1x40xf32>
    %cst_320 = arith.constant 0.000000e+00 : f32
    %525 = linalg.fill ins(%cst_320 : f32) outs(%524 : tensor<1x1x1x40xf32>) -> tensor<1x1x1x40xf32>
    %526 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%523, %cst_253 : tensor<1x1x1x144xf32>, tensor<1x1x144x40xf32>) outs(%525 : tensor<1x1x1x40xf32>) -> tensor<1x1x1x40xf32>
    %527 = linalg.init_tensor [1, 1, 1, 40] : tensor<1x1x1x40xf32>
    %528 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%526, %cst_109 : tensor<1x1x1x40xf32>, tensor<1x1x1x40xf32>) outs(%527 : tensor<1x1x1x40xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x40xf32>
    %529 = linalg.init_tensor [1, 1, 1, 40] : tensor<1x1x1x40xf32>
    %530 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%528, %cst_215 : tensor<1x1x1x40xf32>, tensor<1x1x1x40xf32>) outs(%529 : tensor<1x1x1x40xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x40xf32>
    %531 = linalg.init_tensor [1, 1, 1, 144] : tensor<1x1x1x144xf32>
    %cst_321 = arith.constant 0.000000e+00 : f32
    %532 = linalg.fill ins(%cst_321 : f32) outs(%531 : tensor<1x1x1x144xf32>) -> tensor<1x1x1x144xf32>
    %533 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%530, %cst_252 : tensor<1x1x1x40xf32>, tensor<1x1x40x144xf32>) outs(%532 : tensor<1x1x1x144xf32>) -> tensor<1x1x1x144xf32>
    %534 = linalg.init_tensor [1, 1, 1, 144] : tensor<1x1x1x144xf32>
    %535 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%533, %cst_110 : tensor<1x1x1x144xf32>, tensor<1x1x1x144xf32>) outs(%534 : tensor<1x1x1x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x144xf32>
    %536 = linalg.init_tensor [1, 1, 1, 144] : tensor<1x1x1x144xf32>
    %537 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%535, %cst_178 : tensor<1x1x1x144xf32>, tensor<1x1x1x144xf32>) outs(%536 : tensor<1x1x1x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x144xf32>
    %538 = linalg.init_tensor [1, 1, 1, 144] : tensor<1x1x1x144xf32>
    %539 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %537, %cst_220 : tensor<f32>, tensor<1x1x1x144xf32>, tensor<f32>) outs(%538 : tensor<1x1x1x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      %861 = arith.minf %860, %arg3 : f32
      linalg.yield %861 : f32
    } -> tensor<1x1x1x144xf32>
    %540 = linalg.init_tensor [1, 1, 1, 144] : tensor<1x1x1x144xf32>
    %541 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%539, %cst_195 : tensor<1x1x1x144xf32>, tensor<1x1x1x144xf32>) outs(%540 : tensor<1x1x1x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x144xf32>
    %542 = linalg.init_tensor [1, 14, 14, 144] : tensor<1x14x14x144xf32>
    %543 = linalg.generic {indexing_maps = [#map5, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%541 : tensor<1x1x1x144xf32>) outs(%542 : tensor<1x14x14x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<1x14x14x144xf32>
    %544 = linalg.init_tensor [1, 14, 14, 144] : tensor<1x14x14x144xf32>
    %545 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%517, %543 : tensor<1x14x14x144xf32>, tensor<1x14x14x144xf32>) outs(%544 : tensor<1x14x14x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x144xf32>
    %546 = linalg.init_tensor [1, 14, 14, 48] : tensor<1x14x14x48xf32>
    %cst_322 = arith.constant 0.000000e+00 : f32
    %547 = linalg.fill ins(%cst_322 : f32) outs(%546 : tensor<1x14x14x48xf32>) -> tensor<1x14x14x48xf32>
    %548 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%545, %cst_251 : tensor<1x14x14x144xf32>, tensor<1x1x144x48xf32>) outs(%547 : tensor<1x14x14x48xf32>) -> tensor<1x14x14x48xf32>
    %549 = linalg.init_tensor [1, 14, 14, 48] : tensor<1x14x14x48xf32>
    %550 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%548, %cst_114 : tensor<1x14x14x48xf32>, tensor<1x14x14x48xf32>) outs(%549 : tensor<1x14x14x48xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.subf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x48xf32>
    %551 = linalg.init_tensor [1, 14, 14, 48] : tensor<1x14x14x48xf32>
    %552 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%550, %cst_112 : tensor<1x14x14x48xf32>, tensor<1x14x14x48xf32>) outs(%551 : tensor<1x14x14x48xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x48xf32>
    %553 = linalg.init_tensor [1, 14, 14, 48] : tensor<1x14x14x48xf32>
    %554 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%552, %cst_111 : tensor<1x14x14x48xf32>, tensor<1x14x14x48xf32>) outs(%553 : tensor<1x14x14x48xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x48xf32>
    %555 = linalg.init_tensor [1, 14, 14, 48] : tensor<1x14x14x48xf32>
    %556 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%554, %cst_113 : tensor<1x14x14x48xf32>, tensor<1x14x14x48xf32>) outs(%555 : tensor<1x14x14x48xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x48xf32>
    %557 = linalg.init_tensor [1, 14, 14, 48] : tensor<1x14x14x48xf32>
    %558 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%477, %556 : tensor<1x14x14x48xf32>, tensor<1x14x14x48xf32>) outs(%557 : tensor<1x14x14x48xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x48xf32>
    %559 = linalg.init_tensor [1, 14, 14, 288] : tensor<1x14x14x288xf32>
    %cst_323 = arith.constant 0.000000e+00 : f32
    %560 = linalg.fill ins(%cst_323 : f32) outs(%559 : tensor<1x14x14x288xf32>) -> tensor<1x14x14x288xf32>
    %561 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%558, %cst_254 : tensor<1x14x14x48xf32>, tensor<1x1x48x288xf32>) outs(%560 : tensor<1x14x14x288xf32>) -> tensor<1x14x14x288xf32>
    %562 = linalg.init_tensor [1, 14, 14, 288] : tensor<1x14x14x288xf32>
    %563 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%561, %cst_118 : tensor<1x14x14x288xf32>, tensor<1x14x14x288xf32>) outs(%562 : tensor<1x14x14x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.subf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x288xf32>
    %564 = linalg.init_tensor [1, 14, 14, 288] : tensor<1x14x14x288xf32>
    %565 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%563, %cst_116 : tensor<1x14x14x288xf32>, tensor<1x14x14x288xf32>) outs(%564 : tensor<1x14x14x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x288xf32>
    %566 = linalg.init_tensor [1, 14, 14, 288] : tensor<1x14x14x288xf32>
    %567 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%565, %cst_115 : tensor<1x14x14x288xf32>, tensor<1x14x14x288xf32>) outs(%566 : tensor<1x14x14x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x288xf32>
    %568 = linalg.init_tensor [1, 14, 14, 288] : tensor<1x14x14x288xf32>
    %569 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%567, %cst_117 : tensor<1x14x14x288xf32>, tensor<1x14x14x288xf32>) outs(%568 : tensor<1x14x14x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x288xf32>
    %570 = linalg.init_tensor [1, 14, 14, 288] : tensor<1x14x14x288xf32>
    %571 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%569, %cst_179 : tensor<1x14x14x288xf32>, tensor<1x14x14x288xf32>) outs(%570 : tensor<1x14x14x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x288xf32>
    %572 = linalg.init_tensor [1, 14, 14, 288] : tensor<1x14x14x288xf32>
    %573 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %571, %cst_220 : tensor<f32>, tensor<1x14x14x288xf32>, tensor<f32>) outs(%572 : tensor<1x14x14x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      %861 = arith.minf %860, %arg3 : f32
      linalg.yield %861 : f32
    } -> tensor<1x14x14x288xf32>
    %574 = linalg.init_tensor [1, 14, 14, 288] : tensor<1x14x14x288xf32>
    %575 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%573, %cst_196 : tensor<1x14x14x288xf32>, tensor<1x14x14x288xf32>) outs(%574 : tensor<1x14x14x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x288xf32>
    %576 = linalg.init_tensor [1, 14, 14, 288] : tensor<1x14x14x288xf32>
    %577 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%575, %569 : tensor<1x14x14x288xf32>, tensor<1x14x14x288xf32>) outs(%576 : tensor<1x14x14x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x14x14x288xf32>
    %cst_324 = arith.constant 0.000000e+00 : f32
    %578 = tensor.pad %577 low[0, 1, 1, 0] high[0, 2, 2, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_324 : f32
    } : tensor<1x14x14x288xf32> to tensor<1x17x17x288xf32>
    %579 = linalg.init_tensor [1, 7, 7, 288] : tensor<1x7x7x288xf32>
    %cst_325 = arith.constant 0.000000e+00 : f32
    %580 = linalg.fill ins(%cst_325 : f32) outs(%579 : tensor<1x7x7x288xf32>) -> tensor<1x7x7x288xf32>
    %581 = tensor.collapse_shape %cst_119 [[0], [1], [2, 3]] : tensor<5x5x1x288xf32> into tensor<5x5x288xf32>
    %582 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%578, %581 : tensor<1x17x17x288xf32>, tensor<5x5x288xf32>) outs(%580 : tensor<1x7x7x288xf32>) -> tensor<1x7x7x288xf32>
    %583 = linalg.init_tensor [1, 7, 7, 288] : tensor<1x7x7x288xf32>
    %584 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%582, %cst_123 : tensor<1x7x7x288xf32>, tensor<1x7x7x288xf32>) outs(%583 : tensor<1x7x7x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.subf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x288xf32>
    %585 = linalg.init_tensor [1, 7, 7, 288] : tensor<1x7x7x288xf32>
    %586 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%584, %cst_121 : tensor<1x7x7x288xf32>, tensor<1x7x7x288xf32>) outs(%585 : tensor<1x7x7x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x288xf32>
    %587 = linalg.init_tensor [1, 7, 7, 288] : tensor<1x7x7x288xf32>
    %588 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%586, %cst_120 : tensor<1x7x7x288xf32>, tensor<1x7x7x288xf32>) outs(%587 : tensor<1x7x7x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x288xf32>
    %589 = linalg.init_tensor [1, 7, 7, 288] : tensor<1x7x7x288xf32>
    %590 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%588, %cst_122 : tensor<1x7x7x288xf32>, tensor<1x7x7x288xf32>) outs(%589 : tensor<1x7x7x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x288xf32>
    %591 = linalg.init_tensor [1, 7, 7, 288] : tensor<1x7x7x288xf32>
    %592 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%590, %cst_180 : tensor<1x7x7x288xf32>, tensor<1x7x7x288xf32>) outs(%591 : tensor<1x7x7x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x288xf32>
    %593 = linalg.init_tensor [1, 7, 7, 288] : tensor<1x7x7x288xf32>
    %594 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %592, %cst_220 : tensor<f32>, tensor<1x7x7x288xf32>, tensor<f32>) outs(%593 : tensor<1x7x7x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      %861 = arith.minf %860, %arg3 : f32
      linalg.yield %861 : f32
    } -> tensor<1x7x7x288xf32>
    %595 = linalg.init_tensor [1, 7, 7, 288] : tensor<1x7x7x288xf32>
    %596 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%594, %cst_197 : tensor<1x7x7x288xf32>, tensor<1x7x7x288xf32>) outs(%595 : tensor<1x7x7x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x288xf32>
    %597 = linalg.init_tensor [1, 7, 7, 288] : tensor<1x7x7x288xf32>
    %598 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%596, %590 : tensor<1x7x7x288xf32>, tensor<1x7x7x288xf32>) outs(%597 : tensor<1x7x7x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x288xf32>
    %cst_326 = arith.constant 0.000000e+00 : f32
    %599 = linalg.init_tensor [1, 288] : tensor<1x288xf32>
    %600 = linalg.fill ins(%cst_326 : f32) outs(%599 : tensor<1x288xf32>) -> tensor<1x288xf32>
    %601 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%598 : tensor<1x7x7x288xf32>) outs(%600 : tensor<1x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x288xf32>
    %602 = linalg.init_tensor [1, 288] : tensor<1x288xf32>
    %603 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%601, %cst_216 : tensor<1x288xf32>, tensor<1x288xf32>) outs(%602 : tensor<1x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x288xf32>
    %604 = tensor.expand_shape %603 [[0], [1, 2, 3]] : tensor<1x288xf32> into tensor<1x1x1x288xf32>
    %605 = linalg.init_tensor [1, 1, 1, 72] : tensor<1x1x1x72xf32>
    %cst_327 = arith.constant 0.000000e+00 : f32
    %606 = linalg.fill ins(%cst_327 : f32) outs(%605 : tensor<1x1x1x72xf32>) -> tensor<1x1x1x72xf32>
    %607 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%604, %cst_257 : tensor<1x1x1x288xf32>, tensor<1x1x288x72xf32>) outs(%606 : tensor<1x1x1x72xf32>) -> tensor<1x1x1x72xf32>
    %608 = linalg.init_tensor [1, 1, 1, 72] : tensor<1x1x1x72xf32>
    %609 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%607, %cst_124 : tensor<1x1x1x72xf32>, tensor<1x1x1x72xf32>) outs(%608 : tensor<1x1x1x72xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x72xf32>
    %610 = linalg.init_tensor [1, 1, 1, 72] : tensor<1x1x1x72xf32>
    %611 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%609, %cst_217 : tensor<1x1x1x72xf32>, tensor<1x1x1x72xf32>) outs(%610 : tensor<1x1x1x72xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x72xf32>
    %612 = linalg.init_tensor [1, 1, 1, 288] : tensor<1x1x1x288xf32>
    %cst_328 = arith.constant 0.000000e+00 : f32
    %613 = linalg.fill ins(%cst_328 : f32) outs(%612 : tensor<1x1x1x288xf32>) -> tensor<1x1x1x288xf32>
    %614 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%611, %cst_256 : tensor<1x1x1x72xf32>, tensor<1x1x72x288xf32>) outs(%613 : tensor<1x1x1x288xf32>) -> tensor<1x1x1x288xf32>
    %615 = linalg.init_tensor [1, 1, 1, 288] : tensor<1x1x1x288xf32>
    %616 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%614, %cst_125 : tensor<1x1x1x288xf32>, tensor<1x1x1x288xf32>) outs(%615 : tensor<1x1x1x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x288xf32>
    %617 = linalg.init_tensor [1, 1, 1, 288] : tensor<1x1x1x288xf32>
    %618 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%616, %cst_181 : tensor<1x1x1x288xf32>, tensor<1x1x1x288xf32>) outs(%617 : tensor<1x1x1x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x288xf32>
    %619 = linalg.init_tensor [1, 1, 1, 288] : tensor<1x1x1x288xf32>
    %620 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %618, %cst_220 : tensor<f32>, tensor<1x1x1x288xf32>, tensor<f32>) outs(%619 : tensor<1x1x1x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      %861 = arith.minf %860, %arg3 : f32
      linalg.yield %861 : f32
    } -> tensor<1x1x1x288xf32>
    %621 = linalg.init_tensor [1, 1, 1, 288] : tensor<1x1x1x288xf32>
    %622 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%620, %cst_198 : tensor<1x1x1x288xf32>, tensor<1x1x1x288xf32>) outs(%621 : tensor<1x1x1x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x288xf32>
    %623 = linalg.init_tensor [1, 7, 7, 288] : tensor<1x7x7x288xf32>
    %624 = linalg.generic {indexing_maps = [#map5, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%622 : tensor<1x1x1x288xf32>) outs(%623 : tensor<1x7x7x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<1x7x7x288xf32>
    %625 = linalg.init_tensor [1, 7, 7, 288] : tensor<1x7x7x288xf32>
    %626 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%598, %624 : tensor<1x7x7x288xf32>, tensor<1x7x7x288xf32>) outs(%625 : tensor<1x7x7x288xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x288xf32>
    %627 = linalg.init_tensor [1, 7, 7, 96] : tensor<1x7x7x96xf32>
    %cst_329 = arith.constant 0.000000e+00 : f32
    %628 = linalg.fill ins(%cst_329 : f32) outs(%627 : tensor<1x7x7x96xf32>) -> tensor<1x7x7x96xf32>
    %629 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%626, %cst_255 : tensor<1x7x7x288xf32>, tensor<1x1x288x96xf32>) outs(%628 : tensor<1x7x7x96xf32>) -> tensor<1x7x7x96xf32>
    %630 = linalg.init_tensor [1, 7, 7, 96] : tensor<1x7x7x96xf32>
    %631 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%629, %cst_129 : tensor<1x7x7x96xf32>, tensor<1x7x7x96xf32>) outs(%630 : tensor<1x7x7x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.subf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x96xf32>
    %632 = linalg.init_tensor [1, 7, 7, 96] : tensor<1x7x7x96xf32>
    %633 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%631, %cst_127 : tensor<1x7x7x96xf32>, tensor<1x7x7x96xf32>) outs(%632 : tensor<1x7x7x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x96xf32>
    %634 = linalg.init_tensor [1, 7, 7, 96] : tensor<1x7x7x96xf32>
    %635 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%633, %cst_126 : tensor<1x7x7x96xf32>, tensor<1x7x7x96xf32>) outs(%634 : tensor<1x7x7x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x96xf32>
    %636 = linalg.init_tensor [1, 7, 7, 96] : tensor<1x7x7x96xf32>
    %637 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%635, %cst_128 : tensor<1x7x7x96xf32>, tensor<1x7x7x96xf32>) outs(%636 : tensor<1x7x7x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x96xf32>
    %638 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %cst_330 = arith.constant 0.000000e+00 : f32
    %639 = linalg.fill ins(%cst_330 : f32) outs(%638 : tensor<1x7x7x576xf32>) -> tensor<1x7x7x576xf32>
    %640 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%637, %cst_258 : tensor<1x7x7x96xf32>, tensor<1x1x96x576xf32>) outs(%639 : tensor<1x7x7x576xf32>) -> tensor<1x7x7x576xf32>
    %641 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %642 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%640, %cst_133 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%641 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.subf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x576xf32>
    %643 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %644 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%642, %cst_131 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%643 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x576xf32>
    %645 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %646 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%644, %cst_130 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%645 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x576xf32>
    %647 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %648 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%646, %cst_132 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%647 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x576xf32>
    %649 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %650 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%648, %cst_183 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%649 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x576xf32>
    %651 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %652 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %650, %cst_220 : tensor<f32>, tensor<1x7x7x576xf32>, tensor<f32>) outs(%651 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      %861 = arith.minf %860, %arg3 : f32
      linalg.yield %861 : f32
    } -> tensor<1x7x7x576xf32>
    %653 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %654 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%652, %cst_200 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%653 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x576xf32>
    %655 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %656 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%654, %648 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%655 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x576xf32>
    %cst_331 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_332 = arith.constant 0.000000e+00 : f32
    %657 = tensor.pad %656 low[0, 2, 2, 0] high[0, 2, 2, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_332 : f32
    } : tensor<1x7x7x576xf32> to tensor<1x11x11x576xf32>
    %658 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %cst_333 = arith.constant 0.000000e+00 : f32
    %659 = linalg.fill ins(%cst_333 : f32) outs(%658 : tensor<1x7x7x576xf32>) -> tensor<1x7x7x576xf32>
    %660 = tensor.collapse_shape %cst_134 [[0], [1], [2, 3]] : tensor<5x5x1x576xf32> into tensor<5x5x576xf32>
    %661 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%657, %660 : tensor<1x11x11x576xf32>, tensor<5x5x576xf32>) outs(%659 : tensor<1x7x7x576xf32>) -> tensor<1x7x7x576xf32>
    %662 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %663 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%661, %cst_138 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%662 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.subf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x576xf32>
    %664 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %665 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%663, %cst_136 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%664 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x576xf32>
    %666 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %667 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%665, %cst_135 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%666 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x576xf32>
    %668 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %669 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%667, %cst_137 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%668 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x576xf32>
    %670 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %671 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%669, %cst_183 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%670 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x576xf32>
    %672 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %673 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %671, %cst_220 : tensor<f32>, tensor<1x7x7x576xf32>, tensor<f32>) outs(%672 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      %861 = arith.minf %860, %arg3 : f32
      linalg.yield %861 : f32
    } -> tensor<1x7x7x576xf32>
    %674 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %675 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%673, %cst_200 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%674 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x576xf32>
    %676 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %677 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%675, %669 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%676 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x576xf32>
    %cst_334 = arith.constant 0.000000e+00 : f32
    %678 = linalg.init_tensor [1, 576] : tensor<1x576xf32>
    %679 = linalg.fill ins(%cst_334 : f32) outs(%678 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %680 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%677 : tensor<1x7x7x576xf32>) outs(%679 : tensor<1x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x576xf32>
    %681 = linalg.init_tensor [1, 576] : tensor<1x576xf32>
    %682 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%680, %cst_219 : tensor<1x576xf32>, tensor<1x576xf32>) outs(%681 : tensor<1x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x576xf32>
    %683 = tensor.expand_shape %682 [[0], [1, 2, 3]] : tensor<1x576xf32> into tensor<1x1x1x576xf32>
    %684 = linalg.init_tensor [1, 1, 1, 144] : tensor<1x1x1x144xf32>
    %cst_335 = arith.constant 0.000000e+00 : f32
    %685 = linalg.fill ins(%cst_335 : f32) outs(%684 : tensor<1x1x1x144xf32>) -> tensor<1x1x1x144xf32>
    %686 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%683, %cst_261 : tensor<1x1x1x576xf32>, tensor<1x1x576x144xf32>) outs(%685 : tensor<1x1x1x144xf32>) -> tensor<1x1x1x144xf32>
    %687 = linalg.init_tensor [1, 1, 1, 144] : tensor<1x1x1x144xf32>
    %688 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%686, %cst_139 : tensor<1x1x1x144xf32>, tensor<1x1x1x144xf32>) outs(%687 : tensor<1x1x1x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x144xf32>
    %689 = linalg.init_tensor [1, 1, 1, 144] : tensor<1x1x1x144xf32>
    %690 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%688, %cst_218 : tensor<1x1x1x144xf32>, tensor<1x1x1x144xf32>) outs(%689 : tensor<1x1x1x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x144xf32>
    %691 = linalg.init_tensor [1, 1, 1, 576] : tensor<1x1x1x576xf32>
    %cst_336 = arith.constant 0.000000e+00 : f32
    %692 = linalg.fill ins(%cst_336 : f32) outs(%691 : tensor<1x1x1x576xf32>) -> tensor<1x1x1x576xf32>
    %693 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%690, %cst_260 : tensor<1x1x1x144xf32>, tensor<1x1x144x576xf32>) outs(%692 : tensor<1x1x1x576xf32>) -> tensor<1x1x1x576xf32>
    %694 = linalg.init_tensor [1, 1, 1, 576] : tensor<1x1x1x576xf32>
    %695 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%693, %cst_140 : tensor<1x1x1x576xf32>, tensor<1x1x1x576xf32>) outs(%694 : tensor<1x1x1x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x576xf32>
    %696 = linalg.init_tensor [1, 1, 1, 576] : tensor<1x1x1x576xf32>
    %697 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%695, %cst_182 : tensor<1x1x1x576xf32>, tensor<1x1x1x576xf32>) outs(%696 : tensor<1x1x1x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x576xf32>
    %698 = linalg.init_tensor [1, 1, 1, 576] : tensor<1x1x1x576xf32>
    %699 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %697, %cst_220 : tensor<f32>, tensor<1x1x1x576xf32>, tensor<f32>) outs(%698 : tensor<1x1x1x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      %861 = arith.minf %860, %arg3 : f32
      linalg.yield %861 : f32
    } -> tensor<1x1x1x576xf32>
    %700 = linalg.init_tensor [1, 1, 1, 576] : tensor<1x1x1x576xf32>
    %701 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%699, %cst_199 : tensor<1x1x1x576xf32>, tensor<1x1x1x576xf32>) outs(%700 : tensor<1x1x1x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x576xf32>
    %702 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %703 = linalg.generic {indexing_maps = [#map5, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%701 : tensor<1x1x1x576xf32>) outs(%702 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<1x7x7x576xf32>
    %704 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %705 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%677, %703 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%704 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x576xf32>
    %706 = linalg.init_tensor [1, 7, 7, 96] : tensor<1x7x7x96xf32>
    %cst_337 = arith.constant 0.000000e+00 : f32
    %707 = linalg.fill ins(%cst_337 : f32) outs(%706 : tensor<1x7x7x96xf32>) -> tensor<1x7x7x96xf32>
    %708 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%705, %cst_259 : tensor<1x7x7x576xf32>, tensor<1x1x576x96xf32>) outs(%707 : tensor<1x7x7x96xf32>) -> tensor<1x7x7x96xf32>
    %709 = linalg.init_tensor [1, 7, 7, 96] : tensor<1x7x7x96xf32>
    %710 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%708, %cst_144 : tensor<1x7x7x96xf32>, tensor<1x7x7x96xf32>) outs(%709 : tensor<1x7x7x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.subf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x96xf32>
    %711 = linalg.init_tensor [1, 7, 7, 96] : tensor<1x7x7x96xf32>
    %712 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%710, %cst_142 : tensor<1x7x7x96xf32>, tensor<1x7x7x96xf32>) outs(%711 : tensor<1x7x7x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x96xf32>
    %713 = linalg.init_tensor [1, 7, 7, 96] : tensor<1x7x7x96xf32>
    %714 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%712, %cst_141 : tensor<1x7x7x96xf32>, tensor<1x7x7x96xf32>) outs(%713 : tensor<1x7x7x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x96xf32>
    %715 = linalg.init_tensor [1, 7, 7, 96] : tensor<1x7x7x96xf32>
    %716 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%714, %cst_143 : tensor<1x7x7x96xf32>, tensor<1x7x7x96xf32>) outs(%715 : tensor<1x7x7x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x96xf32>
    %717 = linalg.init_tensor [1, 7, 7, 96] : tensor<1x7x7x96xf32>
    %718 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%637, %716 : tensor<1x7x7x96xf32>, tensor<1x7x7x96xf32>) outs(%717 : tensor<1x7x7x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x96xf32>
    %719 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %cst_338 = arith.constant 0.000000e+00 : f32
    %720 = linalg.fill ins(%cst_338 : f32) outs(%719 : tensor<1x7x7x576xf32>) -> tensor<1x7x7x576xf32>
    %721 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%718, %cst_226 : tensor<1x7x7x96xf32>, tensor<1x1x96x576xf32>) outs(%720 : tensor<1x7x7x576xf32>) -> tensor<1x7x7x576xf32>
    %722 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %723 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%721, %cst_148 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%722 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.subf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x576xf32>
    %724 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %725 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%723, %cst_146 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%724 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x576xf32>
    %726 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %727 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%725, %cst_145 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%726 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x576xf32>
    %728 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %729 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%727, %cst_147 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%728 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x576xf32>
    %730 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %731 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%729, %cst_183 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%730 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x576xf32>
    %732 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %733 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %731, %cst_220 : tensor<f32>, tensor<1x7x7x576xf32>, tensor<f32>) outs(%732 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      %861 = arith.minf %860, %arg3 : f32
      linalg.yield %861 : f32
    } -> tensor<1x7x7x576xf32>
    %734 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %735 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%733, %cst_200 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%734 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x576xf32>
    %736 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %737 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%735, %729 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%736 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x576xf32>
    %cst_339 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_340 = arith.constant 0.000000e+00 : f32
    %738 = tensor.pad %737 low[0, 2, 2, 0] high[0, 2, 2, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_340 : f32
    } : tensor<1x7x7x576xf32> to tensor<1x11x11x576xf32>
    %739 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %cst_341 = arith.constant 0.000000e+00 : f32
    %740 = linalg.fill ins(%cst_341 : f32) outs(%739 : tensor<1x7x7x576xf32>) -> tensor<1x7x7x576xf32>
    %741 = tensor.collapse_shape %cst_149 [[0], [1], [2, 3]] : tensor<5x5x1x576xf32> into tensor<5x5x576xf32>
    %742 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%738, %741 : tensor<1x11x11x576xf32>, tensor<5x5x576xf32>) outs(%740 : tensor<1x7x7x576xf32>) -> tensor<1x7x7x576xf32>
    %743 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %744 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%742, %cst_153 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%743 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.subf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x576xf32>
    %745 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %746 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%744, %cst_151 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%745 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x576xf32>
    %747 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %748 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%746, %cst_150 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%747 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x576xf32>
    %749 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %750 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%748, %cst_152 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%749 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x576xf32>
    %751 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %752 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%750, %cst_183 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%751 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x576xf32>
    %753 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %754 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %752, %cst_220 : tensor<f32>, tensor<1x7x7x576xf32>, tensor<f32>) outs(%753 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      %861 = arith.minf %860, %arg3 : f32
      linalg.yield %861 : f32
    } -> tensor<1x7x7x576xf32>
    %755 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %756 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%754, %cst_200 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%755 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x576xf32>
    %757 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %758 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%756, %750 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%757 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x576xf32>
    %cst_342 = arith.constant 0.000000e+00 : f32
    %759 = linalg.init_tensor [1, 576] : tensor<1x576xf32>
    %760 = linalg.fill ins(%cst_342 : f32) outs(%759 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %761 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%758 : tensor<1x7x7x576xf32>) outs(%760 : tensor<1x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x576xf32>
    %762 = linalg.init_tensor [1, 576] : tensor<1x576xf32>
    %763 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%761, %cst_219 : tensor<1x576xf32>, tensor<1x576xf32>) outs(%762 : tensor<1x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x576xf32>
    %764 = tensor.expand_shape %763 [[0], [1, 2, 3]] : tensor<1x576xf32> into tensor<1x1x1x576xf32>
    %765 = linalg.init_tensor [1, 1, 1, 144] : tensor<1x1x1x144xf32>
    %cst_343 = arith.constant 0.000000e+00 : f32
    %766 = linalg.fill ins(%cst_343 : f32) outs(%765 : tensor<1x1x1x144xf32>) -> tensor<1x1x1x144xf32>
    %767 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%764, %cst_229 : tensor<1x1x1x576xf32>, tensor<1x1x576x144xf32>) outs(%766 : tensor<1x1x1x144xf32>) -> tensor<1x1x1x144xf32>
    %768 = linalg.init_tensor [1, 1, 1, 144] : tensor<1x1x1x144xf32>
    %769 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%767, %cst_154 : tensor<1x1x1x144xf32>, tensor<1x1x1x144xf32>) outs(%768 : tensor<1x1x1x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x144xf32>
    %770 = linalg.init_tensor [1, 1, 1, 144] : tensor<1x1x1x144xf32>
    %771 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%769, %cst_218 : tensor<1x1x1x144xf32>, tensor<1x1x1x144xf32>) outs(%770 : tensor<1x1x1x144xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x144xf32>
    %772 = linalg.init_tensor [1, 1, 1, 576] : tensor<1x1x1x576xf32>
    %cst_344 = arith.constant 0.000000e+00 : f32
    %773 = linalg.fill ins(%cst_344 : f32) outs(%772 : tensor<1x1x1x576xf32>) -> tensor<1x1x1x576xf32>
    %774 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%771, %cst_228 : tensor<1x1x1x144xf32>, tensor<1x1x144x576xf32>) outs(%773 : tensor<1x1x1x576xf32>) -> tensor<1x1x1x576xf32>
    %775 = linalg.init_tensor [1, 1, 1, 576] : tensor<1x1x1x576xf32>
    %776 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%774, %cst_155 : tensor<1x1x1x576xf32>, tensor<1x1x1x576xf32>) outs(%775 : tensor<1x1x1x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x576xf32>
    %777 = linalg.init_tensor [1, 1, 1, 576] : tensor<1x1x1x576xf32>
    %778 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%776, %cst_182 : tensor<1x1x1x576xf32>, tensor<1x1x1x576xf32>) outs(%777 : tensor<1x1x1x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x576xf32>
    %779 = linalg.init_tensor [1, 1, 1, 576] : tensor<1x1x1x576xf32>
    %780 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %778, %cst_220 : tensor<f32>, tensor<1x1x1x576xf32>, tensor<f32>) outs(%779 : tensor<1x1x1x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      %861 = arith.minf %860, %arg3 : f32
      linalg.yield %861 : f32
    } -> tensor<1x1x1x576xf32>
    %781 = linalg.init_tensor [1, 1, 1, 576] : tensor<1x1x1x576xf32>
    %782 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%780, %cst_199 : tensor<1x1x1x576xf32>, tensor<1x1x1x576xf32>) outs(%781 : tensor<1x1x1x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x576xf32>
    %783 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %784 = linalg.generic {indexing_maps = [#map5, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%782 : tensor<1x1x1x576xf32>) outs(%783 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<1x7x7x576xf32>
    %785 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %786 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%758, %784 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%785 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x576xf32>
    %787 = linalg.init_tensor [1, 7, 7, 96] : tensor<1x7x7x96xf32>
    %cst_345 = arith.constant 0.000000e+00 : f32
    %788 = linalg.fill ins(%cst_345 : f32) outs(%787 : tensor<1x7x7x96xf32>) -> tensor<1x7x7x96xf32>
    %789 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%786, %cst_227 : tensor<1x7x7x576xf32>, tensor<1x1x576x96xf32>) outs(%788 : tensor<1x7x7x96xf32>) -> tensor<1x7x7x96xf32>
    %790 = linalg.init_tensor [1, 7, 7, 96] : tensor<1x7x7x96xf32>
    %791 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%789, %cst_159 : tensor<1x7x7x96xf32>, tensor<1x7x7x96xf32>) outs(%790 : tensor<1x7x7x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.subf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x96xf32>
    %792 = linalg.init_tensor [1, 7, 7, 96] : tensor<1x7x7x96xf32>
    %793 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%791, %cst_157 : tensor<1x7x7x96xf32>, tensor<1x7x7x96xf32>) outs(%792 : tensor<1x7x7x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x96xf32>
    %794 = linalg.init_tensor [1, 7, 7, 96] : tensor<1x7x7x96xf32>
    %795 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%793, %cst_156 : tensor<1x7x7x96xf32>, tensor<1x7x7x96xf32>) outs(%794 : tensor<1x7x7x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x96xf32>
    %796 = linalg.init_tensor [1, 7, 7, 96] : tensor<1x7x7x96xf32>
    %797 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%795, %cst_158 : tensor<1x7x7x96xf32>, tensor<1x7x7x96xf32>) outs(%796 : tensor<1x7x7x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x96xf32>
    %798 = linalg.init_tensor [1, 7, 7, 96] : tensor<1x7x7x96xf32>
    %799 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%718, %797 : tensor<1x7x7x96xf32>, tensor<1x7x7x96xf32>) outs(%798 : tensor<1x7x7x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x96xf32>
    %800 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %cst_346 = arith.constant 0.000000e+00 : f32
    %801 = linalg.fill ins(%cst_346 : f32) outs(%800 : tensor<1x7x7x576xf32>) -> tensor<1x7x7x576xf32>
    %802 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%799, %cst_223 : tensor<1x7x7x96xf32>, tensor<1x1x96x576xf32>) outs(%801 : tensor<1x7x7x576xf32>) -> tensor<1x7x7x576xf32>
    %803 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %804 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%802, %cst_163 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%803 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.subf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x576xf32>
    %805 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %806 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%804, %cst_161 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%805 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x576xf32>
    %807 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %808 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%806, %cst_160 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%807 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x576xf32>
    %809 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %810 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%808, %cst_162 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%809 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x576xf32>
    %811 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %812 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%810, %cst_183 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%811 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x576xf32>
    %813 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %814 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %812, %cst_220 : tensor<f32>, tensor<1x7x7x576xf32>, tensor<f32>) outs(%813 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      %861 = arith.minf %860, %arg3 : f32
      linalg.yield %861 : f32
    } -> tensor<1x7x7x576xf32>
    %815 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %816 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%814, %cst_200 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%815 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x576xf32>
    %817 = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
    %818 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%816, %810 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%817 : tensor<1x7x7x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x7x7x576xf32>
    %cst_347 = arith.constant 0.000000e+00 : f32
    %819 = linalg.init_tensor [1, 576] : tensor<1x576xf32>
    %820 = linalg.fill ins(%cst_347 : f32) outs(%819 : tensor<1x576xf32>) -> tensor<1x576xf32>
    %821 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%818 : tensor<1x7x7x576xf32>) outs(%820 : tensor<1x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x576xf32>
    %822 = linalg.init_tensor [1, 576] : tensor<1x576xf32>
    %823 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%821, %cst_219 : tensor<1x576xf32>, tensor<1x576xf32>) outs(%822 : tensor<1x576xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x576xf32>
    %824 = tensor.expand_shape %823 [[0], [1, 2, 3]] : tensor<1x576xf32> into tensor<1x1x1x576xf32>
    %825 = linalg.init_tensor [1, 1, 1, 1024] : tensor<1x1x1x1024xf32>
    %cst_348 = arith.constant 0.000000e+00 : f32
    %826 = linalg.fill ins(%cst_348 : f32) outs(%825 : tensor<1x1x1x1024xf32>) -> tensor<1x1x1x1024xf32>
    %827 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%824, %cst_224 : tensor<1x1x1x576xf32>, tensor<1x1x576x1024xf32>) outs(%826 : tensor<1x1x1x1024xf32>) -> tensor<1x1x1x1024xf32>
    %828 = linalg.init_tensor [1, 1, 1, 1024] : tensor<1x1x1x1024xf32>
    %829 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%827, %cst_164 : tensor<1x1x1x1024xf32>, tensor<1x1x1x1024xf32>) outs(%828 : tensor<1x1x1x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x1024xf32>
    %830 = linalg.init_tensor [1, 1, 1, 1024] : tensor<1x1x1x1024xf32>
    %831 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%829, %cst_184 : tensor<1x1x1x1024xf32>, tensor<1x1x1x1024xf32>) outs(%830 : tensor<1x1x1x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x1024xf32>
    %832 = linalg.init_tensor [1, 1, 1, 1024] : tensor<1x1x1x1024xf32>
    %833 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_222, %831, %cst_220 : tensor<f32>, tensor<1x1x1x1024xf32>, tensor<f32>) outs(%832 : tensor<1x1x1x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      %861 = arith.minf %860, %arg3 : f32
      linalg.yield %861 : f32
    } -> tensor<1x1x1x1024xf32>
    %834 = linalg.init_tensor [1, 1, 1, 1024] : tensor<1x1x1x1024xf32>
    %835 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%833, %cst_201 : tensor<1x1x1x1024xf32>, tensor<1x1x1x1024xf32>) outs(%834 : tensor<1x1x1x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x1024xf32>
    %836 = linalg.init_tensor [1, 1, 1, 1024] : tensor<1x1x1x1024xf32>
    %837 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%835, %829 : tensor<1x1x1x1024xf32>, tensor<1x1x1x1024xf32>) outs(%836 : tensor<1x1x1x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x1024xf32>
    %838 = linalg.init_tensor [1, 1, 1, 1000] : tensor<1x1x1x1000xf32>
    %cst_349 = arith.constant 0.000000e+00 : f32
    %839 = linalg.fill ins(%cst_349 : f32) outs(%838 : tensor<1x1x1x1000xf32>) -> tensor<1x1x1x1000xf32>
    %840 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%837, %cst_265 : tensor<1x1x1x1024xf32>, tensor<1x1x1024x1000xf32>) outs(%839 : tensor<1x1x1x1000xf32>) -> tensor<1x1x1x1000xf32>
    %841 = linalg.init_tensor [1, 1, 1, 1000] : tensor<1x1x1x1000xf32>
    %842 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%840, %cst_165 : tensor<1x1x1x1000xf32>, tensor<1x1x1x1000xf32>) outs(%841 : tensor<1x1x1x1000xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1x1x1000xf32>
    %843 = tensor.collapse_shape %842 [[0], [1, 2, 3]] : tensor<1x1x1x1000xf32> into tensor<1x1000xf32>
    %cst_350 = arith.constant 0xFF800000 : f32
    %844 = linalg.init_tensor [1] : tensor<1xf32>
    %845 = linalg.fill ins(%cst_350 : f32) outs(%844 : tensor<1xf32>) -> tensor<1xf32>
    %846 = linalg.generic {indexing_maps = [#map4, #map6], iterator_types = ["parallel", "reduction"]} ins(%843 : tensor<1x1000xf32>) outs(%845 : tensor<1xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %860 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1xf32>
    %847 = linalg.init_tensor [1, 1000] : tensor<1x1000xf32>
    %848 = linalg.generic {indexing_maps = [#map6, #map4], iterator_types = ["parallel", "parallel"]} ins(%846 : tensor<1xf32>) outs(%847 : tensor<1x1000xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<1x1000xf32>
    %849 = linalg.init_tensor [1, 1000] : tensor<1x1000xf32>
    %850 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%843, %848 : tensor<1x1000xf32>, tensor<1x1000xf32>) outs(%849 : tensor<1x1000xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.subf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1000xf32>
    %851 = linalg.init_tensor [1, 1000] : tensor<1x1000xf32>
    %852 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%850 : tensor<1x1000xf32>) outs(%851 : tensor<1x1000xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %860 = math.exp %arg1 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1000xf32>
    %cst_351 = arith.constant 0.000000e+00 : f32
    %853 = linalg.init_tensor [1] : tensor<1xf32>
    %854 = linalg.fill ins(%cst_351 : f32) outs(%853 : tensor<1xf32>) -> tensor<1xf32>
    %855 = linalg.generic {indexing_maps = [#map4, #map6], iterator_types = ["parallel", "reduction"]} ins(%852 : tensor<1x1000xf32>) outs(%854 : tensor<1xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %860 = arith.addf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1xf32>
    %856 = linalg.init_tensor [1, 1000] : tensor<1x1000xf32>
    %857 = linalg.generic {indexing_maps = [#map6, #map4], iterator_types = ["parallel", "parallel"]} ins(%855 : tensor<1xf32>) outs(%856 : tensor<1x1000xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<1x1000xf32>
    %858 = linalg.init_tensor [1, 1000] : tensor<1x1000xf32>
    %859 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%852, %857 : tensor<1x1000xf32>, tensor<1x1000xf32>) outs(%858 : tensor<1x1000xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %860 = arith.divf %arg1, %arg2 : f32
      linalg.yield %860 : f32
    } -> tensor<1x1000xf32>
    return %859 : tensor<1x1000xf32>
  }
}

