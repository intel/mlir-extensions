#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d0)>
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
    %cst = arith.constant dense<5.000000e-01> : tensor<1x112x112x64xf32>
    %cst_0 = arith.constant dense<0.408260554> : tensor<1x112x112x64xf32>
    %cst_1 = arith.constant dense<0.333333343> : tensor<1x112x112x64xf32>
    %cst_2 = arith.constant dense<2.500000e-01> : tensor<1x112x112x64xf32>
    %cst_3 = arith.constant dense<2.000000e-01> : tensor<1x112x112x64xf32>
    %cst_4 = arith.constant dense<5.000000e-02> : tensor<1x56x56x256xf32>
    %cst_5 = arith.constant dense<0.19614166> : tensor<1x56x56x256xf32>
    %cst_6 = arith.constant dense<0.0434782617> : tensor<1x56x56x256xf32>
    %cst_7 = arith.constant dense<0.0416666679> : tensor<1x56x56x256xf32>
    %cst_8 = arith.constant dense<4.000000e-02> : tensor<1x56x56x256xf32>
    %cst_9 = arith.constant dense<1.250000e-01> : tensor<1x56x56x64xf32>
    %cst_10 = arith.constant dense<0.288692474> : tensor<1x56x56x64xf32>
    %cst_11 = arith.constant dense<0.111111112> : tensor<1x56x56x64xf32>
    %cst_12 = arith.constant dense<1.000000e-01> : tensor<1x56x56x64xf32>
    %cst_13 = arith.constant dense<0.0909090936> : tensor<1x56x56x64xf32>
    %cst_14 = arith.constant dense<0.0714285746> : tensor<1x56x56x64xf32>
    %cst_15 = arith.constant dense<0.235723495> : tensor<1x56x56x64xf32>
    %cst_16 = arith.constant dense<0.0666666701> : tensor<1x56x56x64xf32>
    %cst_17 = arith.constant dense<6.250000e-02> : tensor<1x56x56x64xf32>
    %cst_18 = arith.constant dense<0.0588235296> : tensor<1x56x56x64xf32>
    %cst_19 = arith.constant dense<0.0454545468> : tensor<1x56x56x256xf32>
    %cst_20 = arith.constant dense<0.182601601> : tensor<1x56x56x256xf32>
    %cst_21 = arith.constant dense<0.0370370373> : tensor<1x56x56x256xf32>
    %cst_22 = arith.constant dense<0.0357142873> : tensor<1x56x56x256xf32>
    %cst_23 = arith.constant dense<0.0344827585> : tensor<1x56x56x256xf32>
    %cst_24 = arith.constant dense<3.125000e-02> : tensor<1x56x56x64xf32>
    %cst_25 = arith.constant dense<0.166696697> : tensor<1x56x56x64xf32>
    %cst_26 = arith.constant dense<0.0303030312> : tensor<1x56x56x64xf32>
    %cst_27 = arith.constant dense<0.0294117648> : tensor<1x56x56x64xf32>
    %cst_28 = arith.constant dense<0.0285714287> : tensor<1x56x56x64xf32>
    %cst_29 = arith.constant dense<0.0263157897> : tensor<1x56x56x64xf32>
    %cst_30 = arith.constant dense<0.154335782> : tensor<1x56x56x64xf32>
    %cst_31 = arith.constant dense<0.025641026> : tensor<1x56x56x64xf32>
    %cst_32 = arith.constant dense<2.500000e-02> : tensor<1x56x56x64xf32>
    %cst_33 = arith.constant dense<0.024390243> : tensor<1x56x56x64xf32>
    %cst_34 = arith.constant dense<0.0227272734> : tensor<1x56x56x256xf32>
    %cst_35 = arith.constant dense<0.14437224> : tensor<1x56x56x256xf32>
    %cst_36 = arith.constant dense<0.0222222228> : tensor<1x56x56x256xf32>
    %cst_37 = arith.constant dense<0.0217391308> : tensor<1x56x56x256xf32>
    %cst_38 = arith.constant dense<0.0212765951> : tensor<1x56x56x256xf32>
    %cst_39 = arith.constant dense<2.000000e-02> : tensor<1x56x56x64xf32>
    %cst_40 = arith.constant dense<0.136119545> : tensor<1x56x56x64xf32>
    %cst_41 = arith.constant dense<0.0196078438> : tensor<1x56x56x64xf32>
    %cst_42 = arith.constant dense<0.0192307699> : tensor<1x56x56x64xf32>
    %cst_43 = arith.constant dense<0.0188679248> : tensor<1x56x56x64xf32>
    %cst_44 = arith.constant dense<0.0178571437> : tensor<1x56x56x64xf32>
    %cst_45 = arith.constant dense<0.129138216> : tensor<1x56x56x64xf32>
    %cst_46 = arith.constant dense<0.0175438598> : tensor<1x56x56x64xf32>
    %cst_47 = arith.constant dense<0.0172413792> : tensor<1x56x56x64xf32>
    %cst_48 = arith.constant dense<0.0169491526> : tensor<1x56x56x64xf32>
    %cst_49 = arith.constant dense<0.0161290318> : tensor<1x56x56x256xf32>
    %cst_50 = arith.constant dense<0.123132147> : tensor<1x56x56x256xf32>
    %cst_51 = arith.constant dense<0.0158730168> : tensor<1x56x56x256xf32>
    %cst_52 = arith.constant dense<1.562500e-02> : tensor<1x56x56x256xf32>
    %cst_53 = arith.constant dense<0.0153846154> : tensor<1x56x56x256xf32>
    %cst_54 = arith.constant dense<1.250000e-02> : tensor<1x28x28x512xf32>
    %cst_55 = arith.constant dense<0.107879177> : tensor<1x28x28x512xf32>
    %cst_56 = arith.constant dense<0.0120481923> : tensor<1x28x28x512xf32>
    %cst_57 = arith.constant dense<0.0119047621> : tensor<1x28x28x512xf32>
    %cst_58 = arith.constant dense<0.0117647061> : tensor<1x28x28x512xf32>
    %cst_59 = arith.constant dense<0.0147058824> : tensor<1x28x28x128xf32>
    %cst_60 = arith.constant dense<0.117893592> : tensor<1x28x28x128xf32>
    %cst_61 = arith.constant dense<0.0144927539> : tensor<1x28x28x128xf32>
    %cst_62 = arith.constant dense<0.0142857144> : tensor<1x28x28x128xf32>
    %cst_63 = arith.constant dense<0.0140845068> : tensor<1x28x28x128xf32>
    %cst_64 = arith.constant dense<0.0135135138> : tensor<1x28x28x128xf32>
    %cst_65 = arith.constant dense<0.1132719> : tensor<1x28x28x128xf32>
    %cst_66 = arith.constant dense<0.0133333337> : tensor<1x28x28x128xf32>
    %cst_67 = arith.constant dense<0.0131578948> : tensor<1x28x28x128xf32>
    %cst_68 = arith.constant dense<0.012987013> : tensor<1x28x28x128xf32>
    %cst_69 = arith.constant dense<0.0121951215> : tensor<1x28x28x512xf32>
    %cst_70 = arith.constant dense<0.105456725> : tensor<1x28x28x512xf32>
    %cst_71 = arith.constant dense<0.0114942528> : tensor<1x28x28x512xf32>
    %cst_72 = arith.constant dense<0.0113636367> : tensor<1x28x28x512xf32>
    %cst_73 = arith.constant dense<0.0112359552> : tensor<1x28x28x512xf32>
    %cst_74 = arith.constant dense<0.0108695654> : tensor<1x28x28x128xf32>
    %cst_75 = arith.constant dense<0.102111101> : tensor<1x28x28x128xf32>
    %cst_76 = arith.constant dense<0.0107526882> : tensor<1x28x28x128xf32>
    %cst_77 = arith.constant dense<0.0106382975> : tensor<1x28x28x128xf32>
    %cst_78 = arith.constant dense<0.0105263162> : tensor<1x28x28x128xf32>
    %cst_79 = arith.constant dense<0.0102040814> : tensor<1x28x28x128xf32>
    %cst_80 = arith.constant dense<0.0990652889> : tensor<1x28x28x128xf32>
    %cst_81 = arith.constant dense<0.0101010101> : tensor<1x28x28x128xf32>
    %cst_82 = arith.constant dense<0.00999999977> : tensor<1x28x28x128xf32>
    %cst_83 = arith.constant dense<9.900990e-03> : tensor<1x28x28x128xf32>
    %cst_84 = arith.constant dense<0.00961538497> : tensor<1x28x28x512xf32>
    %cst_85 = arith.constant dense<0.0962770432> : tensor<1x28x28x512xf32>
    %cst_86 = arith.constant dense<9.523810e-03> : tensor<1x28x28x512xf32>
    %cst_87 = arith.constant dense<0.0094339624> : tensor<1x28x28x512xf32>
    %cst_88 = arith.constant dense<0.00934579409> : tensor<1x28x28x512xf32>
    %cst_89 = arith.constant dense<0.0090909088> : tensor<1x28x28x128xf32>
    %cst_90 = arith.constant dense<9.371200e-02> : tensor<1x28x28x128xf32>
    %cst_91 = arith.constant dense<0.00900900922> : tensor<1x28x28x128xf32>
    %cst_92 = arith.constant dense<0.00892857183> : tensor<1x28x28x128xf32>
    %cst_93 = arith.constant dense<0.00884955748> : tensor<1x28x28x128xf32>
    %cst_94 = arith.constant dense<8.620690e-03> : tensor<1x28x28x128xf32>
    %cst_95 = arith.constant dense<0.0913419052> : tensor<1x28x28x128xf32>
    %cst_96 = arith.constant dense<0.00854700897> : tensor<1x28x28x128xf32>
    %cst_97 = arith.constant dense<0.00847457629> : tensor<1x28x28x128xf32>
    %cst_98 = arith.constant dense<0.00840336177> : tensor<1x28x28x128xf32>
    %cst_99 = arith.constant dense<0.00819672085> : tensor<1x28x28x512xf32>
    %cst_100 = arith.constant dense<0.0891432464> : tensor<1x28x28x512xf32>
    %cst_101 = arith.constant dense<0.008130081> : tensor<1x28x28x512xf32>
    %cst_102 = arith.constant dense<0.00806451589> : tensor<1x28x28x512xf32>
    %cst_103 = arith.constant dense<8.000000e-03> : tensor<1x28x28x512xf32>
    %cst_104 = arith.constant dense<7.812500e-03> : tensor<1x28x28x128xf32>
    %cst_105 = arith.constant dense<0.0870963111> : tensor<1x28x28x128xf32>
    %cst_106 = arith.constant dense<0.00775193795> : tensor<1x28x28x128xf32>
    %cst_107 = arith.constant dense<0.0076923077> : tensor<1x28x28x128xf32>
    %cst_108 = arith.constant dense<0.00763358781> : tensor<1x28x28x128xf32>
    %cst_109 = arith.constant dense<0.00746268639> : tensor<1x28x28x128xf32>
    %cst_110 = arith.constant dense<0.0851844251> : tensor<1x28x28x128xf32>
    %cst_111 = arith.constant dense<0.00740740728> : tensor<1x28x28x128xf32>
    %cst_112 = arith.constant dense<0.0073529412> : tensor<1x28x28x128xf32>
    %cst_113 = arith.constant dense<7.299270e-03> : tensor<1x28x28x128xf32>
    %cst_114 = arith.constant dense<0.00714285718> : tensor<1x28x28x512xf32>
    %cst_115 = arith.constant dense<0.0833933725> : tensor<1x28x28x512xf32>
    %cst_116 = arith.constant dense<0.00709219835> : tensor<1x28x28x512xf32>
    %cst_117 = arith.constant dense<0.00704225338> : tensor<1x28x28x512xf32>
    %cst_118 = arith.constant dense<0.00699300691> : tensor<1x28x28x512xf32>
    %cst_119 = arith.constant dense<0.00632911408> : tensor<1x14x14x1024xf32>
    %cst_120 = arith.constant dense<0.0781509503> : tensor<1x14x14x1024xf32>
    %cst_121 = arith.constant dense<0.00621118024> : tensor<1x14x14x1024xf32>
    %cst_122 = arith.constant dense<0.00617283955> : tensor<1x14x14x1024xf32>
    %cst_123 = arith.constant dense<0.00613496918> : tensor<1x14x14x1024xf32>
    %cst_124 = arith.constant dense<0.00684931502> : tensor<1x14x14x256xf32>
    %cst_125 = arith.constant dense<0.0817109346> : tensor<1x14x14x256xf32>
    %cst_126 = arith.constant dense<0.00680272094> : tensor<1x14x14x256xf32>
    %cst_127 = arith.constant dense<0.00675675692> : tensor<1x14x14x256xf32>
    %cst_128 = arith.constant dense<0.00671140943> : tensor<1x14x14x256xf32>
    %cst_129 = arith.constant dense<0.00657894742> : tensor<1x14x14x256xf32>
    %cst_130 = arith.constant dense<0.0801265612> : tensor<1x14x14x256xf32>
    %cst_131 = arith.constant dense<0.00653594779> : tensor<1x14x14x256xf32>
    %cst_132 = arith.constant dense<0.00649350649> : tensor<1x14x14x256xf32>
    %cst_133 = arith.constant dense<0.0064516128> : tensor<1x14x14x256xf32>
    %cst_134 = arith.constant dense<6.250000e-03> : tensor<1x14x14x1024xf32>
    %cst_135 = arith.constant dense<0.0772165209> : tensor<1x14x14x1024xf32>
    %cst_136 = arith.constant dense<0.00606060587> : tensor<1x14x14x1024xf32>
    %cst_137 = arith.constant dense<0.00602409616> : tensor<1x14x14x1024xf32>
    %cst_138 = arith.constant dense<0.00598802418> : tensor<1x14x14x1024xf32>
    %cst_139 = arith.constant dense<0.00588235306> : tensor<1x14x14x256xf32>
    %cst_140 = arith.constant dense<7.587580e-02> : tensor<1x14x14x256xf32>
    %cst_141 = arith.constant dense<0.00584795326> : tensor<1x14x14x256xf32>
    %cst_142 = arith.constant dense<0.00581395347> : tensor<1x14x14x256xf32>
    %cst_143 = arith.constant dense<0.00578034669> : tensor<1x14x14x256xf32>
    %cst_144 = arith.constant dense<0.00568181835> : tensor<1x14x14x256xf32>
    %cst_145 = arith.constant dense<0.0746027157> : tensor<1x14x14x256xf32>
    %cst_146 = arith.constant dense<0.00564971752> : tensor<1x14x14x256xf32>
    %cst_147 = arith.constant dense<0.00561797759> : tensor<1x14x14x256xf32>
    %cst_148 = arith.constant dense<0.00558659201> : tensor<1x14x14x256xf32>
    %cst_149 = arith.constant dense<0.00549450563> : tensor<1x14x14x1024xf32>
    %cst_150 = arith.constant dense<0.0733917803> : tensor<1x14x14x1024xf32>
    %cst_151 = arith.constant dense<0.00546448072> : tensor<1x14x14x1024xf32>
    %cst_152 = arith.constant dense<0.00543478271> : tensor<1x14x14x1024xf32>
    %cst_153 = arith.constant dense<0.00540540554> : tensor<1x14x14x1024xf32>
    %cst_154 = arith.constant dense<0.00531914877> : tensor<1x14x14x256xf32>
    %cst_155 = arith.constant dense<7.223810e-02> : tensor<1x14x14x256xf32>
    %cst_156 = arith.constant dense<0.00529100513> : tensor<1x14x14x256xf32>
    %cst_157 = arith.constant dense<0.00526315812> : tensor<1x14x14x256xf32>
    %cst_158 = arith.constant dense<0.00523560215> : tensor<1x14x14x256xf32>
    %cst_159 = arith.constant dense<0.00515463902> : tensor<1x14x14x256xf32>
    %cst_160 = arith.constant dense<0.0711372942> : tensor<1x14x14x256xf32>
    %cst_161 = arith.constant dense<0.00512820529> : tensor<1x14x14x256xf32>
    %cst_162 = arith.constant dense<0.00510204071> : tensor<1x14x14x256xf32>
    %cst_163 = arith.constant dense<0.00507614203> : tensor<1x14x14x256xf32>
    %cst_164 = arith.constant dense<5.000000e-03> : tensor<1x14x14x1024xf32>
    %cst_165 = arith.constant dense<0.070085451> : tensor<1x14x14x1024xf32>
    %cst_166 = arith.constant dense<0.00497512426> : tensor<1x14x14x1024xf32>
    %cst_167 = arith.constant dense<0.00495049497> : tensor<1x14x14x1024xf32>
    %cst_168 = arith.constant dense<0.00492610829> : tensor<1x14x14x1024xf32>
    %cst_169 = arith.constant dense<0.00485436898> : tensor<1x14x14x256xf32>
    %cst_170 = arith.constant dense<0.0690790489> : tensor<1x14x14x256xf32>
    %cst_171 = arith.constant dense<0.00483091781> : tensor<1x14x14x256xf32>
    %cst_172 = arith.constant dense<0.00480769249> : tensor<1x14x14x256xf32>
    %cst_173 = arith.constant dense<0.00478468882> : tensor<1x14x14x256xf32>
    %cst_174 = arith.constant dense<0.0047169812> : tensor<1x14x14x256xf32>
    %cst_175 = arith.constant dense<6.811490e-02> : tensor<1x14x14x256xf32>
    %cst_176 = arith.constant dense<0.00469483575> : tensor<1x14x14x256xf32>
    %cst_177 = arith.constant dense<0.00467289705> : tensor<1x14x14x256xf32>
    %cst_178 = arith.constant dense<0.00465116277> : tensor<1x14x14x256xf32>
    %cst_179 = arith.constant dense<0.00458715577> : tensor<1x14x14x1024xf32>
    %cst_180 = arith.constant dense<0.067190133> : tensor<1x14x14x1024xf32>
    %cst_181 = arith.constant dense<4.566210e-03> : tensor<1x14x14x1024xf32>
    %cst_182 = arith.constant dense<0.0045454544> : tensor<1x14x14x1024xf32>
    %cst_183 = arith.constant dense<0.00452488707> : tensor<1x14x14x1024xf32>
    %cst_184 = arith.constant dense<0.00446428591> : tensor<1x14x14x256xf32>
    %cst_185 = arith.constant dense<0.0663021504> : tensor<1x14x14x256xf32>
    %cst_186 = arith.constant dense<0.00444444455> : tensor<1x14x14x256xf32>
    %cst_187 = arith.constant dense<0.00442477874> : tensor<1x14x14x256xf32>
    %cst_188 = arith.constant dense<0.00440528616> : tensor<1x14x14x256xf32>
    %cst_189 = arith.constant dense<0.00434782589> : tensor<1x14x14x256xf32>
    %cst_190 = arith.constant dense<0.0654485598> : tensor<1x14x14x256xf32>
    %cst_191 = arith.constant dense<0.00432900432> : tensor<1x14x14x256xf32>
    %cst_192 = arith.constant dense<0.00431034481> : tensor<1x14x14x256xf32>
    %cst_193 = arith.constant dense<0.00429184549> : tensor<1x14x14x256xf32>
    %cst_194 = arith.constant dense<0.00423728814> : tensor<1x14x14x1024xf32>
    %cst_195 = arith.constant dense<0.0646272153> : tensor<1x14x14x1024xf32>
    %cst_196 = arith.constant dense<0.00421940908> : tensor<1x14x14x1024xf32>
    %cst_197 = arith.constant dense<0.00420168089> : tensor<1x14x14x1024xf32>
    %cst_198 = arith.constant dense<0.00418410031> : tensor<1x14x14x1024xf32>
    %cst_199 = arith.constant dense<0.00413223123> : tensor<1x14x14x256xf32>
    %cst_200 = arith.constant dense<0.0638361201> : tensor<1x14x14x256xf32>
    %cst_201 = arith.constant dense<0.00411522621> : tensor<1x14x14x256xf32>
    %cst_202 = arith.constant dense<0.00409836043> : tensor<1x14x14x256xf32>
    %cst_203 = arith.constant dense<0.00408163248> : tensor<1x14x14x256xf32>
    %cst_204 = arith.constant dense<0.00403225794> : tensor<1x14x14x256xf32>
    %cst_205 = arith.constant dense<0.0630734786> : tensor<1x14x14x256xf32>
    %cst_206 = arith.constant dense<0.00401606411> : tensor<1x14x14x256xf32>
    %cst_207 = arith.constant dense<4.000000e-03> : tensor<1x14x14x256xf32>
    %cst_208 = arith.constant dense<0.00398406386> : tensor<1x14x14x256xf32>
    %cst_209 = arith.constant dense<0.00393700786> : tensor<1x14x14x1024xf32>
    %cst_210 = arith.constant dense<0.062337622> : tensor<1x14x14x1024xf32>
    %cst_211 = arith.constant dense<0.00392156886> : tensor<1x14x14x1024xf32>
    %cst_212 = arith.constant dense<3.906250e-03> : tensor<1x14x14x1024xf32>
    %cst_213 = arith.constant dense<0.00389105058> : tensor<1x14x14x1024xf32>
    %cst_214 = arith.constant dense<0.0036764706> : tensor<1x7x7x2048xf32>
    %cst_215 = arith.constant dense<0.0600594096> : tensor<1x7x7x2048xf32>
    %cst_216 = arith.constant dense<0.00363636366> : tensor<1x7x7x2048xf32>
    %cst_217 = arith.constant dense<0.00362318847> : tensor<1x7x7x2048xf32>
    %cst_218 = arith.constant dense<0.00361010828> : tensor<1x7x7x2048xf32>
    %cst_219 = arith.constant dense<0.00384615385> : tensor<1x7x7x512xf32>
    %cst_220 = arith.constant dense<0.0616270155> : tensor<1x7x7x512xf32>
    %cst_221 = arith.constant dense<0.00383141753> : tensor<1x7x7x512xf32>
    %cst_222 = arith.constant dense<0.00381679391> : tensor<1x7x7x512xf32>
    %cst_223 = arith.constant dense<0.00380228134> : tensor<1x7x7x512xf32>
    %cst_224 = arith.constant dense<0.00375939859> : tensor<1x7x7x512xf32>
    %cst_225 = arith.constant dense<0.060940247> : tensor<1x7x7x512xf32>
    %cst_226 = arith.constant dense<0.00374531839> : tensor<1x7x7x512xf32>
    %cst_227 = arith.constant dense<0.0037313432> : tensor<1x7x7x512xf32>
    %cst_228 = arith.constant dense<0.00371747208> : tensor<1x7x7x512xf32>
    %cst_229 = arith.constant dense<0.00364963501> : tensor<1x7x7x2048xf32>
    %cst_230 = arith.constant dense<0.0596331209> : tensor<1x7x7x2048xf32>
    %cst_231 = arith.constant dense<0.00358422939> : tensor<1x7x7x2048xf32>
    %cst_232 = arith.constant dense<0.00357142859> : tensor<1x7x7x2048xf32>
    %cst_233 = arith.constant dense<0.00355871883> : tensor<1x7x7x2048xf32>
    %cst_234 = arith.constant dense<0.00352112669> : tensor<1x7x7x512xf32>
    %cst_235 = arith.constant dense<0.0590104423> : tensor<1x7x7x512xf32>
    %cst_236 = arith.constant dense<0.003508772> : tensor<1x7x7x512xf32>
    %cst_237 = arith.constant dense<0.00349650346> : tensor<1x7x7x512xf32>
    %cst_238 = arith.constant dense<0.00348432059> : tensor<1x7x7x512xf32>
    %cst_239 = arith.constant dense<0.00344827585> : tensor<1x7x7x512xf32>
    %cst_240 = arith.constant dense<0.0584069379> : tensor<1x7x7x512xf32>
    %cst_241 = arith.constant dense<0.00343642617> : tensor<1x7x7x512xf32>
    %cst_242 = arith.constant dense<0.00342465751> : tensor<1x7x7x512xf32>
    %cst_243 = arith.constant dense<0.00341296918> : tensor<1x7x7x512xf32>
    %cst_244 = arith.constant dense<0.00337837846> : tensor<1x7x7x2048xf32>
    %cst_245 = arith.constant dense<0.0578216538> : tensor<1x7x7x2048xf32>
    %cst_246 = arith.constant dense<0.00336700329> : tensor<1x7x7x2048xf32>
    %cst_247 = arith.constant dense<0.00335570471> : tensor<1x7x7x2048xf32>
    %cst_248 = arith.constant dense<0.00334448158> : tensor<1x7x7x2048xf32>
    %cst_249 = arith.constant dense<0.00331125828> : tensor<1x7x7x512xf32>
    %cst_250 = arith.constant dense<0.0572536811> : tensor<1x7x7x512xf32>
    %cst_251 = arith.constant dense<0.00330033014> : tensor<1x7x7x512xf32>
    %cst_252 = arith.constant dense<0.00328947371> : tensor<1x7x7x512xf32>
    %cst_253 = arith.constant dense<0.00327868853> : tensor<1x7x7x512xf32>
    %cst_254 = arith.constant dense<0.00324675324> : tensor<1x7x7x512xf32>
    %cst_255 = arith.constant dense<0.0567021891> : tensor<1x7x7x512xf32>
    %cst_256 = arith.constant dense<0.00323624606> : tensor<1x7x7x512xf32>
    %cst_257 = arith.constant dense<0.0032258064> : tensor<1x7x7x512xf32>
    %cst_258 = arith.constant dense<0.00321543403> : tensor<1x7x7x512xf32>
    %cst_259 = arith.constant dense<0.00318471342> : tensor<1x7x7x2048xf32>
    %cst_260 = arith.constant dense<0.0561663955> : tensor<1x7x7x2048xf32>
    %cst_261 = arith.constant dense<0.00317460322> : tensor<1x7x7x2048xf32>
    %cst_262 = arith.constant dense<0.00316455704> : tensor<1x7x7x2048xf32>
    %cst_263 = arith.constant dense<0.00315457419> : tensor<1x7x7x2048xf32>
    %cst_264 = arith.constant dense<3.125000e-03> : tensor<1x1000xf32>
    %cst_265 = arith.constant dense<0.000000e+00> : tensor<1x112x112x64xf32>
    %cst_266 = arith.constant dense<0.000000e+00> : tensor<1x56x56x64xf32>
    %cst_267 = arith.constant dense<0.000000e+00> : tensor<1x56x56x256xf32>
    %cst_268 = arith.constant dense<0.000000e+00> : tensor<1x28x28x128xf32>
    %cst_269 = arith.constant dense<0.000000e+00> : tensor<1x28x28x512xf32>
    %cst_270 = arith.constant dense<0.000000e+00> : tensor<1x14x14x256xf32>
    %cst_271 = arith.constant dense<0.000000e+00> : tensor<1x14x14x1024xf32>
    %cst_272 = arith.constant dense<0.000000e+00> : tensor<1x7x7x512xf32>
    %cst_273 = arith.constant dense<0.000000e+00> : tensor<1x7x7x2048xf32>
    %cst_274 = arith.constant dense<4.900000e+01> : tensor<1x2048xf32>
    %cst_275 = arith.constant dense<0xFF800000> : tensor<f32>
    %cst_276 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_277 = arith.constant dense<1.000000e+00> : tensor<7x7x3x64xf32>
    %cst_278 = arith.constant dense<0.0526315793> : tensor<1x1x64x256xf32>
    %cst_279 = arith.constant dense<0.142857149> : tensor<1x1x64x64xf32>
    %cst_280 = arith.constant dense<0.0769230798> : tensor<3x3x64x64xf32>
    %cst_281 = arith.constant dense<0.0476190485> : tensor<1x1x64x256xf32>
    %cst_282 = arith.constant dense<0.0322580636> : tensor<1x1x256x64xf32>
    %cst_283 = arith.constant dense<0.0270270277> : tensor<3x3x64x64xf32>
    %cst_284 = arith.constant dense<0.0232558139> : tensor<1x1x64x256xf32>
    %cst_285 = arith.constant dense<0.0204081628> : tensor<1x1x256x64xf32>
    %cst_286 = arith.constant dense<0.0181818176> : tensor<3x3x64x64xf32>
    %cst_287 = arith.constant dense<0.0163934417> : tensor<1x1x64x256xf32>
    %cst_288 = arith.constant dense<0.0126582282> : tensor<1x1x256x512xf32>
    %cst_289 = arith.constant dense<0.0149253728> : tensor<1x1x256x128xf32>
    %cst_290 = arith.constant dense<0.01369863> : tensor<3x3x128x128xf32>
    %cst_291 = arith.constant dense<0.0123456791> : tensor<1x1x128x512xf32>
    %cst_292 = arith.constant dense<0.0109890113> : tensor<1x1x512x128xf32>
    %cst_293 = arith.constant dense<0.010309278> : tensor<3x3x128x128xf32>
    %cst_294 = arith.constant dense<0.00970873795> : tensor<1x1x128x512xf32>
    %cst_295 = arith.constant dense<0.00917431153> : tensor<1x1x512x128xf32>
    %cst_296 = arith.constant dense<0.00869565178> : tensor<3x3x128x128xf32>
    %cst_297 = arith.constant dense<0.00826446246> : tensor<1x1x128x512xf32>
    %cst_298 = arith.constant dense<0.00787401571> : tensor<1x1x512x128xf32>
    %cst_299 = arith.constant dense<0.00751879718> : tensor<3x3x128x128xf32>
    %cst_300 = arith.constant dense<0.00719424477> : tensor<1x1x128x512xf32>
    %cst_301 = arith.constant dense<0.00636942684> : tensor<1x1x512x1024xf32>
    %cst_302 = arith.constant dense<0.0068965517> : tensor<1x1x512x256xf32>
    %cst_303 = arith.constant dense<0.00662251655> : tensor<3x3x256x256xf32>
    %cst_304 = arith.constant dense<0.00628930796> : tensor<1x1x256x1024xf32>
    %cst_305 = arith.constant dense<5.917160e-03> : tensor<1x1x1024x256xf32>
    %cst_306 = arith.constant dense<0.00571428565> : tensor<3x3x256x256xf32>
    %cst_307 = arith.constant dense<0.00552486209> : tensor<1x1x256x1024xf32>
    %cst_308 = arith.constant dense<0.00534759369> : tensor<1x1x1024x256xf32>
    %cst_309 = arith.constant dense<0.00518134702> : tensor<3x3x256x256xf32>
    %cst_310 = arith.constant dense<0.00502512557> : tensor<1x1x256x1024xf32>
    %cst_311 = arith.constant dense<0.00487804879> : tensor<1x1x1024x256xf32>
    %cst_312 = arith.constant dense<0.00473933667> : tensor<3x3x256x256xf32>
    %cst_313 = arith.constant dense<0.00460829493> : tensor<1x1x256x1024xf32>
    %cst_314 = arith.constant dense<0.00448430516> : tensor<1x1x1024x256xf32>
    %cst_315 = arith.constant dense<0.0043668123> : tensor<3x3x256x256xf32>
    %cst_316 = arith.constant dense<0.00425531901> : tensor<1x1x256x1024xf32>
    %cst_317 = arith.constant dense<0.00414937781> : tensor<1x1x1024x256xf32>
    %cst_318 = arith.constant dense<0.0040485831> : tensor<3x3x256x256xf32>
    %cst_319 = arith.constant dense<0.00395256933> : tensor<1x1x256x1024xf32>
    %cst_320 = arith.constant dense<0.00369003695> : tensor<1x1x1024x2048xf32>
    %cst_321 = arith.constant dense<0.00386100379> : tensor<1x1x1024x512xf32>
    %cst_322 = arith.constant dense<0.00377358496> : tensor<3x3x512x512xf32>
    %cst_323 = arith.constant dense<0.00366300368> : tensor<1x1x512x2048xf32>
    %cst_324 = arith.constant dense<0.00353356893> : tensor<1x1x2048x512xf32>
    %cst_325 = arith.constant dense<0.00346020772> : tensor<3x3x512x512xf32>
    %cst_326 = arith.constant dense<0.00338983047> : tensor<1x1x512x2048xf32>
    %cst_327 = arith.constant dense<0.00332225906> : tensor<1x1x2048x512xf32>
    %cst_328 = arith.constant dense<0.00325732888> : tensor<3x3x512x512xf32>
    %cst_329 = arith.constant dense<0.00319488812> : tensor<1x1x512x2048xf32>
    %cst_330 = arith.constant dense<0.00313479616> : tensor<2048x1000xf32>
    %cst_331 = arith.constant 0.000000e+00 : f32
    %0 = linalg.init_tensor [1, 230, 230, 3] : tensor<1x230x230x3xf32>
    %1 = linalg.fill ins(%cst_331 : f32) outs(%0 : tensor<1x230x230x3xf32>) -> tensor<1x230x230x3xf32>
    %2 = tensor.insert_slice %arg0 into %1[0, 3, 3, 0] [1, 224, 224, 3] [1, 1, 1, 1] : tensor<1x224x224x3xf32> into tensor<1x230x230x3xf32>
    %3 = linalg.init_tensor [1, 112, 112, 64] : tensor<1x112x112x64xf32>
    %cst_332 = arith.constant 0.000000e+00 : f32
    %4 = linalg.fill ins(%cst_332 : f32) outs(%3 : tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
    %5 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%2, %cst_277 : tensor<1x230x230x3xf32>, tensor<7x7x3x64xf32>) outs(%4 : tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
    %6 = linalg.init_tensor [1, 112, 112, 64] : tensor<1x112x112x64xf32>
    %7 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%5, %cst : tensor<1x112x112x64xf32>, tensor<1x112x112x64xf32>) outs(%6 : tensor<1x112x112x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x112x112x64xf32>
    %8 = linalg.init_tensor [1, 112, 112, 64] : tensor<1x112x112x64xf32>
    %9 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%7, %cst_3 : tensor<1x112x112x64xf32>, tensor<1x112x112x64xf32>) outs(%8 : tensor<1x112x112x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x112x112x64xf32>
    %10 = linalg.init_tensor [1, 112, 112, 64] : tensor<1x112x112x64xf32>
    %11 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%9, %cst_1 : tensor<1x112x112x64xf32>, tensor<1x112x112x64xf32>) outs(%10 : tensor<1x112x112x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x112x112x64xf32>
    %12 = linalg.init_tensor [1, 112, 112, 64] : tensor<1x112x112x64xf32>
    %13 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%11, %cst_0 : tensor<1x112x112x64xf32>, tensor<1x112x112x64xf32>) outs(%12 : tensor<1x112x112x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x112x112x64xf32>
    %14 = linalg.init_tensor [1, 112, 112, 64] : tensor<1x112x112x64xf32>
    %15 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%13, %cst_2 : tensor<1x112x112x64xf32>, tensor<1x112x112x64xf32>) outs(%14 : tensor<1x112x112x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x112x112x64xf32>
    %16 = linalg.init_tensor [1, 112, 112, 64] : tensor<1x112x112x64xf32>
    %17 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%15, %cst_265 : tensor<1x112x112x64xf32>, tensor<1x112x112x64xf32>) outs(%16 : tensor<1x112x112x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x112x112x64xf32>
    %cst_333 = arith.constant 0.000000e+00 : f32
    %18 = linalg.init_tensor [1, 114, 114, 64] : tensor<1x114x114x64xf32>
    %19 = linalg.fill ins(%cst_333 : f32) outs(%18 : tensor<1x114x114x64xf32>) -> tensor<1x114x114x64xf32>
    %20 = tensor.insert_slice %17 into %19[0, 1, 1, 0] [1, 112, 112, 64] [1, 1, 1, 1] : tensor<1x112x112x64xf32> into tensor<1x114x114x64xf32>
    %21 = linalg.init_tensor [3, 3] : tensor<3x3xf32>
    %22 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %23 = tensor.extract %cst_275[] : tensor<f32>
    %24 = linalg.fill ins(%23 : f32) outs(%22 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %25 = linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%20, %21 : tensor<1x114x114x64xf32>, tensor<3x3xf32>) outs(%24 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %26 = linalg.init_tensor [1, 56, 56, 256] : tensor<1x56x56x256xf32>
    %cst_334 = arith.constant 0.000000e+00 : f32
    %27 = linalg.fill ins(%cst_334 : f32) outs(%26 : tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %28 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%25, %cst_278 : tensor<1x56x56x64xf32>, tensor<1x1x64x256xf32>) outs(%27 : tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %29 = linalg.init_tensor [1, 56, 56, 256] : tensor<1x56x56x256xf32>
    %30 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%28, %cst_4 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%29 : tensor<1x56x56x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x256xf32>
    %31 = linalg.init_tensor [1, 56, 56, 256] : tensor<1x56x56x256xf32>
    %32 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%30, %cst_8 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%31 : tensor<1x56x56x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x256xf32>
    %33 = linalg.init_tensor [1, 56, 56, 256] : tensor<1x56x56x256xf32>
    %34 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%32, %cst_6 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%33 : tensor<1x56x56x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x256xf32>
    %35 = linalg.init_tensor [1, 56, 56, 256] : tensor<1x56x56x256xf32>
    %36 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%34, %cst_5 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%35 : tensor<1x56x56x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x256xf32>
    %37 = linalg.init_tensor [1, 56, 56, 256] : tensor<1x56x56x256xf32>
    %38 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%36, %cst_7 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%37 : tensor<1x56x56x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x256xf32>
    %39 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %cst_335 = arith.constant 0.000000e+00 : f32
    %40 = linalg.fill ins(%cst_335 : f32) outs(%39 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %41 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%25, %cst_279 : tensor<1x56x56x64xf32>, tensor<1x1x64x64xf32>) outs(%40 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %42 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %43 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%41, %cst_9 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%42 : tensor<1x56x56x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x64xf32>
    %44 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %45 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%43, %cst_13 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%44 : tensor<1x56x56x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x64xf32>
    %46 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %47 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%45, %cst_11 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%46 : tensor<1x56x56x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x64xf32>
    %48 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %49 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%47, %cst_10 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%48 : tensor<1x56x56x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x64xf32>
    %50 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %51 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%49, %cst_12 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%50 : tensor<1x56x56x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x64xf32>
    %52 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %53 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%51, %cst_266 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%52 : tensor<1x56x56x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x64xf32>
    %54 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %cst_336 = arith.constant 0.000000e+00 : f32
    %55 = linalg.fill ins(%cst_336 : f32) outs(%54 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %cst_337 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_338 = arith.constant 0.000000e+00 : f32
    %56 = linalg.init_tensor [1, 58, 58, 64] : tensor<1x58x58x64xf32>
    %57 = linalg.fill ins(%cst_338 : f32) outs(%56 : tensor<1x58x58x64xf32>) -> tensor<1x58x58x64xf32>
    %58 = tensor.insert_slice %53 into %57[0, 1, 1, 0] [1, 56, 56, 64] [1, 1, 1, 1] : tensor<1x56x56x64xf32> into tensor<1x58x58x64xf32>
    %59 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%58, %cst_280 : tensor<1x58x58x64xf32>, tensor<3x3x64x64xf32>) outs(%55 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %60 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %61 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%59, %cst_14 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%60 : tensor<1x56x56x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x64xf32>
    %62 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %63 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%61, %cst_18 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%62 : tensor<1x56x56x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x64xf32>
    %64 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %65 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%63, %cst_16 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%64 : tensor<1x56x56x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x64xf32>
    %66 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %67 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%65, %cst_15 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%66 : tensor<1x56x56x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x64xf32>
    %68 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %69 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%67, %cst_17 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%68 : tensor<1x56x56x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x64xf32>
    %70 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %71 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%69, %cst_266 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%70 : tensor<1x56x56x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x64xf32>
    %72 = linalg.init_tensor [1, 56, 56, 256] : tensor<1x56x56x256xf32>
    %cst_339 = arith.constant 0.000000e+00 : f32
    %73 = linalg.fill ins(%cst_339 : f32) outs(%72 : tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %74 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%71, %cst_281 : tensor<1x56x56x64xf32>, tensor<1x1x64x256xf32>) outs(%73 : tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %75 = linalg.init_tensor [1, 56, 56, 256] : tensor<1x56x56x256xf32>
    %76 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%74, %cst_19 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%75 : tensor<1x56x56x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x256xf32>
    %77 = linalg.init_tensor [1, 56, 56, 256] : tensor<1x56x56x256xf32>
    %78 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%76, %cst_23 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%77 : tensor<1x56x56x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x256xf32>
    %79 = linalg.init_tensor [1, 56, 56, 256] : tensor<1x56x56x256xf32>
    %80 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%78, %cst_21 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%79 : tensor<1x56x56x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x256xf32>
    %81 = linalg.init_tensor [1, 56, 56, 256] : tensor<1x56x56x256xf32>
    %82 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%80, %cst_20 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%81 : tensor<1x56x56x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x256xf32>
    %83 = linalg.init_tensor [1, 56, 56, 256] : tensor<1x56x56x256xf32>
    %84 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%82, %cst_22 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%83 : tensor<1x56x56x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x256xf32>
    %85 = linalg.init_tensor [1, 56, 56, 256] : tensor<1x56x56x256xf32>
    %86 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%38, %84 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%85 : tensor<1x56x56x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x256xf32>
    %87 = linalg.init_tensor [1, 56, 56, 256] : tensor<1x56x56x256xf32>
    %88 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%86, %cst_267 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%87 : tensor<1x56x56x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x256xf32>
    %89 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %cst_340 = arith.constant 0.000000e+00 : f32
    %90 = linalg.fill ins(%cst_340 : f32) outs(%89 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %91 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%88, %cst_282 : tensor<1x56x56x256xf32>, tensor<1x1x256x64xf32>) outs(%90 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %92 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %93 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%91, %cst_24 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%92 : tensor<1x56x56x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x64xf32>
    %94 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %95 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%93, %cst_28 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%94 : tensor<1x56x56x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x64xf32>
    %96 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %97 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%95, %cst_26 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%96 : tensor<1x56x56x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x64xf32>
    %98 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %99 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%97, %cst_25 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%98 : tensor<1x56x56x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x64xf32>
    %100 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %101 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%99, %cst_27 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%100 : tensor<1x56x56x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x64xf32>
    %102 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %103 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%101, %cst_266 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%102 : tensor<1x56x56x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x64xf32>
    %104 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %cst_341 = arith.constant 0.000000e+00 : f32
    %105 = linalg.fill ins(%cst_341 : f32) outs(%104 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %cst_342 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_343 = arith.constant 0.000000e+00 : f32
    %106 = linalg.init_tensor [1, 58, 58, 64] : tensor<1x58x58x64xf32>
    %107 = linalg.fill ins(%cst_343 : f32) outs(%106 : tensor<1x58x58x64xf32>) -> tensor<1x58x58x64xf32>
    %108 = tensor.insert_slice %103 into %107[0, 1, 1, 0] [1, 56, 56, 64] [1, 1, 1, 1] : tensor<1x56x56x64xf32> into tensor<1x58x58x64xf32>
    %109 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%108, %cst_283 : tensor<1x58x58x64xf32>, tensor<3x3x64x64xf32>) outs(%105 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %110 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %111 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%109, %cst_29 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%110 : tensor<1x56x56x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x64xf32>
    %112 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %113 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%111, %cst_33 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%112 : tensor<1x56x56x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x64xf32>
    %114 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %115 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%113, %cst_31 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%114 : tensor<1x56x56x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x64xf32>
    %116 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %117 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%115, %cst_30 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%116 : tensor<1x56x56x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x64xf32>
    %118 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %119 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%117, %cst_32 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%118 : tensor<1x56x56x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x64xf32>
    %120 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %121 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%119, %cst_266 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%120 : tensor<1x56x56x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x64xf32>
    %122 = linalg.init_tensor [1, 56, 56, 256] : tensor<1x56x56x256xf32>
    %cst_344 = arith.constant 0.000000e+00 : f32
    %123 = linalg.fill ins(%cst_344 : f32) outs(%122 : tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %124 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%121, %cst_284 : tensor<1x56x56x64xf32>, tensor<1x1x64x256xf32>) outs(%123 : tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %125 = linalg.init_tensor [1, 56, 56, 256] : tensor<1x56x56x256xf32>
    %126 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%124, %cst_34 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%125 : tensor<1x56x56x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x256xf32>
    %127 = linalg.init_tensor [1, 56, 56, 256] : tensor<1x56x56x256xf32>
    %128 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%126, %cst_38 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%127 : tensor<1x56x56x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x256xf32>
    %129 = linalg.init_tensor [1, 56, 56, 256] : tensor<1x56x56x256xf32>
    %130 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%128, %cst_36 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%129 : tensor<1x56x56x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x256xf32>
    %131 = linalg.init_tensor [1, 56, 56, 256] : tensor<1x56x56x256xf32>
    %132 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%130, %cst_35 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%131 : tensor<1x56x56x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x256xf32>
    %133 = linalg.init_tensor [1, 56, 56, 256] : tensor<1x56x56x256xf32>
    %134 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%132, %cst_37 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%133 : tensor<1x56x56x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x256xf32>
    %135 = linalg.init_tensor [1, 56, 56, 256] : tensor<1x56x56x256xf32>
    %136 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%88, %134 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%135 : tensor<1x56x56x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x256xf32>
    %137 = linalg.init_tensor [1, 56, 56, 256] : tensor<1x56x56x256xf32>
    %138 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%136, %cst_267 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%137 : tensor<1x56x56x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x256xf32>
    %139 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %cst_345 = arith.constant 0.000000e+00 : f32
    %140 = linalg.fill ins(%cst_345 : f32) outs(%139 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %141 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%138, %cst_285 : tensor<1x56x56x256xf32>, tensor<1x1x256x64xf32>) outs(%140 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %142 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %143 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%141, %cst_39 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%142 : tensor<1x56x56x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x64xf32>
    %144 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %145 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%143, %cst_43 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%144 : tensor<1x56x56x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x64xf32>
    %146 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %147 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%145, %cst_41 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%146 : tensor<1x56x56x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x64xf32>
    %148 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %149 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%147, %cst_40 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%148 : tensor<1x56x56x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x64xf32>
    %150 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %151 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%149, %cst_42 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%150 : tensor<1x56x56x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x64xf32>
    %152 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %153 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%151, %cst_266 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%152 : tensor<1x56x56x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x64xf32>
    %154 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %cst_346 = arith.constant 0.000000e+00 : f32
    %155 = linalg.fill ins(%cst_346 : f32) outs(%154 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %cst_347 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_348 = arith.constant 0.000000e+00 : f32
    %156 = linalg.init_tensor [1, 58, 58, 64] : tensor<1x58x58x64xf32>
    %157 = linalg.fill ins(%cst_348 : f32) outs(%156 : tensor<1x58x58x64xf32>) -> tensor<1x58x58x64xf32>
    %158 = tensor.insert_slice %153 into %157[0, 1, 1, 0] [1, 56, 56, 64] [1, 1, 1, 1] : tensor<1x56x56x64xf32> into tensor<1x58x58x64xf32>
    %159 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%158, %cst_286 : tensor<1x58x58x64xf32>, tensor<3x3x64x64xf32>) outs(%155 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %160 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %161 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%159, %cst_44 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%160 : tensor<1x56x56x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x64xf32>
    %162 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %163 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%161, %cst_48 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%162 : tensor<1x56x56x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x64xf32>
    %164 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %165 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%163, %cst_46 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%164 : tensor<1x56x56x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x64xf32>
    %166 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %167 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%165, %cst_45 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%166 : tensor<1x56x56x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x64xf32>
    %168 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %169 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%167, %cst_47 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%168 : tensor<1x56x56x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x64xf32>
    %170 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
    %171 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%169, %cst_266 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%170 : tensor<1x56x56x64xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x64xf32>
    %172 = linalg.init_tensor [1, 56, 56, 256] : tensor<1x56x56x256xf32>
    %cst_349 = arith.constant 0.000000e+00 : f32
    %173 = linalg.fill ins(%cst_349 : f32) outs(%172 : tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %174 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%171, %cst_287 : tensor<1x56x56x64xf32>, tensor<1x1x64x256xf32>) outs(%173 : tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
    %175 = linalg.init_tensor [1, 56, 56, 256] : tensor<1x56x56x256xf32>
    %176 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%174, %cst_49 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%175 : tensor<1x56x56x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x256xf32>
    %177 = linalg.init_tensor [1, 56, 56, 256] : tensor<1x56x56x256xf32>
    %178 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%176, %cst_53 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%177 : tensor<1x56x56x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x256xf32>
    %179 = linalg.init_tensor [1, 56, 56, 256] : tensor<1x56x56x256xf32>
    %180 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%178, %cst_51 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%179 : tensor<1x56x56x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x256xf32>
    %181 = linalg.init_tensor [1, 56, 56, 256] : tensor<1x56x56x256xf32>
    %182 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%180, %cst_50 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%181 : tensor<1x56x56x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x256xf32>
    %183 = linalg.init_tensor [1, 56, 56, 256] : tensor<1x56x56x256xf32>
    %184 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%182, %cst_52 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%183 : tensor<1x56x56x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x256xf32>
    %185 = linalg.init_tensor [1, 56, 56, 256] : tensor<1x56x56x256xf32>
    %186 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%138, %184 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%185 : tensor<1x56x56x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x256xf32>
    %187 = linalg.init_tensor [1, 56, 56, 256] : tensor<1x56x56x256xf32>
    %188 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%186, %cst_267 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%187 : tensor<1x56x56x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x56x56x256xf32>
    %189 = linalg.init_tensor [1, 28, 28, 512] : tensor<1x28x28x512xf32>
    %cst_350 = arith.constant 0.000000e+00 : f32
    %190 = linalg.fill ins(%cst_350 : f32) outs(%189 : tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %191 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%188, %cst_288 : tensor<1x56x56x256xf32>, tensor<1x1x256x512xf32>) outs(%190 : tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %192 = linalg.init_tensor [1, 28, 28, 512] : tensor<1x28x28x512xf32>
    %193 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%191, %cst_54 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%192 : tensor<1x28x28x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x512xf32>
    %194 = linalg.init_tensor [1, 28, 28, 512] : tensor<1x28x28x512xf32>
    %195 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%193, %cst_58 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%194 : tensor<1x28x28x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x512xf32>
    %196 = linalg.init_tensor [1, 28, 28, 512] : tensor<1x28x28x512xf32>
    %197 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%195, %cst_56 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%196 : tensor<1x28x28x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x512xf32>
    %198 = linalg.init_tensor [1, 28, 28, 512] : tensor<1x28x28x512xf32>
    %199 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%197, %cst_55 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%198 : tensor<1x28x28x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x512xf32>
    %200 = linalg.init_tensor [1, 28, 28, 512] : tensor<1x28x28x512xf32>
    %201 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%199, %cst_57 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%200 : tensor<1x28x28x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x512xf32>
    %202 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %cst_351 = arith.constant 0.000000e+00 : f32
    %203 = linalg.fill ins(%cst_351 : f32) outs(%202 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %204 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%188, %cst_289 : tensor<1x56x56x256xf32>, tensor<1x1x256x128xf32>) outs(%203 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %205 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %206 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%204, %cst_59 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%205 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %207 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %208 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%206, %cst_63 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%207 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %209 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %210 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%208, %cst_61 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%209 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %211 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %212 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%210, %cst_60 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%211 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %213 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %214 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%212, %cst_62 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%213 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %215 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %216 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%214, %cst_268 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%215 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %217 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %cst_352 = arith.constant 0.000000e+00 : f32
    %218 = linalg.fill ins(%cst_352 : f32) outs(%217 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %cst_353 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_354 = arith.constant 0.000000e+00 : f32
    %219 = linalg.init_tensor [1, 30, 30, 128] : tensor<1x30x30x128xf32>
    %220 = linalg.fill ins(%cst_354 : f32) outs(%219 : tensor<1x30x30x128xf32>) -> tensor<1x30x30x128xf32>
    %221 = tensor.insert_slice %216 into %220[0, 1, 1, 0] [1, 28, 28, 128] [1, 1, 1, 1] : tensor<1x28x28x128xf32> into tensor<1x30x30x128xf32>
    %222 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%221, %cst_290 : tensor<1x30x30x128xf32>, tensor<3x3x128x128xf32>) outs(%218 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %223 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %224 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%222, %cst_64 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%223 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %225 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %226 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%224, %cst_68 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%225 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %227 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %228 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%226, %cst_66 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%227 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %229 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %230 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%228, %cst_65 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%229 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %231 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %232 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%230, %cst_67 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%231 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %233 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %234 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%232, %cst_268 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%233 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %235 = linalg.init_tensor [1, 28, 28, 512] : tensor<1x28x28x512xf32>
    %cst_355 = arith.constant 0.000000e+00 : f32
    %236 = linalg.fill ins(%cst_355 : f32) outs(%235 : tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %237 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%234, %cst_291 : tensor<1x28x28x128xf32>, tensor<1x1x128x512xf32>) outs(%236 : tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %238 = linalg.init_tensor [1, 28, 28, 512] : tensor<1x28x28x512xf32>
    %239 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%237, %cst_69 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%238 : tensor<1x28x28x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x512xf32>
    %240 = linalg.init_tensor [1, 28, 28, 512] : tensor<1x28x28x512xf32>
    %241 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%239, %cst_73 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%240 : tensor<1x28x28x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x512xf32>
    %242 = linalg.init_tensor [1, 28, 28, 512] : tensor<1x28x28x512xf32>
    %243 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%241, %cst_71 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%242 : tensor<1x28x28x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x512xf32>
    %244 = linalg.init_tensor [1, 28, 28, 512] : tensor<1x28x28x512xf32>
    %245 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%243, %cst_70 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%244 : tensor<1x28x28x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x512xf32>
    %246 = linalg.init_tensor [1, 28, 28, 512] : tensor<1x28x28x512xf32>
    %247 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%245, %cst_72 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%246 : tensor<1x28x28x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x512xf32>
    %248 = linalg.init_tensor [1, 28, 28, 512] : tensor<1x28x28x512xf32>
    %249 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%201, %247 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%248 : tensor<1x28x28x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x512xf32>
    %250 = linalg.init_tensor [1, 28, 28, 512] : tensor<1x28x28x512xf32>
    %251 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%249, %cst_269 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%250 : tensor<1x28x28x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x512xf32>
    %252 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %cst_356 = arith.constant 0.000000e+00 : f32
    %253 = linalg.fill ins(%cst_356 : f32) outs(%252 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %254 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%251, %cst_292 : tensor<1x28x28x512xf32>, tensor<1x1x512x128xf32>) outs(%253 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %255 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %256 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%254, %cst_74 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%255 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %257 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %258 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%256, %cst_78 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%257 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %259 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %260 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%258, %cst_76 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%259 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %261 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %262 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%260, %cst_75 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%261 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %263 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %264 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%262, %cst_77 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%263 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %265 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %266 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%264, %cst_268 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%265 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %267 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %cst_357 = arith.constant 0.000000e+00 : f32
    %268 = linalg.fill ins(%cst_357 : f32) outs(%267 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %cst_358 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_359 = arith.constant 0.000000e+00 : f32
    %269 = linalg.init_tensor [1, 30, 30, 128] : tensor<1x30x30x128xf32>
    %270 = linalg.fill ins(%cst_359 : f32) outs(%269 : tensor<1x30x30x128xf32>) -> tensor<1x30x30x128xf32>
    %271 = tensor.insert_slice %266 into %270[0, 1, 1, 0] [1, 28, 28, 128] [1, 1, 1, 1] : tensor<1x28x28x128xf32> into tensor<1x30x30x128xf32>
    %272 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%271, %cst_293 : tensor<1x30x30x128xf32>, tensor<3x3x128x128xf32>) outs(%268 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %273 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %274 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%272, %cst_79 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%273 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %275 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %276 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%274, %cst_83 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%275 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %277 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %278 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%276, %cst_81 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%277 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %279 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %280 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%278, %cst_80 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%279 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %281 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %282 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%280, %cst_82 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%281 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %283 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %284 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%282, %cst_268 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%283 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %285 = linalg.init_tensor [1, 28, 28, 512] : tensor<1x28x28x512xf32>
    %cst_360 = arith.constant 0.000000e+00 : f32
    %286 = linalg.fill ins(%cst_360 : f32) outs(%285 : tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %287 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%284, %cst_294 : tensor<1x28x28x128xf32>, tensor<1x1x128x512xf32>) outs(%286 : tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %288 = linalg.init_tensor [1, 28, 28, 512] : tensor<1x28x28x512xf32>
    %289 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%287, %cst_84 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%288 : tensor<1x28x28x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x512xf32>
    %290 = linalg.init_tensor [1, 28, 28, 512] : tensor<1x28x28x512xf32>
    %291 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%289, %cst_88 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%290 : tensor<1x28x28x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x512xf32>
    %292 = linalg.init_tensor [1, 28, 28, 512] : tensor<1x28x28x512xf32>
    %293 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%291, %cst_86 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%292 : tensor<1x28x28x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x512xf32>
    %294 = linalg.init_tensor [1, 28, 28, 512] : tensor<1x28x28x512xf32>
    %295 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%293, %cst_85 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%294 : tensor<1x28x28x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x512xf32>
    %296 = linalg.init_tensor [1, 28, 28, 512] : tensor<1x28x28x512xf32>
    %297 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%295, %cst_87 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%296 : tensor<1x28x28x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x512xf32>
    %298 = linalg.init_tensor [1, 28, 28, 512] : tensor<1x28x28x512xf32>
    %299 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%251, %297 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%298 : tensor<1x28x28x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x512xf32>
    %300 = linalg.init_tensor [1, 28, 28, 512] : tensor<1x28x28x512xf32>
    %301 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%299, %cst_269 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%300 : tensor<1x28x28x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x512xf32>
    %302 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %cst_361 = arith.constant 0.000000e+00 : f32
    %303 = linalg.fill ins(%cst_361 : f32) outs(%302 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %304 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%301, %cst_295 : tensor<1x28x28x512xf32>, tensor<1x1x512x128xf32>) outs(%303 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %305 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %306 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%304, %cst_89 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%305 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %307 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %308 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%306, %cst_93 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%307 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %309 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %310 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%308, %cst_91 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%309 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %311 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %312 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%310, %cst_90 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%311 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %313 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %314 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%312, %cst_92 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%313 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %315 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %316 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%314, %cst_268 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%315 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %317 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %cst_362 = arith.constant 0.000000e+00 : f32
    %318 = linalg.fill ins(%cst_362 : f32) outs(%317 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %cst_363 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_364 = arith.constant 0.000000e+00 : f32
    %319 = linalg.init_tensor [1, 30, 30, 128] : tensor<1x30x30x128xf32>
    %320 = linalg.fill ins(%cst_364 : f32) outs(%319 : tensor<1x30x30x128xf32>) -> tensor<1x30x30x128xf32>
    %321 = tensor.insert_slice %316 into %320[0, 1, 1, 0] [1, 28, 28, 128] [1, 1, 1, 1] : tensor<1x28x28x128xf32> into tensor<1x30x30x128xf32>
    %322 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%321, %cst_296 : tensor<1x30x30x128xf32>, tensor<3x3x128x128xf32>) outs(%318 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %323 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %324 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%322, %cst_94 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%323 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %325 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %326 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%324, %cst_98 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%325 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %327 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %328 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%326, %cst_96 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%327 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %329 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %330 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%328, %cst_95 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%329 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %331 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %332 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%330, %cst_97 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%331 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %333 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %334 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%332, %cst_268 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%333 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %335 = linalg.init_tensor [1, 28, 28, 512] : tensor<1x28x28x512xf32>
    %cst_365 = arith.constant 0.000000e+00 : f32
    %336 = linalg.fill ins(%cst_365 : f32) outs(%335 : tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %337 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%334, %cst_297 : tensor<1x28x28x128xf32>, tensor<1x1x128x512xf32>) outs(%336 : tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %338 = linalg.init_tensor [1, 28, 28, 512] : tensor<1x28x28x512xf32>
    %339 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%337, %cst_99 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%338 : tensor<1x28x28x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x512xf32>
    %340 = linalg.init_tensor [1, 28, 28, 512] : tensor<1x28x28x512xf32>
    %341 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%339, %cst_103 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%340 : tensor<1x28x28x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x512xf32>
    %342 = linalg.init_tensor [1, 28, 28, 512] : tensor<1x28x28x512xf32>
    %343 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%341, %cst_101 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%342 : tensor<1x28x28x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x512xf32>
    %344 = linalg.init_tensor [1, 28, 28, 512] : tensor<1x28x28x512xf32>
    %345 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%343, %cst_100 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%344 : tensor<1x28x28x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x512xf32>
    %346 = linalg.init_tensor [1, 28, 28, 512] : tensor<1x28x28x512xf32>
    %347 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%345, %cst_102 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%346 : tensor<1x28x28x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x512xf32>
    %348 = linalg.init_tensor [1, 28, 28, 512] : tensor<1x28x28x512xf32>
    %349 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%301, %347 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%348 : tensor<1x28x28x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x512xf32>
    %350 = linalg.init_tensor [1, 28, 28, 512] : tensor<1x28x28x512xf32>
    %351 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%349, %cst_269 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%350 : tensor<1x28x28x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x512xf32>
    %352 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %cst_366 = arith.constant 0.000000e+00 : f32
    %353 = linalg.fill ins(%cst_366 : f32) outs(%352 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %354 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%351, %cst_298 : tensor<1x28x28x512xf32>, tensor<1x1x512x128xf32>) outs(%353 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %355 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %356 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%354, %cst_104 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%355 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %357 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %358 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%356, %cst_108 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%357 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %359 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %360 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%358, %cst_106 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%359 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %361 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %362 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%360, %cst_105 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%361 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %363 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %364 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%362, %cst_107 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%363 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %365 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %366 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%364, %cst_268 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%365 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %367 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %cst_367 = arith.constant 0.000000e+00 : f32
    %368 = linalg.fill ins(%cst_367 : f32) outs(%367 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %cst_368 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_369 = arith.constant 0.000000e+00 : f32
    %369 = linalg.init_tensor [1, 30, 30, 128] : tensor<1x30x30x128xf32>
    %370 = linalg.fill ins(%cst_369 : f32) outs(%369 : tensor<1x30x30x128xf32>) -> tensor<1x30x30x128xf32>
    %371 = tensor.insert_slice %366 into %370[0, 1, 1, 0] [1, 28, 28, 128] [1, 1, 1, 1] : tensor<1x28x28x128xf32> into tensor<1x30x30x128xf32>
    %372 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%371, %cst_299 : tensor<1x30x30x128xf32>, tensor<3x3x128x128xf32>) outs(%368 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %373 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %374 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%372, %cst_109 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%373 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %375 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %376 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%374, %cst_113 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%375 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %377 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %378 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%376, %cst_111 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%377 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %379 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %380 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%378, %cst_110 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%379 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %381 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %382 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%380, %cst_112 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%381 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %383 = linalg.init_tensor [1, 28, 28, 128] : tensor<1x28x28x128xf32>
    %384 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%382, %cst_268 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%383 : tensor<1x28x28x128xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x128xf32>
    %385 = linalg.init_tensor [1, 28, 28, 512] : tensor<1x28x28x512xf32>
    %cst_370 = arith.constant 0.000000e+00 : f32
    %386 = linalg.fill ins(%cst_370 : f32) outs(%385 : tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %387 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%384, %cst_300 : tensor<1x28x28x128xf32>, tensor<1x1x128x512xf32>) outs(%386 : tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
    %388 = linalg.init_tensor [1, 28, 28, 512] : tensor<1x28x28x512xf32>
    %389 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%387, %cst_114 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%388 : tensor<1x28x28x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x512xf32>
    %390 = linalg.init_tensor [1, 28, 28, 512] : tensor<1x28x28x512xf32>
    %391 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%389, %cst_118 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%390 : tensor<1x28x28x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x512xf32>
    %392 = linalg.init_tensor [1, 28, 28, 512] : tensor<1x28x28x512xf32>
    %393 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%391, %cst_116 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%392 : tensor<1x28x28x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x512xf32>
    %394 = linalg.init_tensor [1, 28, 28, 512] : tensor<1x28x28x512xf32>
    %395 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%393, %cst_115 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%394 : tensor<1x28x28x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x512xf32>
    %396 = linalg.init_tensor [1, 28, 28, 512] : tensor<1x28x28x512xf32>
    %397 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%395, %cst_117 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%396 : tensor<1x28x28x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x512xf32>
    %398 = linalg.init_tensor [1, 28, 28, 512] : tensor<1x28x28x512xf32>
    %399 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%351, %397 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%398 : tensor<1x28x28x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x512xf32>
    %400 = linalg.init_tensor [1, 28, 28, 512] : tensor<1x28x28x512xf32>
    %401 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%399, %cst_269 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%400 : tensor<1x28x28x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x28x512xf32>
    %402 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %cst_371 = arith.constant 0.000000e+00 : f32
    %403 = linalg.fill ins(%cst_371 : f32) outs(%402 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %404 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%401, %cst_301 : tensor<1x28x28x512xf32>, tensor<1x1x512x1024xf32>) outs(%403 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %405 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %406 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%404, %cst_119 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%405 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %407 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %408 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%406, %cst_123 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%407 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %409 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %410 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%408, %cst_121 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%409 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %411 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %412 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%410, %cst_120 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%411 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %413 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %414 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%412, %cst_122 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%413 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %415 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %cst_372 = arith.constant 0.000000e+00 : f32
    %416 = linalg.fill ins(%cst_372 : f32) outs(%415 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %417 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%401, %cst_302 : tensor<1x28x28x512xf32>, tensor<1x1x512x256xf32>) outs(%416 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %418 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %419 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%417, %cst_124 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%418 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %420 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %421 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%419, %cst_128 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%420 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %422 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %423 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%421, %cst_126 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%422 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %424 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %425 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%423, %cst_125 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%424 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %426 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %427 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%425, %cst_127 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%426 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %428 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %429 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%427, %cst_270 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%428 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %430 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %cst_373 = arith.constant 0.000000e+00 : f32
    %431 = linalg.fill ins(%cst_373 : f32) outs(%430 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %cst_374 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_375 = arith.constant 0.000000e+00 : f32
    %432 = linalg.init_tensor [1, 16, 16, 256] : tensor<1x16x16x256xf32>
    %433 = linalg.fill ins(%cst_375 : f32) outs(%432 : tensor<1x16x16x256xf32>) -> tensor<1x16x16x256xf32>
    %434 = tensor.insert_slice %429 into %433[0, 1, 1, 0] [1, 14, 14, 256] [1, 1, 1, 1] : tensor<1x14x14x256xf32> into tensor<1x16x16x256xf32>
    %435 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%434, %cst_303 : tensor<1x16x16x256xf32>, tensor<3x3x256x256xf32>) outs(%431 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %436 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %437 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%435, %cst_129 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%436 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %438 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %439 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%437, %cst_133 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%438 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %440 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %441 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%439, %cst_131 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%440 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %442 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %443 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%441, %cst_130 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%442 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %444 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %445 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%443, %cst_132 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%444 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %446 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %447 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%445, %cst_270 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%446 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %448 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %cst_376 = arith.constant 0.000000e+00 : f32
    %449 = linalg.fill ins(%cst_376 : f32) outs(%448 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %450 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%447, %cst_304 : tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32>) outs(%449 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %451 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %452 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%450, %cst_134 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%451 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %453 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %454 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%452, %cst_138 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%453 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %455 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %456 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%454, %cst_136 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%455 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %457 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %458 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%456, %cst_135 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%457 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %459 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %460 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%458, %cst_137 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%459 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %461 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %462 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%414, %460 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%461 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %463 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %464 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%462, %cst_271 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%463 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %465 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %cst_377 = arith.constant 0.000000e+00 : f32
    %466 = linalg.fill ins(%cst_377 : f32) outs(%465 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %467 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%464, %cst_305 : tensor<1x14x14x1024xf32>, tensor<1x1x1024x256xf32>) outs(%466 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %468 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %469 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%467, %cst_139 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%468 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %470 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %471 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%469, %cst_143 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%470 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %472 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %473 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%471, %cst_141 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%472 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %474 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %475 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%473, %cst_140 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%474 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %476 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %477 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%475, %cst_142 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%476 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %478 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %479 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%477, %cst_270 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%478 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %480 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %cst_378 = arith.constant 0.000000e+00 : f32
    %481 = linalg.fill ins(%cst_378 : f32) outs(%480 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %cst_379 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_380 = arith.constant 0.000000e+00 : f32
    %482 = linalg.init_tensor [1, 16, 16, 256] : tensor<1x16x16x256xf32>
    %483 = linalg.fill ins(%cst_380 : f32) outs(%482 : tensor<1x16x16x256xf32>) -> tensor<1x16x16x256xf32>
    %484 = tensor.insert_slice %479 into %483[0, 1, 1, 0] [1, 14, 14, 256] [1, 1, 1, 1] : tensor<1x14x14x256xf32> into tensor<1x16x16x256xf32>
    %485 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%484, %cst_306 : tensor<1x16x16x256xf32>, tensor<3x3x256x256xf32>) outs(%481 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %486 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %487 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%485, %cst_144 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%486 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %488 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %489 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%487, %cst_148 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%488 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %490 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %491 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%489, %cst_146 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%490 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %492 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %493 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%491, %cst_145 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%492 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %494 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %495 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%493, %cst_147 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%494 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %496 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %497 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%495, %cst_270 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%496 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %498 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %cst_381 = arith.constant 0.000000e+00 : f32
    %499 = linalg.fill ins(%cst_381 : f32) outs(%498 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %500 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%497, %cst_307 : tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32>) outs(%499 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %501 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %502 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%500, %cst_149 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%501 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %503 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %504 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%502, %cst_153 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%503 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %505 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %506 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%504, %cst_151 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%505 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %507 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %508 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%506, %cst_150 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%507 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %509 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %510 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%508, %cst_152 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%509 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %511 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %512 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%464, %510 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%511 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %513 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %514 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%512, %cst_271 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%513 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %515 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %cst_382 = arith.constant 0.000000e+00 : f32
    %516 = linalg.fill ins(%cst_382 : f32) outs(%515 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %517 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%514, %cst_308 : tensor<1x14x14x1024xf32>, tensor<1x1x1024x256xf32>) outs(%516 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %518 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %519 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%517, %cst_154 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%518 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %520 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %521 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%519, %cst_158 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%520 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %522 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %523 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%521, %cst_156 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%522 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %524 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %525 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%523, %cst_155 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%524 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %526 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %527 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%525, %cst_157 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%526 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %528 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %529 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%527, %cst_270 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%528 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %530 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %cst_383 = arith.constant 0.000000e+00 : f32
    %531 = linalg.fill ins(%cst_383 : f32) outs(%530 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %cst_384 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_385 = arith.constant 0.000000e+00 : f32
    %532 = linalg.init_tensor [1, 16, 16, 256] : tensor<1x16x16x256xf32>
    %533 = linalg.fill ins(%cst_385 : f32) outs(%532 : tensor<1x16x16x256xf32>) -> tensor<1x16x16x256xf32>
    %534 = tensor.insert_slice %529 into %533[0, 1, 1, 0] [1, 14, 14, 256] [1, 1, 1, 1] : tensor<1x14x14x256xf32> into tensor<1x16x16x256xf32>
    %535 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%534, %cst_309 : tensor<1x16x16x256xf32>, tensor<3x3x256x256xf32>) outs(%531 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %536 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %537 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%535, %cst_159 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%536 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %538 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %539 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%537, %cst_163 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%538 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %540 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %541 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%539, %cst_161 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%540 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %542 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %543 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%541, %cst_160 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%542 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %544 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %545 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%543, %cst_162 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%544 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %546 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %547 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%545, %cst_270 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%546 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %548 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %cst_386 = arith.constant 0.000000e+00 : f32
    %549 = linalg.fill ins(%cst_386 : f32) outs(%548 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %550 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%547, %cst_310 : tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32>) outs(%549 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %551 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %552 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%550, %cst_164 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%551 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %553 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %554 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%552, %cst_168 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%553 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %555 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %556 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%554, %cst_166 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%555 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %557 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %558 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%556, %cst_165 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%557 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %559 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %560 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%558, %cst_167 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%559 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %561 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %562 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%514, %560 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%561 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %563 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %564 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%562, %cst_271 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%563 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %565 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %cst_387 = arith.constant 0.000000e+00 : f32
    %566 = linalg.fill ins(%cst_387 : f32) outs(%565 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %567 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%564, %cst_311 : tensor<1x14x14x1024xf32>, tensor<1x1x1024x256xf32>) outs(%566 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %568 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %569 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%567, %cst_169 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%568 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %570 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %571 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%569, %cst_173 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%570 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %572 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %573 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%571, %cst_171 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%572 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %574 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %575 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%573, %cst_170 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%574 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %576 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %577 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%575, %cst_172 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%576 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %578 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %579 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%577, %cst_270 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%578 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %580 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %cst_388 = arith.constant 0.000000e+00 : f32
    %581 = linalg.fill ins(%cst_388 : f32) outs(%580 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %cst_389 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_390 = arith.constant 0.000000e+00 : f32
    %582 = linalg.init_tensor [1, 16, 16, 256] : tensor<1x16x16x256xf32>
    %583 = linalg.fill ins(%cst_390 : f32) outs(%582 : tensor<1x16x16x256xf32>) -> tensor<1x16x16x256xf32>
    %584 = tensor.insert_slice %579 into %583[0, 1, 1, 0] [1, 14, 14, 256] [1, 1, 1, 1] : tensor<1x14x14x256xf32> into tensor<1x16x16x256xf32>
    %585 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%584, %cst_312 : tensor<1x16x16x256xf32>, tensor<3x3x256x256xf32>) outs(%581 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %586 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %587 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%585, %cst_174 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%586 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %588 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %589 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%587, %cst_178 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%588 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %590 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %591 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%589, %cst_176 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%590 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %592 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %593 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%591, %cst_175 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%592 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %594 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %595 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%593, %cst_177 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%594 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %596 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %597 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%595, %cst_270 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%596 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %598 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %cst_391 = arith.constant 0.000000e+00 : f32
    %599 = linalg.fill ins(%cst_391 : f32) outs(%598 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %600 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%597, %cst_313 : tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32>) outs(%599 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %601 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %602 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%600, %cst_179 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%601 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %603 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %604 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%602, %cst_183 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%603 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %605 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %606 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%604, %cst_181 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%605 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %607 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %608 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%606, %cst_180 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%607 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %609 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %610 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%608, %cst_182 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%609 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %611 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %612 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%564, %610 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%611 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %613 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %614 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%612, %cst_271 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%613 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %615 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %cst_392 = arith.constant 0.000000e+00 : f32
    %616 = linalg.fill ins(%cst_392 : f32) outs(%615 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %617 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%614, %cst_314 : tensor<1x14x14x1024xf32>, tensor<1x1x1024x256xf32>) outs(%616 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %618 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %619 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%617, %cst_184 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%618 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %620 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %621 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%619, %cst_188 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%620 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %622 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %623 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%621, %cst_186 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%622 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %624 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %625 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%623, %cst_185 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%624 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %626 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %627 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%625, %cst_187 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%626 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %628 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %629 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%627, %cst_270 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%628 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %630 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %cst_393 = arith.constant 0.000000e+00 : f32
    %631 = linalg.fill ins(%cst_393 : f32) outs(%630 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %cst_394 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_395 = arith.constant 0.000000e+00 : f32
    %632 = linalg.init_tensor [1, 16, 16, 256] : tensor<1x16x16x256xf32>
    %633 = linalg.fill ins(%cst_395 : f32) outs(%632 : tensor<1x16x16x256xf32>) -> tensor<1x16x16x256xf32>
    %634 = tensor.insert_slice %629 into %633[0, 1, 1, 0] [1, 14, 14, 256] [1, 1, 1, 1] : tensor<1x14x14x256xf32> into tensor<1x16x16x256xf32>
    %635 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%634, %cst_315 : tensor<1x16x16x256xf32>, tensor<3x3x256x256xf32>) outs(%631 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %636 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %637 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%635, %cst_189 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%636 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %638 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %639 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%637, %cst_193 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%638 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %640 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %641 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%639, %cst_191 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%640 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %642 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %643 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%641, %cst_190 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%642 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %644 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %645 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%643, %cst_192 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%644 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %646 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %647 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%645, %cst_270 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%646 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %648 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %cst_396 = arith.constant 0.000000e+00 : f32
    %649 = linalg.fill ins(%cst_396 : f32) outs(%648 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %650 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%647, %cst_316 : tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32>) outs(%649 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %651 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %652 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%650, %cst_194 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%651 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %653 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %654 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%652, %cst_198 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%653 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %655 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %656 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%654, %cst_196 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%655 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %657 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %658 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%656, %cst_195 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%657 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %659 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %660 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%658, %cst_197 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%659 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %661 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %662 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%614, %660 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%661 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %663 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %664 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%662, %cst_271 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%663 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %665 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %cst_397 = arith.constant 0.000000e+00 : f32
    %666 = linalg.fill ins(%cst_397 : f32) outs(%665 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %667 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%664, %cst_317 : tensor<1x14x14x1024xf32>, tensor<1x1x1024x256xf32>) outs(%666 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %668 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %669 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%667, %cst_199 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%668 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %670 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %671 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%669, %cst_203 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%670 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %672 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %673 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%671, %cst_201 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%672 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %674 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %675 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%673, %cst_200 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%674 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %676 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %677 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%675, %cst_202 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%676 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %678 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %679 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%677, %cst_270 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%678 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %680 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %cst_398 = arith.constant 0.000000e+00 : f32
    %681 = linalg.fill ins(%cst_398 : f32) outs(%680 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %cst_399 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_400 = arith.constant 0.000000e+00 : f32
    %682 = linalg.init_tensor [1, 16, 16, 256] : tensor<1x16x16x256xf32>
    %683 = linalg.fill ins(%cst_400 : f32) outs(%682 : tensor<1x16x16x256xf32>) -> tensor<1x16x16x256xf32>
    %684 = tensor.insert_slice %679 into %683[0, 1, 1, 0] [1, 14, 14, 256] [1, 1, 1, 1] : tensor<1x14x14x256xf32> into tensor<1x16x16x256xf32>
    %685 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%684, %cst_318 : tensor<1x16x16x256xf32>, tensor<3x3x256x256xf32>) outs(%681 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %686 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %687 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%685, %cst_204 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%686 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %688 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %689 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%687, %cst_208 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%688 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %690 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %691 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%689, %cst_206 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%690 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %692 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %693 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%691, %cst_205 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%692 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %694 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %695 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%693, %cst_207 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%694 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %696 = linalg.init_tensor [1, 14, 14, 256] : tensor<1x14x14x256xf32>
    %697 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%695, %cst_270 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%696 : tensor<1x14x14x256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x256xf32>
    %698 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %cst_401 = arith.constant 0.000000e+00 : f32
    %699 = linalg.fill ins(%cst_401 : f32) outs(%698 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %700 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%697, %cst_319 : tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32>) outs(%699 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
    %701 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %702 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%700, %cst_209 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%701 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %703 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %704 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%702, %cst_213 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%703 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %705 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %706 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%704, %cst_211 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%705 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %707 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %708 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%706, %cst_210 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%707 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %709 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %710 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%708, %cst_212 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%709 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %711 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %712 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%664, %710 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%711 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %713 = linalg.init_tensor [1, 14, 14, 1024] : tensor<1x14x14x1024xf32>
    %714 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%712, %cst_271 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%713 : tensor<1x14x14x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x14x14x1024xf32>
    %715 = linalg.init_tensor [1, 7, 7, 2048] : tensor<1x7x7x2048xf32>
    %cst_402 = arith.constant 0.000000e+00 : f32
    %716 = linalg.fill ins(%cst_402 : f32) outs(%715 : tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
    %717 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%714, %cst_320 : tensor<1x14x14x1024xf32>, tensor<1x1x1024x2048xf32>) outs(%716 : tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
    %718 = linalg.init_tensor [1, 7, 7, 2048] : tensor<1x7x7x2048xf32>
    %719 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%717, %cst_214 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%718 : tensor<1x7x7x2048xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x2048xf32>
    %720 = linalg.init_tensor [1, 7, 7, 2048] : tensor<1x7x7x2048xf32>
    %721 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%719, %cst_218 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%720 : tensor<1x7x7x2048xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x2048xf32>
    %722 = linalg.init_tensor [1, 7, 7, 2048] : tensor<1x7x7x2048xf32>
    %723 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%721, %cst_216 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%722 : tensor<1x7x7x2048xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x2048xf32>
    %724 = linalg.init_tensor [1, 7, 7, 2048] : tensor<1x7x7x2048xf32>
    %725 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%723, %cst_215 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%724 : tensor<1x7x7x2048xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x2048xf32>
    %726 = linalg.init_tensor [1, 7, 7, 2048] : tensor<1x7x7x2048xf32>
    %727 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%725, %cst_217 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%726 : tensor<1x7x7x2048xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x2048xf32>
    %728 = linalg.init_tensor [1, 7, 7, 512] : tensor<1x7x7x512xf32>
    %cst_403 = arith.constant 0.000000e+00 : f32
    %729 = linalg.fill ins(%cst_403 : f32) outs(%728 : tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %730 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%714, %cst_321 : tensor<1x14x14x1024xf32>, tensor<1x1x1024x512xf32>) outs(%729 : tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %731 = linalg.init_tensor [1, 7, 7, 512] : tensor<1x7x7x512xf32>
    %732 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%730, %cst_219 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%731 : tensor<1x7x7x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x512xf32>
    %733 = linalg.init_tensor [1, 7, 7, 512] : tensor<1x7x7x512xf32>
    %734 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%732, %cst_223 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%733 : tensor<1x7x7x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x512xf32>
    %735 = linalg.init_tensor [1, 7, 7, 512] : tensor<1x7x7x512xf32>
    %736 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%734, %cst_221 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%735 : tensor<1x7x7x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x512xf32>
    %737 = linalg.init_tensor [1, 7, 7, 512] : tensor<1x7x7x512xf32>
    %738 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%736, %cst_220 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%737 : tensor<1x7x7x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x512xf32>
    %739 = linalg.init_tensor [1, 7, 7, 512] : tensor<1x7x7x512xf32>
    %740 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%738, %cst_222 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%739 : tensor<1x7x7x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x512xf32>
    %741 = linalg.init_tensor [1, 7, 7, 512] : tensor<1x7x7x512xf32>
    %742 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%740, %cst_272 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%741 : tensor<1x7x7x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x512xf32>
    %743 = linalg.init_tensor [1, 7, 7, 512] : tensor<1x7x7x512xf32>
    %cst_404 = arith.constant 0.000000e+00 : f32
    %744 = linalg.fill ins(%cst_404 : f32) outs(%743 : tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %cst_405 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_406 = arith.constant 0.000000e+00 : f32
    %745 = linalg.init_tensor [1, 9, 9, 512] : tensor<1x9x9x512xf32>
    %746 = linalg.fill ins(%cst_406 : f32) outs(%745 : tensor<1x9x9x512xf32>) -> tensor<1x9x9x512xf32>
    %747 = tensor.insert_slice %742 into %746[0, 1, 1, 0] [1, 7, 7, 512] [1, 1, 1, 1] : tensor<1x7x7x512xf32> into tensor<1x9x9x512xf32>
    %748 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%747, %cst_322 : tensor<1x9x9x512xf32>, tensor<3x3x512x512xf32>) outs(%744 : tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %749 = linalg.init_tensor [1, 7, 7, 512] : tensor<1x7x7x512xf32>
    %750 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%748, %cst_224 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%749 : tensor<1x7x7x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x512xf32>
    %751 = linalg.init_tensor [1, 7, 7, 512] : tensor<1x7x7x512xf32>
    %752 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%750, %cst_228 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%751 : tensor<1x7x7x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x512xf32>
    %753 = linalg.init_tensor [1, 7, 7, 512] : tensor<1x7x7x512xf32>
    %754 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%752, %cst_226 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%753 : tensor<1x7x7x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x512xf32>
    %755 = linalg.init_tensor [1, 7, 7, 512] : tensor<1x7x7x512xf32>
    %756 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%754, %cst_225 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%755 : tensor<1x7x7x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x512xf32>
    %757 = linalg.init_tensor [1, 7, 7, 512] : tensor<1x7x7x512xf32>
    %758 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%756, %cst_227 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%757 : tensor<1x7x7x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x512xf32>
    %759 = linalg.init_tensor [1, 7, 7, 512] : tensor<1x7x7x512xf32>
    %760 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%758, %cst_272 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%759 : tensor<1x7x7x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x512xf32>
    %761 = linalg.init_tensor [1, 7, 7, 2048] : tensor<1x7x7x2048xf32>
    %cst_407 = arith.constant 0.000000e+00 : f32
    %762 = linalg.fill ins(%cst_407 : f32) outs(%761 : tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
    %763 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%760, %cst_323 : tensor<1x7x7x512xf32>, tensor<1x1x512x2048xf32>) outs(%762 : tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
    %764 = linalg.init_tensor [1, 7, 7, 2048] : tensor<1x7x7x2048xf32>
    %765 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%763, %cst_229 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%764 : tensor<1x7x7x2048xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x2048xf32>
    %766 = linalg.init_tensor [1, 7, 7, 2048] : tensor<1x7x7x2048xf32>
    %767 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%765, %cst_233 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%766 : tensor<1x7x7x2048xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x2048xf32>
    %768 = linalg.init_tensor [1, 7, 7, 2048] : tensor<1x7x7x2048xf32>
    %769 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%767, %cst_231 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%768 : tensor<1x7x7x2048xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x2048xf32>
    %770 = linalg.init_tensor [1, 7, 7, 2048] : tensor<1x7x7x2048xf32>
    %771 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%769, %cst_230 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%770 : tensor<1x7x7x2048xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x2048xf32>
    %772 = linalg.init_tensor [1, 7, 7, 2048] : tensor<1x7x7x2048xf32>
    %773 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%771, %cst_232 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%772 : tensor<1x7x7x2048xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x2048xf32>
    %774 = linalg.init_tensor [1, 7, 7, 2048] : tensor<1x7x7x2048xf32>
    %775 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%727, %773 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%774 : tensor<1x7x7x2048xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x2048xf32>
    %776 = linalg.init_tensor [1, 7, 7, 2048] : tensor<1x7x7x2048xf32>
    %777 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%775, %cst_273 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%776 : tensor<1x7x7x2048xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x2048xf32>
    %778 = linalg.init_tensor [1, 7, 7, 512] : tensor<1x7x7x512xf32>
    %cst_408 = arith.constant 0.000000e+00 : f32
    %779 = linalg.fill ins(%cst_408 : f32) outs(%778 : tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %780 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%777, %cst_324 : tensor<1x7x7x2048xf32>, tensor<1x1x2048x512xf32>) outs(%779 : tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %781 = linalg.init_tensor [1, 7, 7, 512] : tensor<1x7x7x512xf32>
    %782 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%780, %cst_234 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%781 : tensor<1x7x7x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x512xf32>
    %783 = linalg.init_tensor [1, 7, 7, 512] : tensor<1x7x7x512xf32>
    %784 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%782, %cst_238 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%783 : tensor<1x7x7x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x512xf32>
    %785 = linalg.init_tensor [1, 7, 7, 512] : tensor<1x7x7x512xf32>
    %786 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%784, %cst_236 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%785 : tensor<1x7x7x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x512xf32>
    %787 = linalg.init_tensor [1, 7, 7, 512] : tensor<1x7x7x512xf32>
    %788 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%786, %cst_235 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%787 : tensor<1x7x7x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x512xf32>
    %789 = linalg.init_tensor [1, 7, 7, 512] : tensor<1x7x7x512xf32>
    %790 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%788, %cst_237 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%789 : tensor<1x7x7x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x512xf32>
    %791 = linalg.init_tensor [1, 7, 7, 512] : tensor<1x7x7x512xf32>
    %792 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%790, %cst_272 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%791 : tensor<1x7x7x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x512xf32>
    %793 = linalg.init_tensor [1, 7, 7, 512] : tensor<1x7x7x512xf32>
    %cst_409 = arith.constant 0.000000e+00 : f32
    %794 = linalg.fill ins(%cst_409 : f32) outs(%793 : tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %cst_410 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_411 = arith.constant 0.000000e+00 : f32
    %795 = linalg.init_tensor [1, 9, 9, 512] : tensor<1x9x9x512xf32>
    %796 = linalg.fill ins(%cst_411 : f32) outs(%795 : tensor<1x9x9x512xf32>) -> tensor<1x9x9x512xf32>
    %797 = tensor.insert_slice %792 into %796[0, 1, 1, 0] [1, 7, 7, 512] [1, 1, 1, 1] : tensor<1x7x7x512xf32> into tensor<1x9x9x512xf32>
    %798 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%797, %cst_325 : tensor<1x9x9x512xf32>, tensor<3x3x512x512xf32>) outs(%794 : tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %799 = linalg.init_tensor [1, 7, 7, 512] : tensor<1x7x7x512xf32>
    %800 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%798, %cst_239 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%799 : tensor<1x7x7x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x512xf32>
    %801 = linalg.init_tensor [1, 7, 7, 512] : tensor<1x7x7x512xf32>
    %802 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%800, %cst_243 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%801 : tensor<1x7x7x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x512xf32>
    %803 = linalg.init_tensor [1, 7, 7, 512] : tensor<1x7x7x512xf32>
    %804 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%802, %cst_241 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%803 : tensor<1x7x7x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x512xf32>
    %805 = linalg.init_tensor [1, 7, 7, 512] : tensor<1x7x7x512xf32>
    %806 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%804, %cst_240 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%805 : tensor<1x7x7x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x512xf32>
    %807 = linalg.init_tensor [1, 7, 7, 512] : tensor<1x7x7x512xf32>
    %808 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%806, %cst_242 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%807 : tensor<1x7x7x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x512xf32>
    %809 = linalg.init_tensor [1, 7, 7, 512] : tensor<1x7x7x512xf32>
    %810 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%808, %cst_272 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%809 : tensor<1x7x7x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x512xf32>
    %811 = linalg.init_tensor [1, 7, 7, 2048] : tensor<1x7x7x2048xf32>
    %cst_412 = arith.constant 0.000000e+00 : f32
    %812 = linalg.fill ins(%cst_412 : f32) outs(%811 : tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
    %813 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%810, %cst_326 : tensor<1x7x7x512xf32>, tensor<1x1x512x2048xf32>) outs(%812 : tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
    %814 = linalg.init_tensor [1, 7, 7, 2048] : tensor<1x7x7x2048xf32>
    %815 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%813, %cst_244 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%814 : tensor<1x7x7x2048xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x2048xf32>
    %816 = linalg.init_tensor [1, 7, 7, 2048] : tensor<1x7x7x2048xf32>
    %817 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%815, %cst_248 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%816 : tensor<1x7x7x2048xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x2048xf32>
    %818 = linalg.init_tensor [1, 7, 7, 2048] : tensor<1x7x7x2048xf32>
    %819 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%817, %cst_246 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%818 : tensor<1x7x7x2048xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x2048xf32>
    %820 = linalg.init_tensor [1, 7, 7, 2048] : tensor<1x7x7x2048xf32>
    %821 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%819, %cst_245 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%820 : tensor<1x7x7x2048xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x2048xf32>
    %822 = linalg.init_tensor [1, 7, 7, 2048] : tensor<1x7x7x2048xf32>
    %823 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%821, %cst_247 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%822 : tensor<1x7x7x2048xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x2048xf32>
    %824 = linalg.init_tensor [1, 7, 7, 2048] : tensor<1x7x7x2048xf32>
    %825 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%777, %823 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%824 : tensor<1x7x7x2048xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x2048xf32>
    %826 = linalg.init_tensor [1, 7, 7, 2048] : tensor<1x7x7x2048xf32>
    %827 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%825, %cst_273 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%826 : tensor<1x7x7x2048xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x2048xf32>
    %828 = linalg.init_tensor [1, 7, 7, 512] : tensor<1x7x7x512xf32>
    %cst_413 = arith.constant 0.000000e+00 : f32
    %829 = linalg.fill ins(%cst_413 : f32) outs(%828 : tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %830 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%827, %cst_327 : tensor<1x7x7x2048xf32>, tensor<1x1x2048x512xf32>) outs(%829 : tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %831 = linalg.init_tensor [1, 7, 7, 512] : tensor<1x7x7x512xf32>
    %832 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%830, %cst_249 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%831 : tensor<1x7x7x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x512xf32>
    %833 = linalg.init_tensor [1, 7, 7, 512] : tensor<1x7x7x512xf32>
    %834 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%832, %cst_253 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%833 : tensor<1x7x7x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x512xf32>
    %835 = linalg.init_tensor [1, 7, 7, 512] : tensor<1x7x7x512xf32>
    %836 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%834, %cst_251 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%835 : tensor<1x7x7x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x512xf32>
    %837 = linalg.init_tensor [1, 7, 7, 512] : tensor<1x7x7x512xf32>
    %838 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%836, %cst_250 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%837 : tensor<1x7x7x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x512xf32>
    %839 = linalg.init_tensor [1, 7, 7, 512] : tensor<1x7x7x512xf32>
    %840 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%838, %cst_252 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%839 : tensor<1x7x7x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x512xf32>
    %841 = linalg.init_tensor [1, 7, 7, 512] : tensor<1x7x7x512xf32>
    %842 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%840, %cst_272 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%841 : tensor<1x7x7x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x512xf32>
    %843 = linalg.init_tensor [1, 7, 7, 512] : tensor<1x7x7x512xf32>
    %cst_414 = arith.constant 0.000000e+00 : f32
    %844 = linalg.fill ins(%cst_414 : f32) outs(%843 : tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %cst_415 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_416 = arith.constant 0.000000e+00 : f32
    %845 = linalg.init_tensor [1, 9, 9, 512] : tensor<1x9x9x512xf32>
    %846 = linalg.fill ins(%cst_416 : f32) outs(%845 : tensor<1x9x9x512xf32>) -> tensor<1x9x9x512xf32>
    %847 = tensor.insert_slice %842 into %846[0, 1, 1, 0] [1, 7, 7, 512] [1, 1, 1, 1] : tensor<1x7x7x512xf32> into tensor<1x9x9x512xf32>
    %848 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%847, %cst_328 : tensor<1x9x9x512xf32>, tensor<3x3x512x512xf32>) outs(%844 : tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %849 = linalg.init_tensor [1, 7, 7, 512] : tensor<1x7x7x512xf32>
    %850 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%848, %cst_254 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%849 : tensor<1x7x7x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x512xf32>
    %851 = linalg.init_tensor [1, 7, 7, 512] : tensor<1x7x7x512xf32>
    %852 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%850, %cst_258 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%851 : tensor<1x7x7x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x512xf32>
    %853 = linalg.init_tensor [1, 7, 7, 512] : tensor<1x7x7x512xf32>
    %854 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%852, %cst_256 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%853 : tensor<1x7x7x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x512xf32>
    %855 = linalg.init_tensor [1, 7, 7, 512] : tensor<1x7x7x512xf32>
    %856 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%854, %cst_255 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%855 : tensor<1x7x7x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x512xf32>
    %857 = linalg.init_tensor [1, 7, 7, 512] : tensor<1x7x7x512xf32>
    %858 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%856, %cst_257 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%857 : tensor<1x7x7x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x512xf32>
    %859 = linalg.init_tensor [1, 7, 7, 512] : tensor<1x7x7x512xf32>
    %860 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%858, %cst_272 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%859 : tensor<1x7x7x512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x512xf32>
    %861 = linalg.init_tensor [1, 7, 7, 2048] : tensor<1x7x7x2048xf32>
    %cst_417 = arith.constant 0.000000e+00 : f32
    %862 = linalg.fill ins(%cst_417 : f32) outs(%861 : tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
    %863 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%860, %cst_329 : tensor<1x7x7x512xf32>, tensor<1x1x512x2048xf32>) outs(%862 : tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
    %864 = linalg.init_tensor [1, 7, 7, 2048] : tensor<1x7x7x2048xf32>
    %865 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%863, %cst_259 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%864 : tensor<1x7x7x2048xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x2048xf32>
    %866 = linalg.init_tensor [1, 7, 7, 2048] : tensor<1x7x7x2048xf32>
    %867 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%865, %cst_263 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%866 : tensor<1x7x7x2048xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x2048xf32>
    %868 = linalg.init_tensor [1, 7, 7, 2048] : tensor<1x7x7x2048xf32>
    %869 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%867, %cst_261 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%868 : tensor<1x7x7x2048xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x2048xf32>
    %870 = linalg.init_tensor [1, 7, 7, 2048] : tensor<1x7x7x2048xf32>
    %871 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%869, %cst_260 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%870 : tensor<1x7x7x2048xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x2048xf32>
    %872 = linalg.init_tensor [1, 7, 7, 2048] : tensor<1x7x7x2048xf32>
    %873 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%871, %cst_262 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%872 : tensor<1x7x7x2048xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x2048xf32>
    %874 = linalg.init_tensor [1, 7, 7, 2048] : tensor<1x7x7x2048xf32>
    %875 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%827, %873 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%874 : tensor<1x7x7x2048xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x2048xf32>
    %876 = linalg.init_tensor [1, 7, 7, 2048] : tensor<1x7x7x2048xf32>
    %877 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%875, %cst_273 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%876 : tensor<1x7x7x2048xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x7x7x2048xf32>
    %cst_418 = arith.constant 0.000000e+00 : f32
    %878 = linalg.init_tensor [1, 2048] : tensor<1x2048xf32>
    %879 = linalg.fill ins(%cst_418 : f32) outs(%878 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
    %880 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%877 : tensor<1x7x7x2048xf32>) outs(%879 : tensor<1x2048xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x2048xf32>
    %881 = linalg.init_tensor [1, 2048] : tensor<1x2048xf32>
    %882 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%880, %cst_274 : tensor<1x2048xf32>, tensor<1x2048xf32>) outs(%881 : tensor<1x2048xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x2048xf32>
    %883 = linalg.init_tensor [1, 1000] : tensor<1x1000xf32>
    %cst_419 = arith.constant 0.000000e+00 : f32
    %884 = linalg.fill ins(%cst_419 : f32) outs(%883 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %885 = linalg.matmul ins(%882, %cst_330 : tensor<1x2048xf32>, tensor<2048x1000xf32>) outs(%884 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %886 = linalg.init_tensor [1, 1000] : tensor<1x1000xf32>
    %887 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%885, %cst_264 : tensor<1x1000xf32>, tensor<1x1000xf32>) outs(%886 : tensor<1x1000xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x1000xf32>
    %cst_420 = arith.constant 0xFF800000 : f32
    %888 = linalg.init_tensor [1] : tensor<1xf32>
    %889 = linalg.fill ins(%cst_420 : f32) outs(%888 : tensor<1xf32>) -> tensor<1xf32>
    %890 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%887 : tensor<1x1000xf32>) outs(%889 : tensor<1xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %904 = arith.maxf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1xf32>
    %891 = linalg.init_tensor [1, 1000] : tensor<1x1000xf32>
    %892 = linalg.generic {indexing_maps = [#map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%890 : tensor<1xf32>) outs(%891 : tensor<1x1000xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<1x1000xf32>
    %893 = linalg.init_tensor [1, 1000] : tensor<1x1000xf32>
    %894 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%887, %892 : tensor<1x1000xf32>, tensor<1x1000xf32>) outs(%893 : tensor<1x1000xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.subf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x1000xf32>
    %895 = linalg.init_tensor [1, 1000] : tensor<1x1000xf32>
    %896 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%894 : tensor<1x1000xf32>) outs(%895 : tensor<1x1000xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %904 = math.exp %arg1 : f32
      linalg.yield %904 : f32
    } -> tensor<1x1000xf32>
    %cst_421 = arith.constant 0.000000e+00 : f32
    %897 = linalg.init_tensor [1] : tensor<1xf32>
    %898 = linalg.fill ins(%cst_421 : f32) outs(%897 : tensor<1xf32>) -> tensor<1xf32>
    %899 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%896 : tensor<1x1000xf32>) outs(%898 : tensor<1xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %904 = arith.addf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1xf32>
    %900 = linalg.init_tensor [1, 1000] : tensor<1x1000xf32>
    %901 = linalg.generic {indexing_maps = [#map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%899 : tensor<1xf32>) outs(%900 : tensor<1x1000xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<1x1000xf32>
    %902 = linalg.init_tensor [1, 1000] : tensor<1x1000xf32>
    %903 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%896, %901 : tensor<1x1000xf32>, tensor<1x1000xf32>) outs(%902 : tensor<1x1000xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %904 = arith.divf %arg1, %arg2 : f32
      linalg.yield %904 : f32
    } -> tensor<1x1000xf32>
    return %903 : tensor<1x1000xf32>
  }
}

