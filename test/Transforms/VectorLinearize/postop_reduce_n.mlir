// RUN: imex-opt %s -split-input-file -imex-vector-linearize -canonicalize | FileCheck %s
module {
  gpu.module @postop_reduce_n attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Bfloat16ConversionINTEL, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, VectorAnyINTEL], [SPV_INTEL_bfloat16_conversion, SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @postop_reduce_n(%arg0: memref<16384x12288xbf16>, %arg1: memref<1536x12288xbf16>, %arg2: memref<16384x4xf32>) kernel attributes {VectorComputeFunctionINTEL, gpu.known_block_size = array<i32: 8, 4, 1>, gpu.known_grid_size = array<i32: 8, 32, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant dense<0.000000e+00> : vector<8x1xf32>
      %c8 = arith.constant 8 : index
      %c256 = arith.constant 256 : index
      %c32 = arith.constant 32 : index
      %c12288 = arith.constant 12288 : index
      %c128 = arith.constant 128 : index
      %cst_0 = arith.constant dense<0.000000e+00> : vector<8x16xf32>
      %c3 = arith.constant 3 : index
      %c1 = arith.constant 1 : index
      %c2048 = arith.constant 2048 : index
      %c512 = arith.constant 512 : index
      %c21 = arith.constant 21 : index
      %cst_1 = arith.constant dense<0.000000e+00> : vector<32xf32>
      %cst_2 = arith.constant dense<0.000000e+00> : vector<8xf32>
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %0 = arith.muli %thread_id_x, %c4 : index
      %1 = arith.addi %0, %thread_id_y : index
      %2 = arith.muli %1, %c8 : index
      %3 = arith.muli %1, %c4 : index
      %block_dim_y = gpu.block_dim  y
      %4 = arith.muli %thread_id_x, %block_dim_y : index
      %5 = arith.addi %4, %thread_id_y : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %6 = arith.divsi %block_id_y, %c4 : index
      %7 = arith.remsi %block_id_y, %c4 : index
      %8 = arith.muli %5, %c8 : index
      %9 = arith.remsi %8, %c256 : index
      %10 = arith.divsi %5, %c4 : index
      %11 = arith.remsi %5, %c4 : index
      %12 = arith.muli %10, %c32 : index
      %13 = arith.remsi %12, %c256 : index
      %14 = arith.muli %11, %c12288 : index
      %15 = arith.remsi %14, %c12288 : index
      %16 = arith.muli %block_id_x, %c2048 : index
      %17 = arith.addi %13, %16 : index
      %18 = arith.muli %6, %c256 : index
      %19 = arith.addi %17, %18 : index
      %20 = arith.muli %11, %c32 : index
      %21 = arith.remsi %20, %c128 : index
      %22 = arith.muli %10, %c12288 : index
      %23 = arith.remsi %22, %c12288 : index
      %24 = arith.muli %7, %c128 : index
      %25 = arith.divsi %19, %c256 : index
      %26 = arith.muli %25, %c256 : index
      %27 = arith.divsi %15, %c32 : index
      %28 = arith.muli %27, %c32 : index
      %29 = arith.addi %19, %c0 : index
      %30 = arith.addi %15, %c0 : index
      %31 = xegpu.create_nd_tdesc %arg0[%29, %30] : memref<16384x12288xbf16> -> !xegpu.tensor_desc<32x16xbf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>
      %32 = arith.divsi %23, %c32 : index
      %33 = arith.muli %32, %c32 : index
      %34 = arith.addi %26, %2 : index
      %35 = arith.addi %34, %c0 : index
      %36 = arith.addi %28, %c0 : index
      %37 = xegpu.create_nd_tdesc %arg0[%35, %36] : memref<16384x12288xbf16> -> !xegpu.tensor_desc<8x32xbf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
      %38 = arith.remsi %11, %c4 : index
      %39 = scf.for %arg3 = %c0 to %c3 step %c1 iter_args(%arg4 = %cst) -> (vector<8x1xf32>) {

        //CHECK: vector.extract_strided_slice %{{.*}} {offsets = [0], sizes = [1], strides = [1]} : vector<8xf32> to vector<1xf32>
        //CHECK: vector.extract_strided_slice %{{.*}} {offsets = [1], sizes = [1], strides = [1]} : vector<8xf32> to vector<1xf32>
        //CHECK: vector.extract_strided_slice %{{.*}} {offsets = [2], sizes = [1], strides = [1]} : vector<8xf32> to vector<1xf32>
        //CHECK: vector.extract_strided_slice %{{.*}} {offsets = [3], sizes = [1], strides = [1]} : vector<8xf32> to vector<1xf32>
        //CHECK: vector.extract_strided_slice %{{.*}} {offsets = [4], sizes = [1], strides = [1]} : vector<8xf32> to vector<1xf32>
        //CHECK: vector.extract_strided_slice %{{.*}} {offsets = [5], sizes = [1], strides = [1]} : vector<8xf32> to vector<1xf32>
        //CHECK: vector.extract_strided_slice %{{.*}} {offsets = [6], sizes = [1], strides = [1]} : vector<8xf32> to vector<1xf32>
        //CHECK: vector.extract_strided_slice %{{.*}} {offsets = [7], sizes = [1], strides = [1]} : vector<8xf32> to vector<1xf32>
        %45 = vector.extract_strided_slice %arg4 {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]} : vector<8x1xf32> to vector<1x1xf32>
        %46 = vector.extract_strided_slice %arg4 {offsets = [1, 0], sizes = [1, 1], strides = [1, 1]} : vector<8x1xf32> to vector<1x1xf32>
        %47 = vector.extract_strided_slice %arg4 {offsets = [2, 0], sizes = [1, 1], strides = [1, 1]} : vector<8x1xf32> to vector<1x1xf32>
        %48 = vector.extract_strided_slice %arg4 {offsets = [3, 0], sizes = [1, 1], strides = [1, 1]} : vector<8x1xf32> to vector<1x1xf32>
        %49 = vector.extract_strided_slice %arg4 {offsets = [4, 0], sizes = [1, 1], strides = [1, 1]} : vector<8x1xf32> to vector<1x1xf32>
        %50 = vector.extract_strided_slice %arg4 {offsets = [5, 0], sizes = [1, 1], strides = [1, 1]} : vector<8x1xf32> to vector<1x1xf32>
        %51 = vector.extract_strided_slice %arg4 {offsets = [6, 0], sizes = [1, 1], strides = [1, 1]} : vector<8x1xf32> to vector<1x1xf32>
        %52 = vector.extract_strided_slice %arg4 {offsets = [7, 0], sizes = [1, 1], strides = [1, 1]} : vector<8x1xf32> to vector<1x1xf32>
        %53 = arith.muli %arg3, %c512 : index
        %54 = arith.addi %53, %21 : index
        %55 = arith.addi %54, %24 : index
        %56 = arith.divsi %55, %c128 : index
        %57 = arith.muli %56, %c128 : index
        %58 = arith.addi %55, %c0 : index
        %59 = arith.addi %23, %c0 : index
        %60 = xegpu.create_nd_tdesc %arg1[%58, %59] : memref<1536x12288xbf16> -> !xegpu.tensor_desc<32x16xbf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>
        xegpu.prefetch_nd %37 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x32xbf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
        %61 = xegpu.update_nd_offset %37, [%c0, %c32] : !xegpu.tensor_desc<8x32xbf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
        xegpu.prefetch_nd %61 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x32xbf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
        %62 = xegpu.update_nd_offset %61, [%c0, %c32] : !xegpu.tensor_desc<8x32xbf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
        xegpu.prefetch_nd %62 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x32xbf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
        %63 = xegpu.update_nd_offset %62, [%c0, %c32] : !xegpu.tensor_desc<8x32xbf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
        %64 = arith.addi %57, %3 : index
        %65 = arith.addi %64, %c0 : index
        %66 = arith.addi %33, %c0 : index
        %67 = xegpu.create_nd_tdesc %arg1[%65, %66] : memref<1536x12288xbf16> -> !xegpu.tensor_desc<4x32xbf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
        xegpu.prefetch_nd %67 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<4x32xbf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
        %68 = xegpu.update_nd_offset %67, [%c0, %c32] : !xegpu.tensor_desc<4x32xbf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
        xegpu.prefetch_nd %68 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<4x32xbf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
        %69 = xegpu.update_nd_offset %68, [%c0, %c32] : !xegpu.tensor_desc<4x32xbf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
        xegpu.prefetch_nd %69 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<4x32xbf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
        %70 = xegpu.update_nd_offset %69, [%c0, %c32] : !xegpu.tensor_desc<4x32xbf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
        %71:13 = scf.for %arg5 = %c0 to %c12288 step %c32 iter_args(%arg6 = %31, %arg7 = %60, %arg8 = %cst_0, %arg9 = %cst_0, %arg10 = %cst_0, %arg11 = %cst_0, %arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0, %arg16 = %63, %arg17 = %70, %arg18 = %c0) -> (!xegpu.tensor_desc<32x16xbf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xbf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, !xegpu.tensor_desc<8x32xbf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<4x32xbf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>, index) {
          %437 = arith.cmpi eq, %arg18, %c21 : index
          %438 = arith.select %437, %c0, %arg18 : index
          scf.if %437 {
            gpu.barrier
          }
          %439 = arith.addi %438, %c1 : index
          %440 = xegpu.load_nd %arg6 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<32x16xbf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>> -> vector<2x32x16xbf16>

          //CHECK: vector.shape_cast %{{.*}} : vector<2x32x16xbf16> to vector<1024xbf16>
          //CHECK: vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511] : vector<1024xbf16>, vector<1024xbf16>
          //CHECK: vector.shuffle %{{.*}}, %{{.*}} [512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023] : vector<1024xbf16>, vector<1024xbf16>
          //CHECK: vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<512xbf16>, vector<512xbf16>
          //CHECK: vector.shape_cast %{{.*}} : vector<128xbf16> to vector<8x16xbf16>
          //CHECK: vector.shuffle %{{.*}}, %{{.*}} [128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255] : vector<512xbf16>, vector<512xbf16>
          //CHECK: vector.shape_cast %{{.*}} : vector<128xbf16> to vector<8x16xbf16>
          //CHECK: vector.shuffle %{{.*}}, %{{.*}} [256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383] : vector<512xbf16>, vector<512xbf16>
          //CHECK: vector.shape_cast %{{.*}}  : vector<128xbf16> to vector<8x16xbf16>
          //CHECK: vector.shuffle %{{.*}}, %{{.*}} [384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511] : vector<512xbf16>, vector<512xbf16>
          //CHECK: vector.shape_cast %{{.*}} : vector<128xbf16> to vector<8x16xbf16>
          //CHECK: vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<512xbf16>, vector<512xbf16>
          //CHECK: vector.shape_cast %{{.*}} : vector<128xbf16> to vector<8x16xbf16>
          //CHECK: vector.shuffle %{{.*}}, %{{.*}} [128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255] : vector<512xbf16>, vector<512xbf16>
          //CHECK: vector.shape_cast %{{.*}} : vector<128xbf16> to vector<8x16xbf16>
          //CHECK: vector.shuffle %{{.*}}, %{{.*}} [256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383] : vector<512xbf16>, vector<512xbf16>
          //CHECK: vector.shape_cast %{{.*}} : vector<128xbf16> to vector<8x16xbf16>
          //CHECK: vector.shuffle %{{.*}}, %{{.*}} [384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511] : vector<512xbf16>, vector<512xbf16>
          //CHECK: vector.shape_cast %{{.*}} : vector<128xbf16> to vector<8x16xbf16>
          %441 = vector.extract %440[0] : vector<32x16xbf16> from vector<2x32x16xbf16>
          %442 = vector.extract %440[1] : vector<32x16xbf16> from vector<2x32x16xbf16>
          %443 = vector.extract_strided_slice %441 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xbf16> to vector<8x16xbf16>
          %444 = vector.extract_strided_slice %441 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xbf16> to vector<8x16xbf16>
          %445 = vector.extract_strided_slice %441 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xbf16> to vector<8x16xbf16>
          %446 = vector.extract_strided_slice %441 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xbf16> to vector<8x16xbf16>
          %447 = vector.extract_strided_slice %442 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xbf16> to vector<8x16xbf16>
          %448 = vector.extract_strided_slice %442 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xbf16> to vector<8x16xbf16>
          %449 = vector.extract_strided_slice %442 {offsets = [16, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xbf16> to vector<8x16xbf16>
          %450 = vector.extract_strided_slice %442 {offsets = [24, 0], sizes = [8, 16], strides = [1, 1]} : vector<32x16xbf16> to vector<8x16xbf16>
          %451 = xegpu.load_nd %arg7 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, packed}> : !xegpu.tensor_desc<32x16xbf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>> -> vector<2x16x16x2xbf16>

          //CHECK: vector.shape_cast %{{.*}} : vector<2x16x16x2xbf16> to vector<1024xbf16>
          //CHECK: vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511] : vector<1024xbf16>, vector<1024xbf16>
          //CHECK: vector.shuffle %{{.*}}, %{{.*}} [512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023] : vector<1024xbf16>, vector<1024xbf16>
          //CHECK: vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255] : vector<512xbf16>, vector<512xbf16>
          //CHECK: vector.shape_cast %{{.*}} : vector<256xbf16> to vector<8x16x2xbf16>
          //CHECK: vector.shuffle %{{.*}}, %{{.*}} [256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511] : vector<512xbf16>, vector<512xbf16>
          //CHECK: vector.shape_cast %{{.*}} : vector<256xbf16> to vector<8x16x2xbf16>
          //CHECK: vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255] : vector<512xbf16>, vector<512xbf16>
          //CHECK: vector.shape_cast %{{.*}} : vector<256xbf16> to vector<8x16x2xbf16>
          //CHECK: vector.shuffle %{{.*}}, %{{.*}} [256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511] : vector<512xbf16>, vector<512xbf16>
          //CHECK: vector.shape_cast %{{.*}} : vector<256xbf16> to vector<8x16x2xbf16>
          %452 = vector.extract %451[0] : vector<16x16x2xbf16> from vector<2x16x16x2xbf16>
          %453 = vector.extract %451[1] : vector<16x16x2xbf16> from vector<2x16x16x2xbf16>
          %454 = vector.extract_strided_slice %452 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16x2xbf16> to vector<8x16x2xbf16>
          %455 = vector.extract_strided_slice %452 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16x2xbf16> to vector<8x16x2xbf16>
          %456 = vector.extract_strided_slice %453 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16x2xbf16> to vector<8x16x2xbf16>
          %457 = vector.extract_strided_slice %453 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16x2xbf16> to vector<8x16x2xbf16>
          xegpu.compile_hint
          xegpu.prefetch_nd %arg16 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x32xbf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
          xegpu.prefetch_nd %arg17 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<4x32xbf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
          xegpu.compile_hint
          %458 = xegpu.update_nd_offset %arg16, [%c0, %c32] : !xegpu.tensor_desc<8x32xbf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
          %459 = xegpu.update_nd_offset %arg17, [%c0, %c32] : !xegpu.tensor_desc<4x32xbf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
          xegpu.compile_hint
          xegpu.compile_hint
          %460 = xegpu.update_nd_offset %arg6, [%c0, %c32] : !xegpu.tensor_desc<32x16xbf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>
          %461 = xegpu.update_nd_offset %arg7, [%c0, %c32] : !xegpu.tensor_desc<32x16xbf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>
          xegpu.compile_hint
          %462 = xegpu.dpas %443, %454, %arg8 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
          %463 = xegpu.dpas %447, %455, %462 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
          %464 = xegpu.dpas %443, %456, %arg9 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
          %465 = xegpu.dpas %447, %457, %464 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
          %466 = xegpu.dpas %444, %454, %arg10 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
          %467 = xegpu.dpas %448, %455, %466 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
          %468 = xegpu.dpas %444, %456, %arg11 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
          %469 = xegpu.dpas %448, %457, %468 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
          %470 = xegpu.dpas %445, %454, %arg12 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
          %471 = xegpu.dpas %449, %455, %470 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
          %472 = xegpu.dpas %445, %456, %arg13 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
          %473 = xegpu.dpas %449, %457, %472 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
          %474 = xegpu.dpas %446, %454, %arg14 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
          %475 = xegpu.dpas %450, %455, %474 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
          %476 = xegpu.dpas %446, %456, %arg15 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
          %477 = xegpu.dpas %450, %457, %476 : vector<8x16xbf16>, vector<8x16x2xbf16>, vector<8x16xf32> -> vector<8x16xf32>
          xegpu.compile_hint
          scf.yield %460, %461, %463, %465, %467, %469, %471, %473, %475, %477, %458, %459, %439 : !xegpu.tensor_desc<32x16xbf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>, !xegpu.tensor_desc<32x16xbf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 2 : i64, boundary_check = true>>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, !xegpu.tensor_desc<8x32xbf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>, !xegpu.tensor_desc<4x32xbf16, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>, index
        }

        //CHECK-COUNT-8: vector.shape_cast %{{.*}} : vector<8x16xf32> to vector<128xf32>
        //CHECK-COUNT-64: vector.shuffle {{.*}} : vector<128xf32>, vector<128xf32>
        %72 = vector.extract_strided_slice %71#2 {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %73 = vector.extract_strided_slice %71#2 {offsets = [1, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %74 = vector.extract_strided_slice %71#2 {offsets = [2, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %75 = vector.extract_strided_slice %71#2 {offsets = [3, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %76 = vector.extract_strided_slice %71#2 {offsets = [4, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %77 = vector.extract_strided_slice %71#2 {offsets = [5, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %78 = vector.extract_strided_slice %71#2 {offsets = [6, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %79 = vector.extract_strided_slice %71#2 {offsets = [7, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %80 = vector.extract_strided_slice %71#3 {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %81 = vector.extract_strided_slice %71#3 {offsets = [1, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %82 = vector.extract_strided_slice %71#3 {offsets = [2, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %83 = vector.extract_strided_slice %71#3 {offsets = [3, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %84 = vector.extract_strided_slice %71#3 {offsets = [4, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %85 = vector.extract_strided_slice %71#3 {offsets = [5, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %86 = vector.extract_strided_slice %71#3 {offsets = [6, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %87 = vector.extract_strided_slice %71#3 {offsets = [7, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %88 = vector.extract_strided_slice %71#4 {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %89 = vector.extract_strided_slice %71#4 {offsets = [1, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %90 = vector.extract_strided_slice %71#4 {offsets = [2, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %91 = vector.extract_strided_slice %71#4 {offsets = [3, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %92 = vector.extract_strided_slice %71#4 {offsets = [4, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %93 = vector.extract_strided_slice %71#4 {offsets = [5, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %94 = vector.extract_strided_slice %71#4 {offsets = [6, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %95 = vector.extract_strided_slice %71#4 {offsets = [7, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %96 = vector.extract_strided_slice %71#5 {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %97 = vector.extract_strided_slice %71#5 {offsets = [1, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %98 = vector.extract_strided_slice %71#5 {offsets = [2, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %99 = vector.extract_strided_slice %71#5 {offsets = [3, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %100 = vector.extract_strided_slice %71#5 {offsets = [4, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %101 = vector.extract_strided_slice %71#5 {offsets = [5, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %102 = vector.extract_strided_slice %71#5 {offsets = [6, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %103 = vector.extract_strided_slice %71#5 {offsets = [7, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %104 = vector.extract_strided_slice %71#6 {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %105 = vector.extract_strided_slice %71#6 {offsets = [1, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %106 = vector.extract_strided_slice %71#6 {offsets = [2, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %107 = vector.extract_strided_slice %71#6 {offsets = [3, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %108 = vector.extract_strided_slice %71#6 {offsets = [4, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %109 = vector.extract_strided_slice %71#6 {offsets = [5, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %110 = vector.extract_strided_slice %71#6 {offsets = [6, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %111 = vector.extract_strided_slice %71#6 {offsets = [7, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %112 = vector.extract_strided_slice %71#7 {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %113 = vector.extract_strided_slice %71#7 {offsets = [1, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %114 = vector.extract_strided_slice %71#7 {offsets = [2, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %115 = vector.extract_strided_slice %71#7 {offsets = [3, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %116 = vector.extract_strided_slice %71#7 {offsets = [4, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %117 = vector.extract_strided_slice %71#7 {offsets = [5, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %118 = vector.extract_strided_slice %71#7 {offsets = [6, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %119 = vector.extract_strided_slice %71#7 {offsets = [7, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %120 = vector.extract_strided_slice %71#8 {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %121 = vector.extract_strided_slice %71#8 {offsets = [1, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %122 = vector.extract_strided_slice %71#8 {offsets = [2, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %123 = vector.extract_strided_slice %71#8 {offsets = [3, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %124 = vector.extract_strided_slice %71#8 {offsets = [4, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %125 = vector.extract_strided_slice %71#8 {offsets = [5, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %126 = vector.extract_strided_slice %71#8 {offsets = [6, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %127 = vector.extract_strided_slice %71#8 {offsets = [7, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %128 = vector.extract_strided_slice %71#9 {offsets = [0, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %129 = vector.extract_strided_slice %71#9 {offsets = [1, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %130 = vector.extract_strided_slice %71#9 {offsets = [2, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %131 = vector.extract_strided_slice %71#9 {offsets = [3, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %132 = vector.extract_strided_slice %71#9 {offsets = [4, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %133 = vector.extract_strided_slice %71#9 {offsets = [5, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %134 = vector.extract_strided_slice %71#9 {offsets = [6, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        %135 = vector.extract_strided_slice %71#9 {offsets = [7, 0], sizes = [1, 16], strides = [1, 1]} : vector<8x16xf32> to vector<1x16xf32>
        // CHECK-COUNT-64: math.exp %{{.*}} : vector<16xf32>
        %136 = math.exp %72 : vector<1x16xf32>
        %137 = math.exp %80 : vector<1x16xf32>
        %138 = math.exp %73 : vector<1x16xf32>
        %139 = math.exp %81 : vector<1x16xf32>
        %140 = math.exp %74 : vector<1x16xf32>
        %141 = math.exp %82 : vector<1x16xf32>
        %142 = math.exp %75 : vector<1x16xf32>
        %143 = math.exp %83 : vector<1x16xf32>
        %144 = math.exp %76 : vector<1x16xf32>
        %145 = math.exp %84 : vector<1x16xf32>
        %146 = math.exp %77 : vector<1x16xf32>
        %147 = math.exp %85 : vector<1x16xf32>
        %148 = math.exp %78 : vector<1x16xf32>
        %149 = math.exp %86 : vector<1x16xf32>
        %150 = math.exp %79 : vector<1x16xf32>
        %151 = math.exp %87 : vector<1x16xf32>
        %152 = math.exp %88 : vector<1x16xf32>
        %153 = math.exp %96 : vector<1x16xf32>
        %154 = math.exp %89 : vector<1x16xf32>
        %155 = math.exp %97 : vector<1x16xf32>
        %156 = math.exp %90 : vector<1x16xf32>
        %157 = math.exp %98 : vector<1x16xf32>
        %158 = math.exp %91 : vector<1x16xf32>
        %159 = math.exp %99 : vector<1x16xf32>
        %160 = math.exp %92 : vector<1x16xf32>
        %161 = math.exp %100 : vector<1x16xf32>
        %162 = math.exp %93 : vector<1x16xf32>
        %163 = math.exp %101 : vector<1x16xf32>
        %164 = math.exp %94 : vector<1x16xf32>
        %165 = math.exp %102 : vector<1x16xf32>
        %166 = math.exp %95 : vector<1x16xf32>
        %167 = math.exp %103 : vector<1x16xf32>
        %168 = math.exp %104 : vector<1x16xf32>
        %169 = math.exp %112 : vector<1x16xf32>
        %170 = math.exp %105 : vector<1x16xf32>
        %171 = math.exp %113 : vector<1x16xf32>
        %172 = math.exp %106 : vector<1x16xf32>
        %173 = math.exp %114 : vector<1x16xf32>
        %174 = math.exp %107 : vector<1x16xf32>
        %175 = math.exp %115 : vector<1x16xf32>
        %176 = math.exp %108 : vector<1x16xf32>
        %177 = math.exp %116 : vector<1x16xf32>
        %178 = math.exp %109 : vector<1x16xf32>
        %179 = math.exp %117 : vector<1x16xf32>
        %180 = math.exp %110 : vector<1x16xf32>
        %181 = math.exp %118 : vector<1x16xf32>
        %182 = math.exp %111 : vector<1x16xf32>
        %183 = math.exp %119 : vector<1x16xf32>
        %184 = math.exp %120 : vector<1x16xf32>
        %185 = math.exp %128 : vector<1x16xf32>
        %186 = math.exp %121 : vector<1x16xf32>
        %187 = math.exp %129 : vector<1x16xf32>
        %188 = math.exp %122 : vector<1x16xf32>
        %189 = math.exp %130 : vector<1x16xf32>
        %190 = math.exp %123 : vector<1x16xf32>
        %191 = math.exp %131 : vector<1x16xf32>
        %192 = math.exp %124 : vector<1x16xf32>
        %193 = math.exp %132 : vector<1x16xf32>
        %194 = math.exp %125 : vector<1x16xf32>
        %195 = math.exp %133 : vector<1x16xf32>
        %196 = math.exp %126 : vector<1x16xf32>
        %197 = math.exp %134 : vector<1x16xf32>
        %198 = math.exp %127 : vector<1x16xf32>
        %199 = math.exp %135 : vector<1x16xf32>

        //CHECK-COUNT-32: arith.addf %{{.*}}, %{{.*}} : vector<16xf32>
        %200 = arith.addf %136, %137 : vector<1x16xf32>
        %201 = vector.shape_cast %200 : vector<1x16xf32> to vector<16xf32>
        %202 = arith.addf %138, %139 : vector<1x16xf32>
        %203 = vector.shape_cast %202 : vector<1x16xf32> to vector<16xf32>
        %204 = arith.addf %140, %141 : vector<1x16xf32>
        %205 = vector.shape_cast %204 : vector<1x16xf32> to vector<16xf32>
        %206 = arith.addf %142, %143 : vector<1x16xf32>
        %207 = vector.shape_cast %206 : vector<1x16xf32> to vector<16xf32>
        %208 = arith.addf %144, %145 : vector<1x16xf32>
        %209 = vector.shape_cast %208 : vector<1x16xf32> to vector<16xf32>
        %210 = arith.addf %146, %147 : vector<1x16xf32>
        %211 = vector.shape_cast %210 : vector<1x16xf32> to vector<16xf32>
        %212 = arith.addf %148, %149 : vector<1x16xf32>
        %213 = vector.shape_cast %212 : vector<1x16xf32> to vector<16xf32>
        %214 = arith.addf %150, %151 : vector<1x16xf32>
        %215 = vector.shape_cast %214 : vector<1x16xf32> to vector<16xf32>
        %216 = arith.addf %152, %153 : vector<1x16xf32>
        %217 = vector.shape_cast %216 : vector<1x16xf32> to vector<16xf32>
        %218 = arith.addf %154, %155 : vector<1x16xf32>
        %219 = vector.shape_cast %218 : vector<1x16xf32> to vector<16xf32>
        %220 = arith.addf %156, %157 : vector<1x16xf32>
        %221 = vector.shape_cast %220 : vector<1x16xf32> to vector<16xf32>
        %222 = arith.addf %158, %159 : vector<1x16xf32>
        %223 = vector.shape_cast %222 : vector<1x16xf32> to vector<16xf32>
        %224 = arith.addf %160, %161 : vector<1x16xf32>
        %225 = vector.shape_cast %224 : vector<1x16xf32> to vector<16xf32>
        %226 = arith.addf %162, %163 : vector<1x16xf32>
        %227 = vector.shape_cast %226 : vector<1x16xf32> to vector<16xf32>
        %228 = arith.addf %164, %165 : vector<1x16xf32>
        %229 = vector.shape_cast %228 : vector<1x16xf32> to vector<16xf32>
        %230 = arith.addf %166, %167 : vector<1x16xf32>
        %231 = vector.shape_cast %230 : vector<1x16xf32> to vector<16xf32>
        %232 = arith.addf %168, %169 : vector<1x16xf32>
        %233 = vector.shape_cast %232 : vector<1x16xf32> to vector<16xf32>
        %234 = arith.addf %170, %171 : vector<1x16xf32>
        %235 = vector.shape_cast %234 : vector<1x16xf32> to vector<16xf32>
        %236 = arith.addf %172, %173 : vector<1x16xf32>
        %237 = vector.shape_cast %236 : vector<1x16xf32> to vector<16xf32>
        %238 = arith.addf %174, %175 : vector<1x16xf32>
        %239 = vector.shape_cast %238 : vector<1x16xf32> to vector<16xf32>
        %240 = arith.addf %176, %177 : vector<1x16xf32>
        %241 = vector.shape_cast %240 : vector<1x16xf32> to vector<16xf32>
        %242 = arith.addf %178, %179 : vector<1x16xf32>
        %243 = vector.shape_cast %242 : vector<1x16xf32> to vector<16xf32>
        %244 = arith.addf %180, %181 : vector<1x16xf32>
        %245 = vector.shape_cast %244 : vector<1x16xf32> to vector<16xf32>
        %246 = arith.addf %182, %183 : vector<1x16xf32>
        %247 = vector.shape_cast %246 : vector<1x16xf32> to vector<16xf32>
        %248 = arith.addf %184, %185 : vector<1x16xf32>
        %249 = vector.shape_cast %248 : vector<1x16xf32> to vector<16xf32>
        %250 = arith.addf %186, %187 : vector<1x16xf32>
        %251 = vector.shape_cast %250 : vector<1x16xf32> to vector<16xf32>
        %252 = arith.addf %188, %189 : vector<1x16xf32>
        %253 = vector.shape_cast %252 : vector<1x16xf32> to vector<16xf32>
        %254 = arith.addf %190, %191 : vector<1x16xf32>
        %255 = vector.shape_cast %254 : vector<1x16xf32> to vector<16xf32>
        %256 = arith.addf %192, %193 : vector<1x16xf32>
        %257 = vector.shape_cast %256 : vector<1x16xf32> to vector<16xf32>
        %258 = arith.addf %194, %195 : vector<1x16xf32>
        %259 = vector.shape_cast %258 : vector<1x16xf32> to vector<16xf32>
        %260 = arith.addf %196, %197 : vector<1x16xf32>
        %261 = vector.shape_cast %260 : vector<1x16xf32> to vector<16xf32>
        %262 = arith.addf %198, %199 : vector<1x16xf32>
        %263 = vector.shape_cast %262 : vector<1x16xf32> to vector<16xf32>
        %264 = vector.shuffle %201, %203 [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf32>, vector<16xf32>
        %265 = vector.shuffle %201, %203 [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
        %266 = arith.addf %264, %265 : vector<16xf32>
        %267 = vector.shuffle %205, %207 [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf32>, vector<16xf32>
        %268 = vector.shuffle %205, %207 [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
        %269 = arith.addf %267, %268 : vector<16xf32>
        %270 = vector.shuffle %209, %211 [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf32>, vector<16xf32>
        %271 = vector.shuffle %209, %211 [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
        %272 = arith.addf %270, %271 : vector<16xf32>
        %273 = vector.shuffle %213, %215 [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf32>, vector<16xf32>
        %274 = vector.shuffle %213, %215 [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
        %275 = arith.addf %273, %274 : vector<16xf32>
        %276 = vector.shuffle %217, %219 [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf32>, vector<16xf32>
        %277 = vector.shuffle %217, %219 [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
        %278 = arith.addf %276, %277 : vector<16xf32>
        %279 = vector.shuffle %221, %223 [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf32>, vector<16xf32>
        %280 = vector.shuffle %221, %223 [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
        %281 = arith.addf %279, %280 : vector<16xf32>
        %282 = vector.shuffle %225, %227 [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf32>, vector<16xf32>
        %283 = vector.shuffle %225, %227 [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
        %284 = arith.addf %282, %283 : vector<16xf32>
        %285 = vector.shuffle %229, %231 [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf32>, vector<16xf32>
        %286 = vector.shuffle %229, %231 [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
        %287 = arith.addf %285, %286 : vector<16xf32>
        %288 = vector.shuffle %233, %235 [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf32>, vector<16xf32>
        %289 = vector.shuffle %233, %235 [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
        %290 = arith.addf %288, %289 : vector<16xf32>
        %291 = vector.shuffle %237, %239 [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf32>, vector<16xf32>
        %292 = vector.shuffle %237, %239 [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
        %293 = arith.addf %291, %292 : vector<16xf32>
        %294 = vector.shuffle %241, %243 [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf32>, vector<16xf32>
        %295 = vector.shuffle %241, %243 [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
        %296 = arith.addf %294, %295 : vector<16xf32>
        %297 = vector.shuffle %245, %247 [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf32>, vector<16xf32>
        %298 = vector.shuffle %245, %247 [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
        %299 = arith.addf %297, %298 : vector<16xf32>
        %300 = vector.shuffle %249, %251 [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf32>, vector<16xf32>
        %301 = vector.shuffle %249, %251 [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
        %302 = arith.addf %300, %301 : vector<16xf32>
        %303 = vector.shuffle %253, %255 [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf32>, vector<16xf32>
        %304 = vector.shuffle %253, %255 [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
        %305 = arith.addf %303, %304 : vector<16xf32>
        %306 = vector.shuffle %257, %259 [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf32>, vector<16xf32>
        %307 = vector.shuffle %257, %259 [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
        %308 = arith.addf %306, %307 : vector<16xf32>
        %309 = vector.shuffle %261, %263 [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xf32>, vector<16xf32>
        %310 = vector.shuffle %261, %263 [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
        %311 = arith.addf %309, %310 : vector<16xf32>
        %312 = vector.shuffle %266, %269 [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
        %313 = vector.shuffle %266, %269 [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
        %314 = arith.addf %312, %313 : vector<16xf32>
        %315 = vector.shuffle %272, %275 [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
        %316 = vector.shuffle %272, %275 [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
        %317 = arith.addf %315, %316 : vector<16xf32>
        %318 = vector.shuffle %278, %281 [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
        %319 = vector.shuffle %278, %281 [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
        %320 = arith.addf %318, %319 : vector<16xf32>
        %321 = vector.shuffle %284, %287 [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
        %322 = vector.shuffle %284, %287 [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
        %323 = arith.addf %321, %322 : vector<16xf32>
        %324 = vector.shuffle %290, %293 [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
        %325 = vector.shuffle %290, %293 [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
        %326 = arith.addf %324, %325 : vector<16xf32>
        %327 = vector.shuffle %296, %299 [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
        %328 = vector.shuffle %296, %299 [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
        %329 = arith.addf %327, %328 : vector<16xf32>
        %330 = vector.shuffle %302, %305 [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
        %331 = vector.shuffle %302, %305 [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
        %332 = arith.addf %330, %331 : vector<16xf32>
        %333 = vector.shuffle %308, %311 [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
        %334 = vector.shuffle %308, %311 [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
        %335 = arith.addf %333, %334 : vector<16xf32>
        %336 = vector.shuffle %314, %317 [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29] : vector<16xf32>, vector<16xf32>
        %337 = vector.shuffle %314, %317 [2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31] : vector<16xf32>, vector<16xf32>
        %338 = arith.addf %336, %337 : vector<16xf32>
        %339 = vector.shuffle %320, %323 [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29] : vector<16xf32>, vector<16xf32>
        %340 = vector.shuffle %320, %323 [2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31] : vector<16xf32>, vector<16xf32>
        %341 = arith.addf %339, %340 : vector<16xf32>
        %342 = vector.shuffle %326, %329 [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29] : vector<16xf32>, vector<16xf32>
        %343 = vector.shuffle %326, %329 [2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31] : vector<16xf32>, vector<16xf32>
        %344 = arith.addf %342, %343 : vector<16xf32>
        %345 = vector.shuffle %332, %335 [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29] : vector<16xf32>, vector<16xf32>
        %346 = vector.shuffle %332, %335 [2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31] : vector<16xf32>, vector<16xf32>
        %347 = arith.addf %345, %346 : vector<16xf32>
        %348 = vector.shuffle %338, %341 [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30] : vector<16xf32>, vector<16xf32>
        %349 = vector.shuffle %338, %341 [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31] : vector<16xf32>, vector<16xf32>
        %350 = arith.addf %348, %349 : vector<16xf32>
        %351 = vector.shuffle %344, %347 [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30] : vector<16xf32>, vector<16xf32>
        %352 = vector.shuffle %344, %347 [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31] : vector<16xf32>, vector<16xf32>
        %353 = arith.addf %351, %352 : vector<16xf32>
        %354 = vector.shape_cast %350 : vector<16xf32> to vector<16xf32>
        %355 = vector.shape_cast %353 : vector<16xf32> to vector<16xf32>
        %356 = vector.shuffle %354, %355 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
        %357 = vector.shape_cast %356 : vector<32xf32> to vector<32xf32>
        %358 = vector.shape_cast %357 : vector<32xf32> to vector<32x1xf32>
        %alloc = memref.alloc() : memref<256x4xf32, #spirv.storage_class<Workgroup>>
        %359 = arith.addi %13, %c0 : index
        %360 = arith.addi %38, %c0 : index
        %361 = xegpu.create_nd_tdesc %alloc[%359, %360] : memref<256x4xf32, #spirv.storage_class<Workgroup>> -> !xegpu.tensor_desc<8x1xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
        %362 = arith.addi %13, %c8 : index
        %363 = xegpu.create_nd_tdesc %alloc[%362, %360] : memref<256x4xf32, #spirv.storage_class<Workgroup>> -> !xegpu.tensor_desc<8x1xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
        %c16 = arith.constant 16 : index
        %364 = arith.addi %13, %c16 : index
        %365 = xegpu.create_nd_tdesc %alloc[%364, %360] : memref<256x4xf32, #spirv.storage_class<Workgroup>> -> !xegpu.tensor_desc<8x1xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
        %c24 = arith.constant 24 : index
        %366 = arith.addi %13, %c24 : index
        %367 = xegpu.create_nd_tdesc %alloc[%366, %360] : memref<256x4xf32, #spirv.storage_class<Workgroup>> -> !xegpu.tensor_desc<8x1xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
        %368 = vector.extract_strided_slice %358 {offsets = [0, 0], sizes = [8, 1], strides = [1, 1]} : vector<32x1xf32> to vector<8x1xf32>
        %369 = vector.extract_strided_slice %358 {offsets = [8, 0], sizes = [8, 1], strides = [1, 1]} : vector<32x1xf32> to vector<8x1xf32>
        %370 = vector.extract_strided_slice %358 {offsets = [16, 0], sizes = [8, 1], strides = [1, 1]} : vector<32x1xf32> to vector<8x1xf32>
        %371 = vector.extract_strided_slice %358 {offsets = [24, 0], sizes = [8, 1], strides = [1, 1]} : vector<32x1xf32> to vector<8x1xf32>
        xegpu.store_nd %368, %361 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x1xf32>, !xegpu.tensor_desc<8x1xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
        xegpu.store_nd %369, %363 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x1xf32>, !xegpu.tensor_desc<8x1xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
        xegpu.store_nd %370, %365 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x1xf32>, !xegpu.tensor_desc<8x1xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
        xegpu.store_nd %371, %367 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x1xf32>, !xegpu.tensor_desc<8x1xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
        gpu.barrier
        %372 = arith.addi %9, %c0 : index
        %373 = xegpu.create_nd_tdesc %alloc[%372, %c0] : memref<256x4xf32, #spirv.storage_class<Workgroup>> -> !xegpu.tensor_desc<8x4xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
        %374 = xegpu.load_nd %373 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>}> : !xegpu.tensor_desc<8x4xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>> -> vector<8x4xf32>
        %375 = vector.extract_strided_slice %374 {offsets = [0, 0], sizes = [1, 4], strides = [1, 1]} : vector<8x4xf32> to vector<1x4xf32>
        %376 = vector.extract_strided_slice %374 {offsets = [1, 0], sizes = [1, 4], strides = [1, 1]} : vector<8x4xf32> to vector<1x4xf32>
        %377 = vector.extract_strided_slice %374 {offsets = [2, 0], sizes = [1, 4], strides = [1, 1]} : vector<8x4xf32> to vector<1x4xf32>
        %378 = vector.extract_strided_slice %374 {offsets = [3, 0], sizes = [1, 4], strides = [1, 1]} : vector<8x4xf32> to vector<1x4xf32>
        %379 = vector.extract_strided_slice %374 {offsets = [4, 0], sizes = [1, 4], strides = [1, 1]} : vector<8x4xf32> to vector<1x4xf32>
        %380 = vector.extract_strided_slice %374 {offsets = [5, 0], sizes = [1, 4], strides = [1, 1]} : vector<8x4xf32> to vector<1x4xf32>
        %381 = vector.extract_strided_slice %374 {offsets = [6, 0], sizes = [1, 4], strides = [1, 1]} : vector<8x4xf32> to vector<1x4xf32>
        %382 = vector.extract_strided_slice %374 {offsets = [7, 0], sizes = [1, 4], strides = [1, 1]} : vector<8x4xf32> to vector<1x4xf32>
        %383 = vector.shape_cast %375 : vector<1x4xf32> to vector<4xf32>
        %384 = vector.shape_cast %376 : vector<1x4xf32> to vector<4xf32>
        %385 = vector.shape_cast %377 : vector<1x4xf32> to vector<4xf32>
        %386 = vector.shape_cast %378 : vector<1x4xf32> to vector<4xf32>
        %387 = vector.shape_cast %379 : vector<1x4xf32> to vector<4xf32>
        %388 = vector.shape_cast %380 : vector<1x4xf32> to vector<4xf32>
        %389 = vector.shape_cast %381 : vector<1x4xf32> to vector<4xf32>
        %390 = vector.shape_cast %382 : vector<1x4xf32> to vector<4xf32>
        %391 = vector.shuffle %383, %384 [0, 1, 4, 5] : vector<4xf32>, vector<4xf32>
        %392 = vector.shuffle %383, %384 [2, 3, 6, 7] : vector<4xf32>, vector<4xf32>
        %393 = arith.addf %391, %392 : vector<4xf32>
        %394 = vector.shuffle %385, %386 [0, 1, 4, 5] : vector<4xf32>, vector<4xf32>
        %395 = vector.shuffle %385, %386 [2, 3, 6, 7] : vector<4xf32>, vector<4xf32>
        %396 = arith.addf %394, %395 : vector<4xf32>
        %397 = vector.shuffle %387, %388 [0, 1, 4, 5] : vector<4xf32>, vector<4xf32>
        %398 = vector.shuffle %387, %388 [2, 3, 6, 7] : vector<4xf32>, vector<4xf32>
        %399 = arith.addf %397, %398 : vector<4xf32>
        %400 = vector.shuffle %389, %390 [0, 1, 4, 5] : vector<4xf32>, vector<4xf32>
        %401 = vector.shuffle %389, %390 [2, 3, 6, 7] : vector<4xf32>, vector<4xf32>
        %402 = arith.addf %400, %401 : vector<4xf32>
        %403 = vector.shuffle %393, %396 [0, 2, 4, 6] : vector<4xf32>, vector<4xf32>
        %404 = vector.shuffle %393, %396 [1, 3, 5, 7] : vector<4xf32>, vector<4xf32>
        %405 = arith.addf %403, %404 : vector<4xf32>
        %406 = vector.shuffle %399, %402 [0, 2, 4, 6] : vector<4xf32>, vector<4xf32>
        %407 = vector.shuffle %399, %402 [1, 3, 5, 7] : vector<4xf32>, vector<4xf32>
        %408 = arith.addf %406, %407 : vector<4xf32>
        %409 = vector.shape_cast %405 : vector<4xf32> to vector<4xf32>
        %410 = vector.shape_cast %408 : vector<4xf32> to vector<4xf32>
        %411 = vector.shuffle %409, %410 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
        %412 = vector.shape_cast %411 : vector<8xf32> to vector<8xf32>
        %413 = vector.shape_cast %412 : vector<8xf32> to vector<8x1xf32>

        //CHECK: vector.extract_strided_slice %{{.*}} {offsets = [0], sizes = [1], strides = [1]} : vector<8xf32> to vector<1xf32>
        //CHECK: vector.extract_strided_slice %{{.*}} {offsets = [1], sizes = [1], strides = [1]} : vector<8xf32> to vector<1xf32>
        //CHECK: vector.extract_strided_slice %{{.*}} {offsets = [2], sizes = [1], strides = [1]} : vector<8xf32> to vector<1xf32>
        //CHECK: vector.extract_strided_slice %{{.*}} {offsets = [3], sizes = [1], strides = [1]} : vector<8xf32> to vector<1xf32>
        //CHECK: vector.extract_strided_slice %{{.*}} {offsets = [4], sizes = [1], strides = [1]} : vector<8xf32> to vector<1xf32>
        //CHECK: vector.extract_strided_slice %{{.*}} {offsets = [5], sizes = [1], strides = [1]} : vector<8xf32> to vector<1xf32>
        //CHECK: vector.extract_strided_slice %{{.*}} {offsets = [6], sizes = [1], strides = [1]} : vector<8xf32> to vector<1xf32>
        //CHECK: vector.extract_strided_slice %{{.*}} {offsets = [7], sizes = [1], strides = [1]} : vector<8xf32> to vector<1xf32>
        %414 = vector.extract_strided_slice %413 {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]} : vector<8x1xf32> to vector<1x1xf32>
        %415 = vector.extract_strided_slice %413 {offsets = [1, 0], sizes = [1, 1], strides = [1, 1]} : vector<8x1xf32> to vector<1x1xf32>
        %416 = vector.extract_strided_slice %413 {offsets = [2, 0], sizes = [1, 1], strides = [1, 1]} : vector<8x1xf32> to vector<1x1xf32>
        %417 = vector.extract_strided_slice %413 {offsets = [3, 0], sizes = [1, 1], strides = [1, 1]} : vector<8x1xf32> to vector<1x1xf32>
        %418 = vector.extract_strided_slice %413 {offsets = [4, 0], sizes = [1, 1], strides = [1, 1]} : vector<8x1xf32> to vector<1x1xf32>
        %419 = vector.extract_strided_slice %413 {offsets = [5, 0], sizes = [1, 1], strides = [1, 1]} : vector<8x1xf32> to vector<1x1xf32>
        %420 = vector.extract_strided_slice %413 {offsets = [6, 0], sizes = [1, 1], strides = [1, 1]} : vector<8x1xf32> to vector<1x1xf32>
        %421 = vector.extract_strided_slice %413 {offsets = [7, 0], sizes = [1, 1], strides = [1, 1]} : vector<8x1xf32> to vector<1x1xf32>
        //CHECK-COUNT-8: arith.addf %{{.*}}, %{{.*}} : vector<1xf32>
        %422 = arith.addf %45, %414 : vector<1x1xf32>
        %423 = arith.addf %46, %415 : vector<1x1xf32>
        %424 = arith.addf %47, %416 : vector<1x1xf32>
        %425 = arith.addf %48, %417 : vector<1x1xf32>
        %426 = arith.addf %49, %418 : vector<1x1xf32>
        %427 = arith.addf %50, %419 : vector<1x1xf32>
        %428 = arith.addf %51, %420 : vector<1x1xf32>
        %429 = arith.addf %52, %421 : vector<1x1xf32>
        //CHECK-COUNT-4: vector.interleave %{{.*}}, %{{.*}} : vector<1xf32>
        %430 = vector.shuffle %422, %423 [0, 1] : vector<1x1xf32>, vector<1x1xf32>
        %431 = vector.shuffle %424, %425 [0, 1] : vector<1x1xf32>, vector<1x1xf32>
        %432 = vector.shuffle %426, %427 [0, 1] : vector<1x1xf32>, vector<1x1xf32>
        %433 = vector.shuffle %428, %429 [0, 1] : vector<1x1xf32>, vector<1x1xf32>
        //CHECK-COUNT-2: vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3] : vector<2xf32>, vector<2xf32>
        %434 = vector.shuffle %430, %431 [0, 1, 2, 3] : vector<2x1xf32>, vector<2x1xf32>
        %435 = vector.shuffle %432, %433 [0, 1, 2, 3] : vector<2x1xf32>, vector<2x1xf32>
        //CHECK: vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
        %436 = vector.shuffle %434, %435 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4x1xf32>, vector<4x1xf32>
        scf.yield %436 : vector<8x1xf32>
      }
      %40 = arith.addi %16, %18 : index
      %41 = arith.addi %40, %9 : index
      %42 = arith.addi %41, %c0 : index
      %43 = arith.addi %7, %c0 : index
      %44 = xegpu.create_nd_tdesc %arg2[%42, %43] : memref<16384x4xf32> -> !xegpu.tensor_desc<8x1xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
      xegpu.store_nd %39, %44 <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<write_back>, l3_hint = #xegpu.cache_hint<write_back>}> : vector<8x1xf32>, !xegpu.tensor_desc<8x1xf32, #xegpu.block_tdesc_attr<memory_scope =  global, array_length = 1 : i64, boundary_check = true>>
      gpu.return
    }
  }
}
