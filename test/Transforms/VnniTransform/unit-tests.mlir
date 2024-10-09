// RUN: imex-opt %s -split-input-file -imex-xegpu-apply-vnni-transformation | FileCheck %s

// CHECK-LABEL: @test
//  CHECK-SAME: (%[[ARG1:.*]]: vector<8x16xf16>, %[[ARG2:.*]]: vector<16x16xf16>)
//       CHECK:  %[[B1:.*]] = vector.shape_cast %[[ARG2]] {packed} : vector<16x16xf16> to vector<256xf16>
//       CHECK:  %[[B2:.*]] = vector.shuffle %[[B1]], %[[B1]]
// CHECK: [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31,
// CHECK: 32, 48, 33, 49, 34, 50, 35, 51, 36, 52, 37, 53, 38, 54, 39, 55, 40, 56, 41, 57, 42, 58, 43, 59, 44, 60, 45, 61, 46, 62, 47, 63,
// CHECK: 64, 80, 65, 81, 66, 82, 67, 83, 68, 84, 69, 85, 70, 86, 71, 87, 72, 88, 73, 89, 74, 90, 75, 91, 76, 92, 77, 93, 78, 94, 79, 95,
// CHECK: 96, 112, 97, 113, 98, 114, 99, 115, 100, 116, 101, 117, 102, 118, 103, 119, 104, 120, 105, 121, 106, 122, 107, 123, 108, 124, 109, 125, 110, 126, 111, 127,
// CHECK: 128, 144, 129, 145, 130, 146, 131, 147, 132, 148, 133, 149, 134, 150, 135, 151, 136, 152, 137, 153, 138, 154, 139, 155, 140, 156, 141, 157, 142, 158, 143, 159,
// CHECK: 160, 176, 161, 177, 162, 178, 163, 179, 164, 180, 165, 181, 166, 182, 167, 183, 168, 184, 169, 185, 170, 186, 171, 187, 172, 188, 173, 189, 174, 190, 175, 191,
// CHECK: 192, 208, 193, 209, 194, 210, 195, 211, 196, 212, 197, 213, 198, 214, 199, 215, 200, 216, 201, 217, 202, 218, 203, 219, 204, 220, 205, 221, 206, 222, 207, 223,
// CHECK: 224, 240, 225, 241, 226, 242, 227, 243, 228, 244, 229, 245, 230, 246, 231, 247, 232, 248, 233, 249, 234, 250, 235, 251, 236, 252, 237, 253, 238, 254, 239, 255] {packed} : vector<256xf16>, vector<256xf16>
//       CHECK:  %[[B3:.*]] = vector.shape_cast %[[B2]] {packed} : vector<256xf16> to vector<8x16x2xf16>
//       CHECK:  %[[RES:.*]] = xegpu.dpas %[[ARG1]], %[[B3]] : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
//       CHECK:  return %[[RES]]

func.func @test(%arg1 : vector<8x16xf16>, %arg2 : vector<16x16xf16>) -> vector<8x16xf32> {
  %0 = xegpu.dpas %arg1, %arg2 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  return %0 : vector<8x16xf32>
}

// -----

// CHECK-LABEL: @test
//  CHECK-SAME: (%[[ARG1:.*]]: vector<8x32xi8>, %[[ARG2:.*]]: vector<32x32xi8>) -> vector<8x32xi32> {
//       CHECK: %[[B1:.*]] = vector.shape_cast %[[ARG2]] {packed} : vector<32x32xi8> to vector<1024xi8>
//       CHECK: %[[B2:.*]] = vector.shuffle %[[B1]], %[[B1]]
// CHECK: [0, 32, 64, 96, 1, 33, 65, 97, 2, 34, 66, 98, 3, 35, 67, 99, 4, 36, 68, 100, 5, 37, 69, 101, 6, 38, 70, 102, 7, 39, 71, 103, 8, 40, 72, 104, 9, 41, 73, 105, 10, 42, 74, 106, 11, 43, 75, 107, 12, 44, 76, 108, 13, 45, 77, 109, 14, 46, 78, 110, 15, 47, 79, 111, 16, 48, 80, 112, 17, 49, 81, 113, 18, 50, 82, 114, 19, 51, 83, 115, 20, 52, 84, 116, 21, 53, 85, 117, 22, 54, 86, 118, 23, 55, 87, 119, 24, 56, 88, 120, 25, 57, 89, 121, 26, 58, 90, 122, 27, 59, 91, 123, 28, 60, 92, 124, 29, 61, 93, 125, 30, 62, 94, 126, 31, 63, 95, 127,
// CHECK: 128, 160, 192, 224, 129, 161, 193, 225, 130, 162, 194, 226, 131, 163, 195, 227, 132, 164, 196, 228, 133, 165, 197, 229, 134, 166, 198, 230, 135, 167, 199, 231, 136, 168, 200, 232, 137, 169, 201, 233, 138, 170, 202, 234, 139, 171, 203, 235, 140, 172, 204, 236, 141, 173, 205, 237, 142, 174, 206, 238, 143, 175, 207, 239, 144, 176, 208, 240, 145, 177, 209, 241, 146, 178, 210, 242, 147, 179, 211, 243, 148, 180, 212, 244, 149, 181, 213, 245, 150, 182, 214, 246, 151, 183, 215, 247, 152, 184, 216, 248, 153, 185, 217, 249, 154, 186, 218, 250, 155, 187, 219, 251, 156, 188, 220, 252, 157, 189, 221, 253, 158, 190, 222, 254, 159, 191, 223, 255,
// CHECK: 256, 288, 320, 352, 257, 289, 321, 353, 258, 290, 322, 354, 259, 291, 323, 355, 260, 292, 324, 356, 261, 293, 325, 357, 262, 294, 326, 358, 263, 295, 327, 359, 264, 296, 328, 360, 265, 297, 329, 361, 266, 298, 330, 362, 267, 299, 331, 363, 268, 300, 332, 364, 269, 301, 333, 365, 270, 302, 334, 366, 271, 303, 335, 367, 272, 304, 336, 368, 273, 305, 337, 369, 274, 306, 338, 370, 275, 307, 339, 371, 276, 308, 340, 372, 277, 309, 341, 373, 278, 310, 342, 374, 279, 311, 343, 375, 280, 312, 344, 376, 281, 313, 345, 377, 282, 314, 346, 378, 283, 315, 347, 379, 284, 316, 348, 380, 285, 317, 349, 381, 286, 318, 350, 382, 287, 319, 351, 383,
// CHECK: 384, 416, 448, 480, 385, 417, 449, 481, 386, 418, 450, 482, 387, 419, 451, 483, 388, 420, 452, 484, 389, 421, 453, 485, 390, 422, 454, 486, 391, 423, 455, 487, 392, 424, 456, 488, 393, 425, 457, 489, 394, 426, 458, 490, 395, 427, 459, 491, 396, 428, 460, 492, 397, 429, 461, 493, 398, 430, 462, 494, 399, 431, 463, 495, 400, 432, 464, 496, 401, 433, 465, 497, 402, 434, 466, 498, 403, 435, 467, 499, 404, 436, 468, 500, 405, 437, 469, 501, 406, 438, 470, 502, 407, 439, 471, 503, 408, 440, 472, 504, 409, 441, 473, 505, 410, 442, 474, 506, 411, 443, 475, 507, 412, 444, 476, 508, 413, 445, 477, 509, 414, 446, 478, 510, 415, 447, 479, 511,
// CHECK: 512, 544, 576, 608, 513, 545, 577, 609, 514, 546, 578, 610, 515, 547, 579, 611, 516, 548, 580, 612, 517, 549, 581, 613, 518, 550, 582, 614, 519, 551, 583, 615, 520, 552, 584, 616, 521, 553, 585, 617, 522, 554, 586, 618, 523, 555, 587, 619, 524, 556, 588, 620, 525, 557, 589, 621, 526, 558, 590, 622, 527, 559, 591, 623, 528, 560, 592, 624, 529, 561, 593, 625, 530, 562, 594, 626, 531, 563, 595, 627, 532, 564, 596, 628, 533, 565, 597, 629, 534, 566, 598, 630, 535, 567, 599, 631, 536, 568, 600, 632, 537, 569, 601, 633, 538, 570, 602, 634, 539, 571, 603, 635, 540, 572, 604, 636, 541, 573, 605, 637, 542, 574, 606, 638, 543, 575, 607, 639,
// CHECK: 640, 672, 704, 736, 641, 673, 705, 737, 642, 674, 706, 738, 643, 675, 707, 739, 644, 676, 708, 740, 645, 677, 709, 741, 646, 678, 710, 742, 647, 679, 711, 743, 648, 680, 712, 744, 649, 681, 713, 745, 650, 682, 714, 746, 651, 683, 715, 747, 652, 684, 716, 748, 653, 685, 717, 749, 654, 686, 718, 750, 655, 687, 719, 751, 656, 688, 720, 752, 657, 689, 721, 753, 658, 690, 722, 754, 659, 691, 723, 755, 660, 692, 724, 756, 661, 693, 725, 757, 662, 694, 726, 758, 663, 695, 727, 759, 664, 696, 728, 760, 665, 697, 729, 761, 666, 698, 730, 762, 667, 699, 731, 763, 668, 700, 732, 764, 669, 701, 733, 765, 670, 702, 734, 766, 671, 703, 735, 767,
// CHECK: 768, 800, 832, 864, 769, 801, 833, 865, 770, 802, 834, 866, 771, 803, 835, 867, 772, 804, 836, 868, 773, 805, 837, 869, 774, 806, 838, 870, 775, 807, 839, 871, 776, 808, 840, 872, 777, 809, 841, 873, 778, 810, 842, 874, 779, 811, 843, 875, 780, 812, 844, 876, 781, 813, 845, 877, 782, 814, 846, 878, 783, 815, 847, 879, 784, 816, 848, 880, 785, 817, 849, 881, 786, 818, 850, 882, 787, 819, 851, 883, 788, 820, 852, 884, 789, 821, 853, 885, 790, 822, 854, 886, 791, 823, 855, 887, 792, 824, 856, 888, 793, 825, 857, 889, 794, 826, 858, 890, 795, 827, 859, 891, 796, 828, 860, 892, 797, 829, 861, 893, 798, 830, 862, 894, 799, 831, 863, 895,
// CHECK: 896, 928, 960, 992, 897, 929, 961, 993, 898, 930, 962, 994, 899, 931, 963, 995, 900, 932, 964, 996, 901, 933, 965, 997, 902, 934, 966, 998, 903, 935, 967, 999, 904, 936, 968, 1000, 905, 937, 969, 1001, 906, 938, 970, 1002, 907, 939, 971, 1003, 908, 940, 972, 1004, 909, 941, 973, 1005, 910, 942, 974, 1006, 911, 943, 975, 1007, 912, 944, 976, 1008, 913, 945, 977, 1009, 914, 946, 978, 1010, 915, 947, 979, 1011, 916, 948, 980, 1012, 917, 949, 981, 1013, 918, 950, 982, 1014, 919, 951, 983, 1015, 920, 952, 984, 1016, 921, 953, 985, 1017, 922, 954, 986, 1018, 923, 955, 987, 1019, 924, 956, 988, 1020, 925, 957, 989, 1021, 926, 958, 990, 1022, 927, 959, 991, 1023] {packed} : vector<1024xi8>, vector<1024xi8>
//       CHECK: %[[B3:.*]] = vector.shape_cast %[[B2]] {packed} : vector<1024xi8> to vector<8x32x4xi8>
//       CHECK: %[[RES:.*]] = xegpu.dpas %[[ARG1]], %[[B3]] : vector<8x32xi8>, vector<8x32x4xi8> -> vector<8x32xi32>
//       CHECK: return %[[RES]] : vector<8x32xi32>

func.func @test(%arg1 : vector<8x32xi8>, %arg2 : vector<32x32xi8>) -> vector<8x32xi32> {
  %0 = xegpu.dpas %arg1, %arg2 : vector<8x32xi8>, vector<32x32xi8> -> vector<8x32xi32>
  return %0 : vector<8x32xi32>
}

// -----

// CHECK-LABEL: @test
//  CHECK-SAME: (%[[ARG1:.*]]: !xegpu.tensor_desc<8x16xf16>, %[[ARG2:.*]]: !xegpu.tensor_desc<16x16xf16>)
//       CHECK:  %[[A:.*]] = xegpu.load_nd %[[ARG1]] : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
//       CHECK:  %[[B:.*]] = xegpu.load_nd %[[ARG2]] <{packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
//       CHECK:  %[[RES:.*]] = xegpu.dpas %[[A]], %[[B]] : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
//       CHECK:  return %[[RES]]

func.func @test(%arg1 : !xegpu.tensor_desc<8x16xf16>, %arg2 : !xegpu.tensor_desc<16x16xf16>) -> vector<8x16xf32> {
  %0 = xegpu.load_nd %arg1 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = xegpu.load_nd %arg2 : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %2 = xegpu.dpas %0, %1 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  return %2 : vector<8x16xf32>
}

// -----

// CHECK-LABEL: @test
//  CHECK-SAME: (%[[ARG1:.*]]: !xegpu.tensor_desc<8x16xf16>, %[[ARG2:.*]]: !xegpu.tensor_desc<16x16xf16>)
//       CHECK:  %[[A:.*]] = xegpu.load_nd %[[ARG1]]  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
//       CHECK:  %[[B:.*]] = xegpu.load_nd %[[ARG2]] <{packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
//       CHECK:  %[[RES1:.*]] = xegpu.dpas %[[A]], %[[B]] : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
//       CHECK:  %[[RES2:.*]] = xegpu.dpas %[[A]], %[[B]] : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
//       CHECK:  return %[[RES1]], %[[RES2]]

func.func @test(%arg1 : !xegpu.tensor_desc<8x16xf16>, %arg2 : !xegpu.tensor_desc<16x16xf16>) -> (vector<8x16xf32>, vector<8x16xf32>) {
  %0 = xegpu.load_nd %arg1 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = xegpu.load_nd %arg2 : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %2 = xegpu.dpas %0, %1 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  %3 = xegpu.dpas %0, %1 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  return %2, %3 : vector<8x16xf32>, vector<8x16xf32>
}

// -----

// CHECK-LABEL: @test
//  CHECK-SAME: (%[[ARG1:.*]]: !xegpu.tensor_desc<8x16xf16>, %[[ARG2:.*]]: !xegpu.tensor_desc<16x16xf16>)
//       CHECK:  %[[A:.*]] = xegpu.load_nd %[[ARG1]]  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
//       CHECK:  %[[B:.*]] = xegpu.load_nd %[[ARG2]] : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
//       CHECK:  %[[B1:.*]] = vector.shape_cast %[[B]] {packed} : vector<16x16xf16> to vector<256xf16>
//       CHECK:  %[[B2:.*]] = vector.shuffle %[[B1]], %[[B1]]
// CHECK: [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31,
// CHECK: 32, 48, 33, 49, 34, 50, 35, 51, 36, 52, 37, 53, 38, 54, 39, 55, 40, 56, 41, 57, 42, 58, 43, 59, 44, 60, 45, 61, 46, 62, 47, 63,
// CHECK: 64, 80, 65, 81, 66, 82, 67, 83, 68, 84, 69, 85, 70, 86, 71, 87, 72, 88, 73, 89, 74, 90, 75, 91, 76, 92, 77, 93, 78, 94, 79, 95,
// CHECK: 96, 112, 97, 113, 98, 114, 99, 115, 100, 116, 101, 117, 102, 118, 103, 119, 104, 120, 105, 121, 106, 122, 107, 123, 108, 124, 109, 125, 110, 126, 111, 127,
// CHECK: 128, 144, 129, 145, 130, 146, 131, 147, 132, 148, 133, 149, 134, 150, 135, 151, 136, 152, 137, 153, 138, 154, 139, 155, 140, 156, 141, 157, 142, 158, 143, 159,
// CHECK: 160, 176, 161, 177, 162, 178, 163, 179, 164, 180, 165, 181, 166, 182, 167, 183, 168, 184, 169, 185, 170, 186, 171, 187, 172, 188, 173, 189, 174, 190, 175, 191,
// CHECK: 192, 208, 193, 209, 194, 210, 195, 211, 196, 212, 197, 213, 198, 214, 199, 215, 200, 216, 201, 217, 202, 218, 203, 219, 204, 220, 205, 221, 206, 222, 207, 223,
// CHECK: 224, 240, 225, 241, 226, 242, 227, 243, 228, 244, 229, 245, 230, 246, 231, 247, 232, 248, 233, 249, 234, 250, 235, 251, 236, 252, 237, 253, 238, 254, 239, 255] {packed} : vector<256xf16>, vector<256xf16>
//       CHECK:  %[[B3:.*]] = vector.shape_cast %[[B2]] {packed} : vector<256xf16> to vector<8x16x2xf16>
//       CHECK:  %[[RES:.*]] = xegpu.dpas %[[A]], %[[B3]] : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
//       CHECK:  return %[[RES]]

func.func @test(%arg1 : !xegpu.tensor_desc<8x16xf16>, %arg2 : !xegpu.tensor_desc<16x16xf16>) -> (vector<8x16xf32>, vector<16x16xf16>) {
  %0 = xegpu.load_nd %arg1 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = xegpu.load_nd %arg2 : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %2 = xegpu.dpas %0, %1 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  return %2, %1 : vector<8x16xf32>, vector<16x16xf16>
}

// -----

// CHECK-LABEL: @test
//  CHECK-SAME: (%[[ARG1:.*]]: !xegpu.tensor_desc<8x16xf16>, %[[ARG2:.*]]: !xegpu.tensor_desc<16x16xf16>)
//       CHECK:  %[[A:.*]] = xegpu.load_nd %[[ARG1]]  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
//       CHECK:  %[[B:.*]] = xegpu.load_nd %[[ARG2]]  : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
//       CHECK:  %[[B1:.*]] = arith.extf %[[B]] : vector<16x16xf16> to vector<16x16xf32>
//       CHECK:  %[[B2:.*]] = arith.truncf %[[B1]] : vector<16x16xf32> to vector<16x16xf16>

//       CHECK:  %[[R4:.*]] = vector.shape_cast %[[B2]] {packed} : vector<16x16xf16> to vector<256xf16>
//       CHECK:  %[[R5:.*]] = vector.shuffle %[[R4]], %[[R4]]
//   CHECK-SAME: [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26,
//   CHECK-SAME: 11, 27, 12, 28, 13, 29, 14, 30, 15, 31, 32, 48, 33, 49, 34, 50, 35, 51, 36, 52,
//   CHECK-SAME: 37, 53, 38, 54, 39, 55, 40, 56, 41, 57, 42, 58, 43, 59, 44, 60, 45, 61, 46, 62,
//   CHECK-SAME: 47, 63, 64, 80, 65, 81, 66, 82, 67, 83, 68, 84, 69, 85, 70, 86, 71, 87, 72, 88,
//   CHECK-SAME: 73, 89, 74, 90, 75, 91, 76, 92, 77, 93, 78, 94, 79, 95, 96, 112, 97, 113, 98, 114,
//   CHECK-SAME: 99, 115, 100, 116, 101, 117, 102, 118, 103, 119, 104, 120, 105, 121, 106, 122,
//   CHECK-SAME: 107, 123, 108, 124, 109, 125, 110, 126, 111, 127, 128, 144, 129, 145, 130, 146,
//   CHECK-SAME: 131, 147, 132, 148, 133, 149, 134, 150, 135, 151, 136, 152, 137, 153, 138, 154,
//   CHECK-SAME: 139, 155, 140, 156, 141, 157, 142, 158, 143, 159, 160, 176, 161, 177, 162, 178,
//   CHECK-SAME: 163, 179, 164, 180, 165, 181, 166, 182, 167, 183, 168, 184, 169, 185, 170, 186,
//   CHECK-SAME: 171, 187, 172, 188, 173, 189, 174, 190, 175, 191, 192, 208, 193, 209, 194, 210,
//   CHECK-SAME: 195, 211, 196, 212, 197, 213, 198, 214, 199, 215, 200, 216, 201, 217, 202, 218,
//   CHECK-SAME: 203, 219, 204, 220, 205, 221, 206, 222, 207, 223, 224, 240, 225, 241, 226, 242,
//   CHECK-SAME: 227, 243, 228, 244, 229, 245, 230, 246, 231, 247, 232, 248, 233, 249, 234, 250,
//   CHECK-SAME: 235, 251, 236, 252, 237, 253, 238, 254, 239, 255] {packed} : vector<256xf16>, vector<256xf16>
//       CHECK:  %[[R6:.*]] = vector.shape_cast %[[R5]] {packed} : vector<256xf16> to vector<8x16x2xf16>
//       CHECK:  %[[RES:.*]] = xegpu.dpas %[[A]], %[[R6]] : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
//       CHECK:  return %[[RES]]

func.func @test(%arg1 : !xegpu.tensor_desc<8x16xf16>, %arg2 : !xegpu.tensor_desc<16x16xf16>) -> vector<8x16xf32> {
  %0 = xegpu.load_nd %arg1 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = xegpu.load_nd %arg2 : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %2 = arith.extf %1: vector<16x16xf16> to vector<16x16xf32>
  %3 = arith.truncf %2: vector<16x16xf32> to vector<16x16xf16>
  %4 = xegpu.dpas %0, %3 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  return %4 : vector<8x16xf32>
}

// -----

// CHECK-LABEL: @test
//  CHECK-SAME: (%[[ARG1:.*]]: !xegpu.tensor_desc<8x16xf16>, %[[ARG2:.*]]: !xegpu.tensor_desc<16x16xf16>)
//       CHECK:  %[[A:.*]] = xegpu.load_nd %[[ARG1]]  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
//       CHECK:  %[[B1:.*]] = xegpu.load_nd %[[ARG2]] <{packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
//       CHECK:  %[[B2:.*]] = xegpu.load_nd %[[ARG2]] <{packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
//       CHECK:  %[[B3:.*]] = arith.addf %[[B1]], %[[B2]] : vector<8x16x2xf16>
//       CHECK:  %[[RES:.*]] = xegpu.dpas %[[A]], %[[B3]] : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
//       CHECK:  return %[[RES]]

func.func @test(%arg1 : !xegpu.tensor_desc<8x16xf16>, %arg2 : !xegpu.tensor_desc<16x16xf16>) -> vector<8x16xf32> {
  %0 = xegpu.load_nd %arg1 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = xegpu.load_nd %arg2 : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %2 = xegpu.load_nd %arg2 : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %3 = arith.addf %1, %2 : vector<16x16xf16>
  %4 = xegpu.dpas %0, %3 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  return %4 : vector<8x16xf32>
}

// -----

// CHECK-LABEL: @test
//  CHECK-SAME: (%[[ARG1:.*]]: !xegpu.tensor_desc<8x16xf16>, %[[ARG2:.*]]: !xegpu.tensor_desc<16x16xf16>)
//       CHECK:  %[[A:.*]] = xegpu.load_nd %[[ARG1]]  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
//       CHECK:  %[[B1:.*]] = xegpu.load_nd %[[ARG2]] <{packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
//       CHECK:  %[[B2:.*]] = arith.constant dense<1.000000e+00> : vector<16x16xf16>
//       CHECK:  %[[B3:.*]] = vector.shape_cast %[[B2]] {packed} : vector<16x16xf16> to vector<256xf16>
//       CHECK:  %[[B4:.*]] = vector.shuffle %[[B3]], %[[B3]]
// CHECK: [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31,
// CHECK: 32, 48, 33, 49, 34, 50, 35, 51, 36, 52, 37, 53, 38, 54, 39, 55, 40, 56, 41, 57, 42, 58, 43, 59, 44, 60, 45, 61, 46, 62, 47, 63,
// CHECK: 64, 80, 65, 81, 66, 82, 67, 83, 68, 84, 69, 85, 70, 86, 71, 87, 72, 88, 73, 89, 74, 90, 75, 91, 76, 92, 77, 93, 78, 94, 79, 95,
// CHECK: 96, 112, 97, 113, 98, 114, 99, 115, 100, 116, 101, 117, 102, 118, 103, 119, 104, 120, 105, 121, 106, 122, 107, 123, 108, 124, 109, 125, 110, 126, 111, 127,
// CHECK: 128, 144, 129, 145, 130, 146, 131, 147, 132, 148, 133, 149, 134, 150, 135, 151, 136, 152, 137, 153, 138, 154, 139, 155, 140, 156, 141, 157, 142, 158, 143, 159,
// CHECK: 160, 176, 161, 177, 162, 178, 163, 179, 164, 180, 165, 181, 166, 182, 167, 183, 168, 184, 169, 185, 170, 186, 171, 187, 172, 188, 173, 189, 174, 190, 175, 191,
// CHECK: 192, 208, 193, 209, 194, 210, 195, 211, 196, 212, 197, 213, 198, 214, 199, 215, 200, 216, 201, 217, 202, 218, 203, 219, 204, 220, 205, 221, 206, 222, 207, 223,
// CHECK: 224, 240, 225, 241, 226, 242, 227, 243, 228, 244, 229, 245, 230, 246, 231, 247, 232, 248, 233, 249, 234, 250, 235, 251, 236, 252, 237, 253, 238, 254, 239, 255] {packed} : vector<256xf16>, vector<256xf16>
//       CHECK:  %[[B5:.*]] = vector.shape_cast %[[B4]] {packed} : vector<256xf16> to vector<8x16x2xf16>
//       CHECK:  %[[B6:.*]] = arith.addf %[[B1]], %[[B5]] : vector<8x16x2xf16>
//       CHECK:  %[[RES:.*]] = xegpu.dpas %[[A]], %[[B6]] : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
//       CHECK:  return %[[RES]]

func.func @test(%arg1 : !xegpu.tensor_desc<8x16xf16>, %arg2 : !xegpu.tensor_desc<16x16xf16>) -> vector<8x16xf32> {
  %0 = xegpu.load_nd %arg1 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = xegpu.load_nd %arg2 : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %2 = arith.constant dense<1.0> : vector<16x16xf16>
  %3 = arith.addf %1, %2 : vector<16x16xf16>
  %4 = xegpu.dpas %0, %3 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  return %4 : vector<8x16xf32>
}

// -----

// CHECK-LABEL: @test
//  CHECK-SAME: (%[[ARG1:.*]]: !xegpu.tensor_desc<8x16xf16>, %[[ARG2:.*]]: !xegpu.tensor_desc<16x16xf16>)
//       CHECK:  %[[A:.*]] = xegpu.load_nd %[[ARG1]]  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
//       CHECK:  %[[B1:.*]] = xegpu.load_nd %[[ARG2]] : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
//       CHECK:  %[[B2:.*]] = xegpu.load_nd %[[ARG2]] : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
//       CHECK:  %[[B3:.*]] = arith.addf %[[B1]], %[[B2]] : vector<16x16xf16>
//       CHECK:  %[[B4:.*]] = vector.shape_cast %[[B3]] {packed} : vector<16x16xf16> to vector<256xf16>
//       CHECK:  %[[B5:.*]] = vector.shuffle %[[B4]], %[[B4]]
// CHECK: [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31,
// CHECK: 32, 48, 33, 49, 34, 50, 35, 51, 36, 52, 37, 53, 38, 54, 39, 55, 40, 56, 41, 57, 42, 58, 43, 59, 44, 60, 45, 61, 46, 62, 47, 63,
// CHECK: 64, 80, 65, 81, 66, 82, 67, 83, 68, 84, 69, 85, 70, 86, 71, 87, 72, 88, 73, 89, 74, 90, 75, 91, 76, 92, 77, 93, 78, 94, 79, 95,
// CHECK: 96, 112, 97, 113, 98, 114, 99, 115, 100, 116, 101, 117, 102, 118, 103, 119, 104, 120, 105, 121, 106, 122, 107, 123, 108, 124, 109, 125, 110, 126, 111, 127,
// CHECK: 128, 144, 129, 145, 130, 146, 131, 147, 132, 148, 133, 149, 134, 150, 135, 151, 136, 152, 137, 153, 138, 154, 139, 155, 140, 156, 141, 157, 142, 158, 143, 159,
// CHECK: 160, 176, 161, 177, 162, 178, 163, 179, 164, 180, 165, 181, 166, 182, 167, 183, 168, 184, 169, 185, 170, 186, 171, 187, 172, 188, 173, 189, 174, 190, 175, 191,
// CHECK: 192, 208, 193, 209, 194, 210, 195, 211, 196, 212, 197, 213, 198, 214, 199, 215, 200, 216, 201, 217, 202, 218, 203, 219, 204, 220, 205, 221, 206, 222, 207, 223,
// CHECK: 224, 240, 225, 241, 226, 242, 227, 243, 228, 244, 229, 245, 230, 246, 231, 247, 232, 248, 233, 249, 234, 250, 235, 251, 236, 252, 237, 253, 238, 254, 239, 255] {packed} : vector<256xf16>, vector<256xf16>
//       CHECK:  %[[B6:.*]] = vector.shape_cast %[[B5]] {packed} : vector<256xf16> to vector<8x16x2xf16>
//       CHECK:  %[[RES:.*]] = xegpu.dpas %[[A]], %[[B6]] : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
//       CHECK:  return %[[RES]], %[[B2]]

func.func @test(%arg1 : !xegpu.tensor_desc<8x16xf16>, %arg2 : !xegpu.tensor_desc<16x16xf16>) -> (vector<8x16xf32>, vector<16x16xf16>) {
  %0 = xegpu.load_nd %arg1 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = xegpu.load_nd %arg2 : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %2 = xegpu.load_nd %arg2 : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %3 = arith.addf %1, %2 : vector<16x16xf16>
  %4 = xegpu.dpas %0, %3 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  return %4, %2 : vector<8x16xf32>, vector<16x16xf16>
}

// -----

// CHECK-LABEL: @test
//  CHECK-SAME: (%[[ARG1:.*]]: !xegpu.tensor_desc<8x16xf16>, %[[ARG2:.*]]: !xegpu.tensor_desc<16x16xf16>, %{{.*}}: index)
//       CHECK:  %[[A:.*]] = xegpu.load_nd %[[ARG1]]  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
//       CHECK:  %[[B:.*]] = xegpu.load_nd %[[ARG2]] <{packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
//       CHECK:  %[[RES:.*]] = xegpu.dpas %[[A]], %[[B]] : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
//       CHECK:  %[[RES1:.*]]:2 = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ITER1:.*]] = %[[B]], %[[ITER2:.*]] = %[[RES]]) -> (vector<8x16x2xf16>, vector<8x16xf32>) {
//       CHECK:  %[[B1:.*]] = xegpu.load_nd %[[ARG2]] <{packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
//       CHECK:  %[[B2:.*]] = arith.addf %[[ITER1]], %[[B1]] : vector<8x16x2xf16>
//       CHECK:  %[[RES2:.*]] = xegpu.dpas %[[A]], %[[B2]] : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
//       CHECK:  scf.yield %[[B2]], %[[RES2]] : vector<8x16x2xf16>, vector<8x16xf32>
//       CHECK:  }
//       CHECK:  return %[[RES1]]#1 : vector<8x16xf32>

func.func @test(%arg1 : !xegpu.tensor_desc<8x16xf16>, %arg2 : !xegpu.tensor_desc<16x16xf16>, %arg3 : index) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = xegpu.load_nd %arg1 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = xegpu.load_nd %arg2 : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %2 = xegpu.dpas %0, %1 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  %3:2 = scf.for %i = %c0 to %arg3 step %c1 iter_args(%iter = %1, %res = %2) -> (vector<16x16xf16>, vector<8x16xf32>) {
    %4 = xegpu.load_nd %arg2 : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
    %5 = arith.addf %iter, %4 : vector<16x16xf16>
    %6 = xegpu.dpas %0, %5 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    scf.yield %5, %6: vector<16x16xf16>, vector<8x16xf32>
  }
  return %3#1 : vector<8x16xf32>
}

// -----

// CHECK-LABEL: @test
//  CHECK-SAME: (%[[ARG1:.*]]: !xegpu.tensor_desc<8x16xf16>, %[[ARG2:.*]]: !xegpu.tensor_desc<16x16xf16>, %{{.*}}: i1)
//       CHECK:  %[[A:.*]] = xegpu.load_nd %[[ARG1]]  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
//       CHECK:  %[[B:.*]] = scf.if %{{.*}} -> (vector<8x16x2xf16>)
//       CHECK:  %[[B1:.*]] = xegpu.load_nd %[[ARG2]] <{packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
//       CHECK:  scf.yield %[[B1]]
//       CHECK:  else
//       CHECK:  %[[B2:.*]] = xegpu.load_nd %[[ARG2]] <{packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
//       CHECK:  scf.yield %[[B2]]
//       CHECK:  %[[RES:.*]] = xegpu.dpas %[[A]], %[[B]] : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
//       CHECK:  return %[[RES]]

func.func @test(%arg1 : !xegpu.tensor_desc<8x16xf16>, %arg2 : !xegpu.tensor_desc<16x16xf16>, %arg3 : i1) -> vector<8x16xf32> {
  %0 = xegpu.load_nd %arg1 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = scf.if %arg3 -> (vector<16x16xf16>) {
    %b = xegpu.load_nd %arg2 : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
    scf.yield %b : vector<16x16xf16>
  } else {
    %b = xegpu.load_nd %arg2 : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
    scf.yield %b : vector<16x16xf16>
  }
  %2 = xegpu.dpas %0, %1 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  return %2 : vector<8x16xf32>
}

// -----

// CHECK-LABEL: @test
//  CHECK-SAME: (%[[ARG1:.*]]: !xegpu.tensor_desc<8x16xf16>, %[[ARG2:.*]]: !xegpu.tensor_desc<16x16xf16>, %{{.*}}: i1)
//       CHECK:  %[[A:.*]] = xegpu.load_nd %[[ARG1]]  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
//       CHECK:  %[[B:.*]] = scf.if %{{.*}} -> (vector<16x16xf16>)
//       CHECK:  %[[B1:.*]] = xegpu.load_nd %[[ARG2]] : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
//       CHECK:  scf.yield %[[B1]]
//       CHECK:  else
//       CHECK:  %[[B2:.*]] = xegpu.load_nd %[[ARG2]] : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
//       CHECK:  scf.yield %[[B2]]
//       CHECK:  %[[B3:.*]] = vector.shape_cast %[[B]] {packed} : vector<16x16xf16> to vector<256xf16>
//       CHECK:  %[[B4:.*]] = vector.shuffle %[[B3]], %[[B3]]
// CHECK: [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31,
// CHECK: 32, 48, 33, 49, 34, 50, 35, 51, 36, 52, 37, 53, 38, 54, 39, 55, 40, 56, 41, 57, 42, 58, 43, 59, 44, 60, 45, 61, 46, 62, 47, 63,
// CHECK: 64, 80, 65, 81, 66, 82, 67, 83, 68, 84, 69, 85, 70, 86, 71, 87, 72, 88, 73, 89, 74, 90, 75, 91, 76, 92, 77, 93, 78, 94, 79, 95,
// CHECK: 96, 112, 97, 113, 98, 114, 99, 115, 100, 116, 101, 117, 102, 118, 103, 119, 104, 120, 105, 121, 106, 122, 107, 123, 108, 124, 109, 125, 110, 126, 111, 127,
// CHECK: 128, 144, 129, 145, 130, 146, 131, 147, 132, 148, 133, 149, 134, 150, 135, 151, 136, 152, 137, 153, 138, 154, 139, 155, 140, 156, 141, 157, 142, 158, 143, 159,
// CHECK: 160, 176, 161, 177, 162, 178, 163, 179, 164, 180, 165, 181, 166, 182, 167, 183, 168, 184, 169, 185, 170, 186, 171, 187, 172, 188, 173, 189, 174, 190, 175, 191,
// CHECK: 192, 208, 193, 209, 194, 210, 195, 211, 196, 212, 197, 213, 198, 214, 199, 215, 200, 216, 201, 217, 202, 218, 203, 219, 204, 220, 205, 221, 206, 222, 207, 223,
// CHECK: 224, 240, 225, 241, 226, 242, 227, 243, 228, 244, 229, 245, 230, 246, 231, 247, 232, 248, 233, 249, 234, 250, 235, 251, 236, 252, 237, 253, 238, 254, 239, 255] {packed} : vector<256xf16>, vector<256xf16>
//       CHECK:  %[[B5:.*]] = vector.shape_cast %[[B4]] {packed} : vector<256xf16> to vector<8x16x2xf16>
//       CHECK:  %[[RES:.*]] = xegpu.dpas %[[A]], %[[B5]] : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
//       CHECK:  return %[[RES]], %[[B]]

func.func @test(%arg1 : !xegpu.tensor_desc<8x16xf16>, %arg2 : !xegpu.tensor_desc<16x16xf16>, %arg3 : i1) -> (vector<8x16xf32>, vector<16x16xf16>) {
  %0 = xegpu.load_nd %arg1 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = scf.if %arg3 -> (vector<16x16xf16>) {
    %b = xegpu.load_nd %arg2 : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
    scf.yield %b : vector<16x16xf16>
  } else {
    %b = xegpu.load_nd %arg2 : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
    scf.yield %b : vector<16x16xf16>
  }
  %2 = xegpu.dpas %0, %1 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  return %2, %1 : vector<8x16xf32>, vector<16x16xf16>
}

// -----

// CHECK-LABEL: @test
//  CHECK-SAME: (%[[ARG1:.*]]: !xegpu.tensor_desc<8x16xf16>, %[[ARG2:.*]]: !xegpu.tensor_desc<16x16xf16>, %[[ARG3:.*]]: vector<16x16xf16>, %{{.*}}: i1)
//       CHECK:  %[[B2:.*]] = vector.shape_cast %[[ARG3]] {packed} : vector<16x16xf16> to vector<256xf16>
//       CHECK:  %[[B3:.*]] = vector.shuffle %[[B2]], %[[B2]]
// CHECK: [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31,
// CHECK: 32, 48, 33, 49, 34, 50, 35, 51, 36, 52, 37, 53, 38, 54, 39, 55, 40, 56, 41, 57, 42, 58, 43, 59, 44, 60, 45, 61, 46, 62, 47, 63,
// CHECK: 64, 80, 65, 81, 66, 82, 67, 83, 68, 84, 69, 85, 70, 86, 71, 87, 72, 88, 73, 89, 74, 90, 75, 91, 76, 92, 77, 93, 78, 94, 79, 95,
// CHECK: 96, 112, 97, 113, 98, 114, 99, 115, 100, 116, 101, 117, 102, 118, 103, 119, 104, 120, 105, 121, 106, 122, 107, 123, 108, 124, 109, 125, 110, 126, 111, 127,
// CHECK: 128, 144, 129, 145, 130, 146, 131, 147, 132, 148, 133, 149, 134, 150, 135, 151, 136, 152, 137, 153, 138, 154, 139, 155, 140, 156, 141, 157, 142, 158, 143, 159,
// CHECK: 160, 176, 161, 177, 162, 178, 163, 179, 164, 180, 165, 181, 166, 182, 167, 183, 168, 184, 169, 185, 170, 186, 171, 187, 172, 188, 173, 189, 174, 190, 175, 191,
// CHECK: 192, 208, 193, 209, 194, 210, 195, 211, 196, 212, 197, 213, 198, 214, 199, 215, 200, 216, 201, 217, 202, 218, 203, 219, 204, 220, 205, 221, 206, 222, 207, 223,
// CHECK: 224, 240, 225, 241, 226, 242, 227, 243, 228, 244, 229, 245, 230, 246, 231, 247, 232, 248, 233, 249, 234, 250, 235, 251, 236, 252, 237, 253, 238, 254, 239, 255] {packed} : vector<256xf16>, vector<256xf16>
//       CHECK:  %[[B4:.*]] = vector.shape_cast %[[B3]] {packed} : vector<256xf16> to vector<8x16x2xf16>
//       CHECK:  %[[A:.*]] = xegpu.load_nd %[[ARG1]]  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
//       CHECK:  %[[B:.*]] = scf.if %{{.*}} -> (vector<8x16x2xf16>)
//       CHECK:  %[[B1:.*]] = xegpu.load_nd %[[ARG2]] <{packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
//       CHECK:  scf.yield %[[B1]]
//       CHECK:  else
//       CHECK:  scf.yield %[[B4]]
//       CHECK:  %[[RES:.*]] = xegpu.dpas %[[A]], %[[B]] : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
//       CHECK:  return %[[RES]]

func.func @test(%arg1 : !xegpu.tensor_desc<8x16xf16>, %arg2 : !xegpu.tensor_desc<16x16xf16>, %arg3 : vector<16x16xf16>, %arg4 : i1) -> vector<8x16xf32> {
  %0 = xegpu.load_nd %arg1 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %1 = scf.if %arg4 -> (vector<16x16xf16>) {
    %b = xegpu.load_nd %arg2 : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
    scf.yield %b : vector<16x16xf16>
  } else {
    scf.yield %arg3 : vector<16x16xf16>
  }
  %2 = xegpu.dpas %0, %1 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  return %2 : vector<8x16xf32>
}
