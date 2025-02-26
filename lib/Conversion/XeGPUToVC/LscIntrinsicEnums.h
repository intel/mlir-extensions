//===- LscIntrinsicEnums.h -  Enums used by lsc intrinsics ------*- C++ -*-===//
//
// Copyright 2024 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains some enum definitions used by lsc intrinsics.
//===----------------------------------------------------------------------===//

#ifndef LSC_INTRINSIC_ENUMS_H
#define LSC_INTRINSIC_ENUMS_H

// these definitions are coppied from IGC,
// https://github.com/intel/intel-graphics-compiler/blob/master/visa/include/visa_igc_common_header.h#L680
// TODO: remove this file and replace it with IGC using e.g., submodules
enum LSC_DATA_SIZE {
  LSC_DATA_SIZE_INVALID,
  LSC_DATA_SIZE_8b,  // DATA:u8...
  LSC_DATA_SIZE_16b, // DATA:u16...
  LSC_DATA_SIZE_32b, // DATA:u32...
  LSC_DATA_SIZE_64b, // DATA:u64...
                     // data types supporting conversion on load
  // 8c32b reads load (8) bits, (c)onvert to (32) bits (zero extending)
  // store truncates
  //
  // In DG2 and PVC the upper bits are undefined.
  // XE2+ makes them zeros.
  LSC_DATA_SIZE_8c32b,   // DATA:u8c32...   (zero-extend / truncate)
  LSC_DATA_SIZE_16c32b,  // DATA:u16c32..   (zero-extend / truncate)
  LSC_DATA_SIZE_16c32bH, // DATA:u16c32h..  h means load to (h)igh 16
                         // data stored in upper 16; zero-fills bottom 16
                         // (bfloat raw conversion to 32b float)
};

// The number of elements per address ("vector" size)
enum LSC_DATA_ELEMS {
  LSC_DATA_ELEMS_INVALID,
  LSC_DATA_ELEMS_1,  // DATA:..x1
  LSC_DATA_ELEMS_2,  // DATA:..x2
  LSC_DATA_ELEMS_3,  // DATA:..x3
  LSC_DATA_ELEMS_4,  // DATA:..x4
  LSC_DATA_ELEMS_8,  // DATA:..x8
  LSC_DATA_ELEMS_16, // DATA:..x16
  LSC_DATA_ELEMS_32, // DATA:..x32
  LSC_DATA_ELEMS_64, // DATA:..x64
};

enum LSC_DATA_ORDER {
  LSC_DATA_ORDER_INVALID,
  LSC_DATA_ORDER_NONTRANSPOSE,
  LSC_DATA_ORDER_TRANSPOSE, // DATA:...t
};

enum LSC_CACHE_OPT {
  LSC_CACHING_DEFAULT,        // .df
  LSC_CACHING_UNCACHED,       // .uc
  LSC_CACHING_CACHED,         // .ca
  LSC_CACHING_WRITEBACK,      // .wb
  LSC_CACHING_WRITETHROUGH,   // .wt
  LSC_CACHING_STREAMING,      // .st
  LSC_CACHING_READINVALIDATE, // .ri last use / invalidate after read
  LSC_CACHING_CONSTCACHED,    // .cc
};

enum LSC_OP {
  LSC_LOAD = 0x00,
  LSC_LOAD_STRIDED = 0x01, // aka "load_block"
  LSC_LOAD_QUAD = 0x02,    // aka "load_cmask"
  LSC_LOAD_BLOCK2D = 0x03,
  LSC_STORE = 0x04,
  LSC_STORE_STRIDED = 0x05, // aka "store_block"
  LSC_STORE_QUAD = 0x06,    // aka "store_cmask"
  LSC_STORE_BLOCK2D = 0x07,
  //
  LSC_ATOMIC_IINC = 0x08,
  LSC_ATOMIC_IDEC = 0x09,
  LSC_ATOMIC_LOAD = 0x0A,
  LSC_ATOMIC_STORE = 0x0B,
  LSC_ATOMIC_IADD = 0x0C,
  LSC_ATOMIC_ISUB = 0x0D,
  LSC_ATOMIC_SMIN = 0x0E,
  LSC_ATOMIC_SMAX = 0x0F,
  LSC_ATOMIC_UMIN = 0x10,
  LSC_ATOMIC_UMAX = 0x11,
  LSC_ATOMIC_ICAS = 0x12,
  LSC_ATOMIC_FADD = 0x13,
  LSC_ATOMIC_FSUB = 0x14,
  LSC_ATOMIC_FMIN = 0x15,
  LSC_ATOMIC_FMAX = 0x16,
  LSC_ATOMIC_FCAS = 0x17,
  LSC_ATOMIC_AND = 0x18,
  LSC_ATOMIC_OR = 0x19,
  LSC_ATOMIC_XOR = 0x1A,
  //
  LSC_LOAD_STATUS = 0x1B,
  LSC_STORE_UNCOMPRESSED = 0x1C,
  LSC_CCS_UPDATE = 0x1D,
  LSC_READ_STATE_INFO = 0x1E,
  LSC_FENCE = 0x1F,
  //
  LSC_APNDCTR_ATOMIC_ADD = 0x28,
  LSC_APNDCTR_ATOMIC_SUB = 0x29,
  LSC_APNDCTR_ATOMIC_STORE = 0x2A,

  LSC_INVALID = 0xFFFFFFFF,
};

enum class GenPrecision : unsigned char {
  INVALID = 0,

  U1 = 1,
  S1 = 2,
  U2 = 3,
  S2 = 4,
  U4 = 5,
  S4 = 6,
  U8 = 7,
  S8 = 8,
  BF16 = 9,  // bfloat16 (s:1, e:8, m:7)
  FP16 = 10, // half (1, 5, 10)
  BF8 = 11,  // bfloat8 (1, 5, 2)
  TF32 = 12, // TensorFloat (1, 8, 10), 19 bits
  TOTAL_NUM
};

#endif // LSC_INTRINSIC_ENUMS_H
