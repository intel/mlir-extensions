//===- XeArch.h - XeuArch interface  Functions --------------------*- C++
//-*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines uArch definition for supported HW configs for
// operations such as dpas, load_2d, store_2d, prefetch_2d, etc.
//
//===----------------------------------------------------------------------===//

#ifndef _IMEX_XEARCH_H_
#define _IMEX_XEARCH_H_

#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/XeGPU/IR/XeGPU.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include "imex/Dialect/XeTile/IR/XeTileOps.h"
#include "imex/Utils/DebugUtils.h"
#include "imex/Utils/XeCommon.h"

namespace imex {

struct Range {
  int min;
  int max;
  bool contains(int val) { return val >= min && val <= max; }
};

// DPAS m x n x k
struct DPASConfig {
  DPASConfig() = default;

  unsigned m = 0;
  unsigned n = 0;
  unsigned k = 0;
};

struct GRFSize {
  int store = 0;
  int load = 0;
};

struct LoadStore2DConfig {
  Range blockHeight;                   // # of rows
  Range blockWidth;                    // # of elements
  llvm::SmallVector<int> array_length; // # of blocks to read/write memory
  int restriction;                     // Max Width in bytes
  GRFSize GRFDataSize;                 // Max GRF Data for load and store
  int minPitch;                        // Min pitch in bytes
  int pitchMultiple;                   // Pitch must be multiple in bytes of
                                       //   this value
};

/// This Base class provides uArch interface for defining HW supported configs
/// that is used to verify XeGPU dialect operations. This gets inherited to
/// platform specific classes that defines HW specific restrictions.
class XeuArchInterface {
public:
  mlir::StringRef gpuArch;

  XeuArchInterface(mlir::StringRef uArch) {
    this->gpuArch = uArch;

    oneGRFSizeBits = 512;
    // DPAS related params - default to PVC
    repeatCount = 8;
    sDepth = 8;
    execSize = 16;
  }

  virtual mlir::LogicalResult checkSupportedDpasTypes(mlir::Operation *op,
                                                      mlir::Type AType,
                                                      mlir::Type BType,
                                                      mlir::Type CType,
                                                      mlir::Type DType) = 0;

  virtual DPASConfig getDPASConfig(const int APrecision, const int BPrecision,
                                   const int CPrecision,
                                   const int DPrecision) = 0;

  virtual mlir::FailureOr<LoadStore2DConfig>
  get2DLoadConfig(mlir::Operation *op, int element_data_size, bool vnni,
                  bool transpose) = 0;

  virtual mlir::FailureOr<LoadStore2DConfig>
  get2DPrefetchConfig(mlir::Operation *op, int element_data_size) = 0;

  virtual mlir::FailureOr<LoadStore2DConfig>
  get2DStoreConfig(int element_data_size) = 0;

  mlir::LogicalResult verify2dBlockRestriction(mlir::Operation *op, int width,
                                               int height, int array_len,
                                               int elemTyBitWidth,
                                               bool transpose, bool vnni,
                                               LoadStore2DConfig configParams,
                                               bool isLoad = true);

  virtual mlir::LogicalResult
  verify2dPrefetchRestriction(mlir::Operation *op, int width, int height,
                              int array_len, int elemTyBitWidth,
                              LoadStore2DConfig configParams) = 0;
  mlir::LogicalResult isLegalDpasOp(mlir::Operation *op);

  mlir::LogicalResult isLegalLoad2dOp(mlir::Operation *op);

  mlir::LogicalResult isLegalStore2dOp(mlir::Operation *op);

  mlir::LogicalResult isLegalPrefetch2dOp(mlir::Operation *op);
  unsigned int getOneGRFSizeBits() const { return oneGRFSizeBits; };

protected:
  ~XeuArchInterface() {}
  unsigned int oneGRFSizeBits;
  unsigned int repeatCount;
  unsigned int sDepth;
  unsigned int execSize; // Maximum number of channels allowed. Number of
                         // Channels operating in parallel for dpas instruction

  /// D (MxN) = C (MxN) + A (MxK) x B (KxN)
  /// M = Repeat Count
  /// N = Fixed Exec Size ==> PVC = 16
  /// K = Systolic Depth * OPS_PER_CHAN; XEHP+, only supports depth of 8
  /// OPS_PER_CHAN
  ///  1: TF32
  ///  2: BF, HF
  ///  4: F8
  ///  8: U4/S4/U2/S2
  DPASConfig setDPASConfig(const int APrecision, const int BPrecision) {

    DPASConfig dpasParams;
    unsigned int opsPerChannel =
        std::max(std::min(32 / std::max(APrecision, BPrecision), 8), 1);
    dpasParams.m = repeatCount;
    dpasParams.n = execSize;
    dpasParams.k = sDepth * opsPerChannel;

    return dpasParams;
  }

  LoadStore2DConfig setLoadStoreParams(int minHeight, int maxHeight,
                                       int minWidth, int maxWidth,
                                       llvm::SmallVector<int> array_len,
                                       int restriction) {
    LoadStore2DConfig configParams;
    configParams.blockHeight.min = minHeight;
    configParams.blockHeight.max = maxHeight;

    configParams.blockWidth.min = minWidth;
    configParams.blockWidth.max = maxWidth;

    configParams.array_length = array_len;
    configParams.restriction = restriction;
    configParams.minPitch = 64;
    configParams.pitchMultiple = 16;
    return configParams;
  }
};
/// This class defines PVC GPU Architecture specific HW config parameters for
/// various dpas related operations like load2d, store2d, prefetch2d. This
/// is used to verify XeGPU Dialect operations for legal ops definitions.
class XePVCuArch : public XeuArchInterface {

public:
  XePVCuArch() : XeuArchInterface("pvc") {}
  virtual ~XePVCuArch() {}

  virtual mlir::LogicalResult
  checkSupportedDpasTypes(mlir::Operation *op, mlir::Type AType,
                          mlir::Type BType, mlir::Type CType,
                          mlir::Type DType) override;

  virtual DPASConfig getDPASConfig(const int APrecision, const int BPrecision,
                                   const int CPrecision,
                                   const int DPrecision) override;

  virtual mlir::FailureOr<LoadStore2DConfig>
  get2DLoadConfig(mlir::Operation *op, int element_data_size, bool vnni,
                  bool transpose) override;

  virtual mlir::FailureOr<LoadStore2DConfig>
  get2DPrefetchConfig(mlir::Operation *op, int element_data_size) override {
    // Load and prefetch configs are same for PVC.
    return get2DLoadConfig(op, element_data_size, false, false);
  }

  mlir::LogicalResult
  verify2dPrefetchRestriction(mlir::Operation *op, int width, int height,
                              int array_len, int elemTyBitWidth,
                              LoadStore2DConfig configParams) override {
    return verify2dBlockRestriction(op, width, height, array_len,
                                    elemTyBitWidth, false, false, configParams,
                                    true);
  }
  virtual mlir::FailureOr<LoadStore2DConfig>
  get2DStoreConfig(int element_data_size) override;
};

} // namespace imex

#endif
