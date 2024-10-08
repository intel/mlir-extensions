//===- XeArch.cpp -  --------------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements uArch definition for supported HW configs for
//  operations such as dpas, load_2d, store_2d...
///
//===----------------------------------------------------------------------===//

#include "imex/Utils/XeArch.h"

namespace imex {

/// Checks Given A,B, C, D Matrix Data types to HW supported configs and
/// verifies HW restrictions for supported combinations.
mlir::LogicalResult XePVCuArch::checkSupportedDpasTypes(mlir::Operation *op,
                                                        mlir::Type AType,
                                                        mlir::Type BType,
                                                        mlir::Type CType,
                                                        mlir::Type DType) {

  if (AType.isF16() || BType.isF16()) {
    if (AType != BType || (CType && (!CType.isF32() && !CType.isF16())) ||
        (!DType.isF32() && !DType.isF16()))
      return op->emitOpError()
             << "Unsupported dpas combinations of Dst, Acc, A and B matrices, "
             << "Supported types are:\n"
             << "  Dst    |   Acc   |   A   |  B  \n"
             << " f, hf   |  f, hf  |   hf  |  hf \n"
             << "AType: " << AType << " BType: " << BType << " CType: " << CType
             << " DType: " << DType;
  } else if (AType.isBF16() || BType.isBF16()) {
    if (AType != BType || (CType && (!CType.isF32() && !CType.isBF16())) ||
        (!DType.isF32() && !DType.isBF16()))
      return op->emitOpError()
             << "Unsupported dpas combinations of Dst, Acc, A and B matrices, "
             << "Supported types are:\n"
             << "  Dst    |   Acc   |   A   |  B  \n"
             << " f, bf   |  f, bf  |   bf  |  bf \n"
             << "AType: " << AType << " BType: " << BType << " CType: " << CType
             << " DType: " << DType;
  } else if (AType.isTF32() || BType.isTF32()) {
    if (AType != BType || (CType && (!CType.isF32() && !DType.isF32())) ||
        (!DType.isF32()))
      return op->emitOpError()
             << "Unsupported dpas combinations of Dst, Acc, A and B matrices, "
             << "Supported types are:\n"
             << "  Dst    |   Acc   |   A    |   B  \n"
             << "   f     |    f    |  tf32  |  tf32 \n"
             << "AType: " << AType << " BType: " << BType << " CType: " << CType
             << " DType: " << DType;
  } else if (!(AType.isInteger(2) || AType.isInteger(4) ||
               AType.isInteger(8)) &&
             !(BType.isInteger(2) || BType.isInteger(4) ||
               BType.isInteger(8))) {
    return op->emitOpError()
           << "Unsupported dpas combinations of Dst, Acc, A and B matrices, "
           << "Supported types are:\n"
           << "  Dst     |   Acc    |         A           |         B          "
              " \n"
           << " ud, d    |  ud,d    |  ub,b,u4,s4,u2,s2   |  ub,b,u4,s4,u2,s2  "
           << "AType: " << AType << " BType: " << BType << " CType: " << CType
           << " DType: " << DType;
  }

  return mlir::success();
}

DPASConfig XePVCuArch::getDPASConfig(const int APrecision, const int BPrecision,
                                     const int CPrecision,
                                     const int DPrecision) {

  execSize = 16;

  return setDPASConfig(APrecision, BPrecision);
}
/// Gets 2D Block load parameters as per HW supported configs
/// Max Load Block size (H x W) PVC -> 2KB per load
/// Max Store Block size (H x W) PVC -> 512B, array_length = 1 only
/// Array_Length supported values for Load in the range of {1,2,4},
/// (array_len X block.width) cannot exceed 64B for Load
mlir::FailureOr<LoadStore2DConfig>
XePVCuArch::get2DLoadConfig(mlir::Operation *op, int element_data_size,
                            bool vnni, bool transpose) {
  LoadStore2DConfig loadParams;
  // both vnni and transpose be set at the same time
  if (vnni && transpose) {
    return op->emitOpError()
           << "transpose and transform are not supported together";
  }

  // FIXME: We do support transpose on f16 wtih transpose_bit_width==32,
  // disable check for now.
  // only d32 and d64 is supported for transpose operations
  // if ((transpose) && (element_data_size != 32 && element_data_size != 64)) {
  //   return op->emitOpError()
  //          << "transposed load only supports d32 and d64 data sizes. "
  //          << "Given element data size: d" << element_data_size;
  // }

  // only d8 and d16 are suported for VNNI transform operations
  if ((vnni) && (element_data_size != 8 && element_data_size != 16)) {
    return op->emitOpError()
           << "transformed load only supports d8 and d16 data sizes. "
           << "Given element data size: d" << element_data_size;
  }

  // clang-format off
// Turning off clang for better readability

  switch (element_data_size) {
  case 8:
    if (vnni) {
    //                                MinBlockHeight | MaxBlockHeight | MinBlockWidth | MaxBlockWidth | Array_Length | Width Restriction in Elements
      loadParams = setLoadStoreParams(     4,              32,               4,             16,         {1, 2, 4},           64);
    } else { // regular 2d block load
      loadParams = setLoadStoreParams(     1,              32,               4,             64,         {1, 2, 4},           64);
    }

    break;
  case 16:
    if (vnni) {
      loadParams = setLoadStoreParams(     2,              32,               2,             16,          {1, 2, 4},          32);
    } else { // regular 2d block load
      loadParams = setLoadStoreParams(     1,              32,               2,             32,          {1, 2, 4},          32);
    }

    break;
  case 32:
    if (transpose) {
      loadParams = setLoadStoreParams(     1,              32,               1,             8,           {1},                8);
    } else { // regular 2d block load
      loadParams = setLoadStoreParams(     1,              32,               1,            16,           {1, 2},             16);
    }

    break;
  case 64:
    if (transpose) {
      loadParams = setLoadStoreParams(     0,              8,                1,            4,            {1},                4);
    } else { // regular 2d block load
      loadParams = setLoadStoreParams(     1,              32,               1,            8,            {1},                8);
    }
    // clang-format on
    break;
  default:
    return op->emitOpError()
           << "unsupported data sizes for 2d block load. "
           << "Given element data size: d" << element_data_size;
    break;
  }
  loadParams.GRFDataSize.load = 2048;
  return loadParams;
}

mlir::FailureOr<LoadStore2DConfig>
XePVCuArch::get2DStoreConfig(int element_data_size) {

  LoadStore2DConfig storeParams;
  // clang-format off
// Turning off clang for better readability
  switch (element_data_size) {
  case 8:
  //                                 MinBlockHeight | MaxBlockHeight | MinBlockWidth | MaxBlockWidth | Array_Length | Width Restriction in Elements
    storeParams = setLoadStoreParams(     1,               8,               4,              64,         {1},                512);

    break;
  case 16:
    storeParams = setLoadStoreParams(     1,               8,               2,              32,         {1},                512);

    break;
  case 32:
    storeParams = setLoadStoreParams(     1,               8,               1,              16,         {1},                512);

    break;
  case 64:
    storeParams = setLoadStoreParams(     1,               8,               1,              8,          {1},                512);

    // clang-format on
    break;
  default:
    return mlir::failure();
  }

  storeParams.GRFDataSize.store = 512;

  return storeParams;
}

mlir::LogicalResult XeuArchInterface::isLegalDpasOp(mlir::Operation *op) {

  if (auto dpasOp = llvm::dyn_cast<mlir::xegpu::DpasOp>(op)) {
    auto lhsTy = dpasOp.getLhsType();
    auto rhsTy = dpasOp.getRhsType();
    auto accTy = dpasOp.getAcc() ? dpasOp.getAccType() : nullptr;
    auto resTy = dpasOp.getResultType();

    auto lhsRank = lhsTy.getRank();
    auto rhsRank = rhsTy.getRank();

    auto lhsShape = lhsTy.getShape();
    auto rhsShape = rhsTy.getShape();

    unsigned int APrecision = lhsTy.getElementTypeBitWidth();
    unsigned int BPrecision = rhsTy.getElementTypeBitWidth();
    unsigned int CPrecision = accTy ? rhsTy.getElementTypeBitWidth() : 0;
    unsigned int DPrecision = resTy.getElementTypeBitWidth();

    if (mlir::failed(this->checkSupportedDpasTypes(
            op, lhsTy.getElementType(), rhsTy.getElementType(),
            accTy ? accTy.getElementType() : nullptr,
            resTy.getElementType()))) {
      return op->emitOpError() << "Unsupported dpas config";
    }

    DPASConfig dpasParams =
        this->getDPASConfig(APrecision, BPrecision, CPrecision, DPrecision);

    unsigned int M = dpasParams.m; // repeatCount
    unsigned int N = dpasParams.n; // Execution size
    unsigned int K = dpasParams.k; // systolicDepth * opsPerChannel

    // TODO: restrict A to be 2D instead of both 2D and 3D
    auto aK = lhsRank == 3 ? lhsShape[1] * lhsShape[2] : lhsShape[1];
    auto bK = rhsRank == 3 ? rhsShape[0] * rhsShape[2] : rhsShape[0];

    if (!(lhsShape[0] >= 1 && lhsShape[0] <= M)) {
      return op->emitOpError()
             << "A matrix has incorrect size and does not match dpas config. "
             << "A[" << lhsShape[0] << "x" << aK << "], dpas config: "
             << "mxnxk = " << M << "x" << N << "x" << K << "\n";
    }

    if (aK != K || bK != K)
      return op->emitOpError() << "K-dim of A matrix (mxk), and B matrix (kxn) "
                                  "should be fixed to "
                               << K << " (dpas config: mxnxk = " << M << "x"
                               << N << "x" << K << ").\n";

    // Execution size for matrix B should match dpas params
    if (rhsShape[1] != N) {
      return op->emitOpError() << "N-dim of B matrix (kxn) should be fixed to "
                               << N << " (dpas config: mxnxk = " << M << "x"
                               << N << "x" << K << ").\n";
    }
  }
  return mlir::success();
}

mlir::LogicalResult XeuArchInterface::verify2dBlockRestriction(
    mlir::Operation *op, int width, int height, int array_len,
    int elemTyByteWidth, bool transpose, bool vnni,
    LoadStore2DConfig configParams, bool isLoad) {

  if (!llvm::isPowerOf2_32(array_len))
    return op->emitOpError() << "Array_Length must be in powers of 2. "
                             << "Given array_len: " << array_len;

  if (array_len > configParams.array_length.back())
    return op->emitOpError() << "Unsupported array size for transposed load.  "
                             << "Given array_len: " << array_len;

  if ((width < configParams.blockWidth.min ||
       width > configParams.blockWidth.max ||
       (width * elemTyByteWidth) % 4 != 0))
    return op->emitOpError()
           << "Invalid width size for 2D block load.  "
           << "The specification expects the value to "
           << "be in range [" << configParams.blockWidth.min << ", "
           << configParams.blockWidth.max << "], and "
           << "the total data size (width * elemTyBytes) to be multiple of 4. "
           << "Given width: " << width
           << " and data size: " << width * elemTyByteWidth;

  if (height < configParams.blockHeight.min ||
      height > configParams.blockHeight.max)
    return op->emitOpError() << "Invalid height size for 2D block load.  "
                             << "The specification expects the value to "
                             << "be in range [" << configParams.blockHeight.min
                             << ", " << configParams.blockHeight.max << "].";

  int GRFSize = width * height * array_len * elemTyByteWidth;
  int supportedSize =
      isLoad ? configParams.GRFDataSize.load : configParams.GRFDataSize.store;

  if (GRFSize > supportedSize)
    return op->emitOpError() << "GRF Data Size exceeds max supported GRF Size."
                             << "Supported GRFSize: " << supportedSize << "\n"
                             << "Given GRFSize: " << GRFSize;

  return mlir::success();
}

mlir::LogicalResult XeuArchInterface::isLegalLoad2dOp(mlir::Operation *op) {

  //  TODO: do we need to check cache hint?

  if (auto loadOp = llvm::dyn_cast<mlir::xegpu::LoadNdOp>(op)) {
    auto tdescTy = loadOp.getTensorDescType();

    // TODO: need more thinking on SLM
    if (tdescTy.getMemorySpace() == mlir::xegpu::MemorySpace::SLM)
      return mlir::success();

    int elementSize = loadOp.getTensorDescType().getElementTypeBitWidth();

    LoadStore2DConfig loadParams;
    bool vnni = loadOp.getPacked().value_or(false);
    bool transpose =
        loadOp.getTranspose() == llvm::ArrayRef<int64_t>({1, 0}) ? true : false;

    if (vnni && transpose) {
      return loadOp->emitOpError(
          "Transpose and VNNI are mutually exclusive. They are "
          "not supported by the PVC hardware at the same time.\n");
    }

    mlir::FailureOr<LoadStore2DConfig> configParams =
        this->get2DLoadConfig(op, elementSize, vnni, transpose);
    if (mlir::succeeded(configParams)) {

      auto width = tdescTy.getShape()[1];
      auto height = tdescTy.getShape()[0];
      auto array_len = tdescTy.getArrayLength();
      auto elemTyByteWidth =
          tdescTy.getElementType().getIntOrFloatBitWidth() / 8;

      return verify2dBlockRestriction(op, width, height, array_len,
                                      elemTyByteWidth, transpose, vnni,
                                      *configParams);
    } else {
      return loadOp->emitOpError("Invalid 2d block load parameters!\n");
    }
  }
  return mlir::success();
}

mlir::LogicalResult XeuArchInterface::isLegalStore2dOp(mlir::Operation *op) {

  if (auto storeOp = llvm::dyn_cast<mlir::xegpu::StoreNdOp>(op)) {
    auto tdescTy = storeOp.getTensorDescType();
    int elementSize = tdescTy.getElementTypeBitWidth();

    // TODO: need more thinking on SLM
    if (tdescTy.getMemorySpace() == mlir::xegpu::MemorySpace::SLM)
      return mlir::success();

    LoadStore2DConfig storeParams;
    bool vnni = false;
    bool transpose = false;

    mlir::FailureOr<LoadStore2DConfig> configParams =
        this->get2DStoreConfig(elementSize);
    if (mlir::succeeded(configParams)) {

      auto width = tdescTy.getShape()[1];
      auto height = tdescTy.getShape()[0];
      auto array_len = tdescTy.getArrayLength();
      auto elemTyByteWidth =
          tdescTy.getElementType().getIntOrFloatBitWidth() / 8;

      return verify2dBlockRestriction(op, width, height, array_len,
                                      elemTyByteWidth, transpose, vnni,
                                      *configParams, false);
    } else {
      return storeOp->emitOpError()
             << "unsupported data sizes for 2d block store. "
             << "Given element data size: d" << elementSize;
    }
  }

  return mlir::success();
}

mlir::LogicalResult XeuArchInterface::isLegalPrefetch2dOp(mlir::Operation *op) {

  if (auto prefetchOp = llvm::dyn_cast<mlir::xegpu::PrefetchNdOp>(op)) {
    auto tdescTy = prefetchOp.getTensorDescType();

    int elementSize = prefetchOp.getTensorDescType().getElementTypeBitWidth();

    mlir::FailureOr<LoadStore2DConfig> configParams =
        this->get2DPrefetchConfig(op, elementSize);
    if (mlir::succeeded(configParams)) {

      auto width = tdescTy.getShape()[1];
      auto height = tdescTy.getShape()[0];
      auto array_len = tdescTy.getArrayLength();
      auto elemTyByteWidth =
          tdescTy.getElementType().getIntOrFloatBitWidth() / 8;

      return verify2dPrefetchRestriction(op, width, height, array_len,
                                         elemTyByteWidth, *configParams);
    } else {
      return prefetchOp->emitOpError()
             << "Invalid 2d block load parameters for prefetch operation!\n";
    }
  }

  return mlir::success();
}

} // namespace imex
