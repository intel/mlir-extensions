//===- Utils.h - Utils for Dist dialect  -----------------------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the utils for the dist dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _DIST_UTILS_H_INCLUDED_
#define _DIST_UTILS_H_INCLUDED_

#include <imex/Dialect/Dist/IR/DistOps.h>
#include <imex/Dialect/DistRuntime/IR/DistRuntimeOps.h>
#include <imex/Dialect/NDArray/IR/NDArrayOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Dominance.h>

#include <algorithm>
#include <vector>

namespace imex {
namespace dist {

// *******************************
// ***** Some helper functions ***
// *******************************

/// @return true if atribute is a DistEnvAttr
inline bool isDist(const ::mlir::Attribute &a) {
  return ::mlir::isa<::imex::dist::DistEnvAttr>(a);
}

/// @return true if type has a DistEnvAttr
inline bool isDist(const ::imex::ndarray::NDArrayType &t) {
  return ::imex::ndarray::hasEnv<::imex::dist::DistEnvAttr>(t);
}

/// @return true if type has a DistEnvAttr
inline bool isDist(const ::mlir::Type &t) {
  auto arType = t.dyn_cast<::imex::ndarray::NDArrayType>();
  return arType ? isDist(arType) : false;
}

/// @return true if value is a DistEnvAttr
inline bool isDist(const ::mlir::Value &v) {
  auto arType = v.getType().dyn_cast<::imex::ndarray::NDArrayType>();
  return arType ? isDist(arType) : false;
}

/// @return first DistEnvAttr, null-attr if none exists
inline ::imex::dist::DistEnvAttr
getDistEnv(const ::imex::ndarray::NDArrayType &t) {
  for (auto a : t.getEnvironments()) {
    if (auto d = ::mlir::dyn_cast<::imex::dist::DistEnvAttr>(a)) {
      return d;
    }
  }
  return {};
}

/// @return return NDArray's env attributes except DistEnvAttrs
inline ::mlir::SmallVector<::mlir::Attribute>
getNonDistEnvs(const ::imex::ndarray::NDArrayType &t) {
  ::mlir::SmallVector<::mlir::Attribute> envs;
  std::copy_if(t.getEnvironments().begin(), t.getEnvironments().end(),
               std::back_inserter(envs), [](auto i) { return !isDist(i); });
  return envs;
}

/// @return clone of type, but with dynamic shapes (local and global)
inline ::imex::ndarray::NDArrayType
cloneWithDynEnv(const ::imex::ndarray::NDArrayType &ary) {
  auto oEnvs = ary.getEnvironments();
  ::mlir::SmallVector<::mlir::Attribute> envs;
  for (auto e : oEnvs) {
    if (auto a = ::mlir::dyn_cast<::imex::dist::DistEnvAttr>(e)) {
      e = a.cloneWithDynOffsAndDims();
    }
    envs.emplace_back(e);
  }
  return ::imex::ndarray::NDArrayType::get(ary.getShape(), ary.getElementType(),
                                           envs);
}

/// @return clone of type, but with dynamic shapes (local and global)
inline ::imex::ndarray::NDArrayType
cloneWithShape(const ::imex::ndarray::NDArrayType &ary,
               const ::mlir::ArrayRef<int64_t> shape) {
  auto oEnvs = ary.getEnvironments();
  ::mlir::SmallVector<::mlir::Attribute> envs;
  for (auto e : oEnvs) {
    if (auto a = ::mlir::dyn_cast<::imex::dist::DistEnvAttr>(e)) {
      e = ::imex::dist::DistEnvAttr::get(a.getTeam(), shape.size());
    }
    envs.emplace_back(e);
  }
  return ::imex::ndarray::NDArrayType::get(shape, ary.getElementType(), envs);
}

/// @return clone of type, but without dist-env and with dynamic shapes
inline ::imex::ndarray::NDArrayType
cloneAsDynNonDist(const ::imex::ndarray::NDArrayType &ary) {
  auto envs = getNonDistEnvs(ary);
  if (ary.hasUnitSize()) {
    return ::imex::ndarray::NDArrayType::get(ary.getShape(),
                                             ary.getElementType(), envs);
  } else {
    return ::imex::ndarray::NDArrayType::get(
        ::mlir::SmallVector<int64_t>(ary.getRank(),
                                     ::mlir::ShapedType::kDynamic),
        ary.getElementType(), envs);
  }
}

/// @return clone of type, but without dist-env
inline ::imex::ndarray::NDArrayType
cloneAsNonDist(const ::imex::ndarray::NDArrayType &ary) {
  auto envs = getNonDistEnvs(ary);
  return ::imex::ndarray::NDArrayType::get(ary.getShape(), ary.getElementType(),
                                           envs);
}

/// @return types of NDArray's parts if it has distenv, empty vector otherwise
inline ::imex::TypVec
getPartsTypes(const ::imex::ndarray::NDArrayType &arType) {
  ::imex::TypVec res;
  if (auto dEnv = getDistEnv(arType)) {
    auto shapes = dEnv.getPartsShapes();
    auto envs = getNonDistEnvs(arType);
    if (arType.getRank() == 0 && shapes.size() == 0) {
      res.emplace_back(::imex::ndarray::NDArrayType::get(
          {1}, arType.getElementType(), envs)); // FIXME env layout
    } else {
      for (auto shp : shapes) {
        res.emplace_back(::imex::ndarray::NDArrayType::get(
            shp, arType.getElementType(), envs)); // FIXME env layout
      }
    }
  }
  return res;
}

/// Create a distributed array from a NDArray and meta data
inline ::mlir::Value
createDistArray(const ::mlir::Location &loc, ::mlir::OpBuilder &builder,
                ::mlir::Attribute team, ::mlir::ArrayRef<int64_t> gshape,
                ::mlir::ValueRange loffs, ::mlir::ValueRange parts,
                ::mlir::ArrayRef<int64_t> sOffs = {}) {
  assert(parts.size() == 1 || parts.size() == 3);
  ::imex::ValVec nParts;
  auto p = parts.front().getType().cast<::imex::ndarray::NDArrayType>();
  auto envs = p.getEnvironments();
  auto rank = p.getRank();
  assert(rank || parts.size() == 1);

  if (parts.size() == 1 && rank) {
    auto elType = p.getElementType();
    ::imex::ValVec shp(rank, createIndex(loc, builder, 0));
    auto lHalo = builder.create<::imex::ndarray::CreateOp>(
        loc, shp, ::imex::ndarray::fromMLIR(elType), nullptr, envs);
    auto rHalo = builder.create<::imex::ndarray::CreateOp>(
        loc, shp, ::imex::ndarray::fromMLIR(elType), nullptr, envs);
    nParts = {lHalo, parts.front(), rHalo};
  } else {
    nParts = parts;
    assert(parts.size() == 3 || rank == 0);
  }

  for (auto x : parts)
    assert(!isDist(x));

  return builder.create<::imex::dist::InitDistArrayOp>(loc, team, gshape, loffs,
                                                       nParts, envs, sOffs);
}

inline ::mlir::Value
createDistArray(const ::mlir::Location &loc, ::mlir::OpBuilder &builder,
                ::mlir::Attribute team, ::imex::ValVec gshape,
                ::mlir::ValueRange loffs, ::mlir::ValueRange parts) {
  auto gshp = mkConstant(gshape);
  return createDistArray(loc, builder, team, gshp, loffs, parts);
}

// create operation returning global shape of distributed array
inline ::imex::ValVec createGlobalShapeOf(const ::mlir::Location &loc,
                                          ::mlir::OpBuilder &builder,
                                          ::mlir::Value ary) {
  auto gshp = ary.getType().cast<::imex::ndarray::NDArrayType>().getShape();
  ::imex::ValVec res;
  for (auto d : gshp) {
    res.emplace_back(createIndex(loc, builder, d));
  }
  return res;
}

// create operation returning local offsets of distributed array
inline ::mlir::ValueRange createLocalOffsetsOf(const ::mlir::Location &loc,
                                               ::mlir::OpBuilder &builder,
                                               ::mlir::Value ary) {
  return builder.create<::imex::dist::LocalOffsetsOfOp>(loc, ary).getLOffsets();
}

// create operation returning halo of distributed array
inline ::mlir::ValueRange createPartsOf(const ::mlir::Location &loc,
                                        ::mlir::OpBuilder &builder,
                                        ::mlir::Value ary) {
  return builder.create<::imex::dist::PartsOfOp>(loc, ary).getParts();
}

inline ::mlir::Value createNProcs(const ::mlir::Location &loc,
                                  ::mlir::OpBuilder &builder,
                                  ::mlir::Attribute team) {
  return builder.createOrFold<::imex::distruntime::TeamSizeOp>(loc, team);
}

inline ::mlir::Value createPRank(const ::mlir::Location &loc,
                                 ::mlir::OpBuilder &builder,
                                 ::mlir::Attribute team) {
  return builder.createOrFold<::imex::distruntime::TeamMemberOp>(loc, team);
}

// create operation returning the re-partitioned array
inline ::mlir::Value createRePartition(const ::mlir::Location &loc,
                                       ::mlir::OpBuilder &builder,
                                       ::mlir::Value ary,
                                       const ::mlir::ValueRange &tOffs = {},
                                       const ::mlir::ValueRange &tSzs = {}) {
  auto retTyp =
      ary.getType().cast<::imex::ndarray::NDArrayType>().cloneWithDynDims();
  return builder.create<::imex::dist::RePartitionOp>(loc, retTyp, ary, tOffs,
                                                     tSzs);
}

inline auto createDefaultPartition(const ::mlir::Location &loc,
                                   ::mlir::OpBuilder &builder,
                                   ::mlir::Attribute team,
                                   ::imex::ValVec gShape) {
  auto nProcs = createNProcs(loc, builder, team);
  auto pRank = createPRank(loc, builder, team);
  return builder.create<::imex::dist::DefaultPartitionOp>(loc, nProcs, pRank,
                                                          gShape);
}

/// @brief compute overlap of given slices with local off/shape
/// @param lShape local shape
/// @param lOff local offset
/// @param slcOff slice's offset
/// @param slcSize slice's size
/// @param slcStride slice's stride
/// @return offsets and sizes of overlap and leading/skipped elements of slice
template <typename I, typename R = I>
inline std::tuple<R, R, R>
createOverlap(::mlir::Location loc, ::mlir::OpBuilder rewriter,
              const ::mlir::ValueRange &lOffs, const ::mlir::ValueRange &lShape,
              const I &slcOffs, const I &slcSizes, const I &slcStrides) {
  auto zero = easyIdx(loc, rewriter, 0);
  auto one = easyIdx(loc, rewriter, 1);
  auto rank = lShape.size();

  R resOffs(rank, zero.get());
  R resSlcOffs(rank, zero.get());
  R resSizes(slcSizes.begin(), slcSizes.end());

  for (unsigned i = 0; i < rank; ++i) {
    // Get the vals from dim
    auto lOff = easyIdx(loc, rewriter, lOffs[i]);
    auto slcOff = easyIdx(loc, rewriter, slcOffs[i]);
    auto slcStride = easyIdx(loc, rewriter, slcStrides[i]);
    auto slcSize = easyIdx(loc, rewriter, slcSizes[i]);
    auto lSize = easyIdx(loc, rewriter, lShape[i]);

    // last index of slice
    auto slcEnd = slcOff + slcSize * slcStride;
    // last index of local partition
    auto lEnd = lOff + lSize;

    auto maxOff = lOff.max(slcOff);
    auto stride_1 = slcStride - one;
    // slc<local    s            l                  s
    // local<slc    s            s                  s
    auto resOff =
        slcOff + (((maxOff + stride_1 - slcOff) / slcStride) * slcStride);
    auto resSz = (slcEnd.min(lEnd) + stride_1 - resOff).max(zero) / slcStride;
    auto resSlcOff = ((resOff - slcOff) / slcStride).min(slcSize);

    resOffs[i] = resOff.get();
    resSizes[i] = resSz.get();
    resSlcOffs[i] = resSlcOff.get();
  }

  return {resOffs, resSizes, resSlcOffs};
};

} // namespace dist
} // namespace imex

#endif // _DIST_UTILS_H_INCLUDED_
