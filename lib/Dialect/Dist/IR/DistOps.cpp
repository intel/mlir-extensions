//===- DistOps.cpp - Dist dialect  ------------------------------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the Dist dialect and its basic operations.
///
//===----------------------------------------------------------------------===//

#include <imex/Dialect/Dist/IR/DistOps.h>
#include <imex/Dialect/Dist/Utils/Utils.h>
#include <imex/Dialect/NDArray/IR/NDArrayOps.h>
#include <imex/Utils/PassUtils.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

namespace imex {
namespace dist {

void DistDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include <imex/Dialect/Dist/IR/DistOpsTypes.cpp.inc>
      >();
  addOperations<
#define GET_OP_LIST
#include <imex/Dialect/Dist/IR/DistOps.cpp.inc>
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include <imex/Dialect/Dist/IR/DistOpsAttrs.cpp.inc>
      >();
}

} // namespace dist
} // namespace imex

static mlir::LogicalResult
parseDistEnv(mlir::AsmParser &parser, ::mlir::Attribute &team,
             llvm::SmallVector<int64_t> &lOffsets,
             llvm::SmallVector<llvm::SmallVector<int64_t>> &lshapes) {
  llvm::SmallVector<llvm::SmallVector<int64_t>> dimensions;
  llvm::SmallVector<int64_t> dims;
  llvm::SmallVector<int64_t> lOffs;

  std::string tmp;
  if (parser.parseKeyword("team")) {
    return mlir::failure();
  }
  if (parser.parseEqual()) {
    return mlir::failure();
  }
  if (parser.parseAttribute(team)) {
    return mlir::failure();
  }

  if (parser.parseOptionalKeyword("loffs")) {
    dimensions.push_back({});
  } else {
    if (parser.parseEqual()) {
      return mlir::failure();
    }
    if (parser.parseCommaSeparatedList([&]() {
          int64_t v = ::mlir::ShapedType::kDynamic;
          auto opr = parser.parseOptionalInteger<int64_t>(v);
          if (!opr.has_value()) {
            if (parser.parseQuestion()) {
              return mlir::failure();
            }
          }
          lOffs.emplace_back(v);
          return mlir::success();
        })) {
      return mlir::failure();
    }
    auto n = lOffs.size();

    if (parser.parseKeyword("lparts")) {
      return mlir::failure();
    }
    if (parser.parseEqual()) {
      return mlir::failure();
    }
    auto prs = [&]() {
      if (parser.parseDimensionList(dims, true, false) || dims.size() != n) {
        return mlir::failure();
      }
      dimensions.emplace_back(dims);
      dims.clear();
      return mlir::success();
    };
    if (parser.parseCommaSeparatedList(prs) ||
        !(dimensions.size() == 1 || dimensions.size() == 3)) {
      return mlir::failure();
    }
  }

  lOffsets = std::move(lOffs);
  lshapes = std::move(dimensions);
  return mlir::success();
}

static void
printDistEnv(mlir::AsmPrinter &printer, ::mlir::Attribute team,
             const llvm::ArrayRef<int64_t> lOffs,
             const llvm::SmallVector<llvm::SmallVector<int64_t>> lshapes) {
  if (team) {
    printer << "team = " << team;

    auto n = lOffs.size();
    if (n) {
      auto printEl = [&](int64_t v, char sep, bool last) {
        if (v == ::mlir::ShapedType::kDynamic) {
          printer << '?';
        } else {
          printer << v;
        }
        if (!last)
          printer << sep;
      };

      printer << " loffs = ";
      for (size_t i = 0; i < n; ++i) {
        printEl(lOffs[i], ',', i >= n - 1);
      }

      n = lshapes.size();
      printer << " lparts = ";
      for (size_t i = 0; i < n; ++i) {
        auto shape = lshapes[i];
        for (size_t j = 0; j < shape.size(); ++j) {
          printEl(shape[j], 'x', j >= shape.size() - 1);
        }
        if (i < n - 1) {
          printer << ',';
        }
      }
    }
  }
}

#include <imex/Dialect/Dist/IR/DistOpsDialect.cpp.inc>
#define GET_TYPEDEF_CLASSES
#include <imex/Dialect/Dist/IR/DistOpsTypes.cpp.inc>
#define GET_ATTRDEF_CLASSES
#include <imex/Dialect/Dist/IR/DistOpsAttrs.cpp.inc>
#define GET_OP_CLASSES
#include <imex/Dialect/Dist/IR/DistOps.cpp.inc>

namespace imex {
namespace dist {

DistEnvAttr DistEnvAttr::get(
    ::mlir::Attribute team, ::llvm::ArrayRef<int64_t> lOffsets,
    ::mlir::SmallVector<::mlir::SmallVector<int64_t>> partsShapes) {
  assert(partsShapes.size() == 3 || partsShapes.size() == 1);
  assert(team);
  return get(team.getContext(), team, lOffsets, partsShapes);
}

DistEnvAttr DistEnvAttr::get(::mlir::Attribute team, int64_t rank) {
  assert(team);
  ::mlir::SmallVector<::mlir::SmallVector<int64_t>> partsShapes(
      rank ? 3 : 1,
      ::mlir::SmallVector<int64_t>(rank, ::mlir::ShapedType::kDynamic));
  ::mlir::SmallVector<int64_t> lOffsets(rank, ::mlir::ShapedType::kDynamic);
  return get(team.getContext(), team, lOffsets, partsShapes);
}

DistEnvAttr DistEnvAttr::cloneWithDynOffsAndDims() const {
  return get(getTeam(), getLOffsets().size());
}

void InitDistArrayOp::build(::mlir::OpBuilder &odsBuilder,
                            ::mlir::OperationState &odsState,
                            ::mlir::Attribute team,
                            ::mlir::ArrayRef<int64_t> g_shape,
                            ::mlir::ValueRange l_offset,
                            ::mlir::ValueRange parts,
                            ::mlir::ArrayRef<::mlir::Attribute> environments,
                            ::mlir::ArrayRef<int64_t> s_Offs) {
  assert(l_offset.size() == g_shape.size());
  auto elTyp = mlir::cast<::imex::ndarray::NDArrayType>(parts.front().getType())
                   .getElementType();
  ::mlir::SmallVector<::mlir::SmallVector<int64_t>> shapes;
  for (auto p : parts) {
    assert(!isDist(p));
    shapes.emplace_back(
        mlir::cast<::imex::ndarray::NDArrayType>(p.getType()).getShape());
  }
  auto resShape = getShapeFromValues(l_offset);
  ::mlir::ArrayRef<int64_t> lOffs =
      s_Offs.size() ? s_Offs : resShape;
  ::mlir::SmallVector<::mlir::Attribute> nEnvs(environments);
  nEnvs.emplace_back(::imex::dist::DistEnvAttr::get(team, lOffs, shapes));
  auto arType = ::imex::ndarray::NDArrayType::get(g_shape, elTyp, nEnvs);
  build(odsBuilder, odsState, arType, l_offset, parts);
}

void PartsOfOp::build(::mlir::OpBuilder &odsBuilder,
                      ::mlir::OperationState &odsState, ::mlir::Value ary) {
  auto pTypes =
      getPartsTypes(mlir::cast<::imex::ndarray::NDArrayType>(ary.getType()));
  assert(pTypes.size() == 1 || pTypes.size() == 3 ||
         (false && "Number of local parts must be 1 or 3"));
  build(odsBuilder, odsState, pTypes, ary);
}

::mlir::LogicalResult PartsOfOp::verify() {
  if (this->getNumResults() == 1 || (this->getNumResults() == 3)) {
    return ::mlir::success();
  }
  return ::mlir::failure();
}

::mlir::LogicalResult EWBinOp::verify() {
  if (isDist(getResult()) && isDist(getLhs()) && isDist(getRhs())) {
    return ::mlir::success();
  }
  return ::mlir::failure();
}

} // namespace dist
} // namespace imex
