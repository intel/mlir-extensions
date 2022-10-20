// Copyright 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Interfaces/ViewLikeInterface.h>

#include "imex/Dialect/plier/PlierOpsDialect.h.inc"
//#include "imex/Dialect/plier/PlierOpsEnums.h.inc"
#define GET_OP_CLASSES
#include "imex/Dialect/plier/PlierOps.h.inc"

namespace plier {
namespace detail {
struct PyTypeStorage;

struct OperatorNamePair {
  mlir::StringRef op;
  mlir::StringRef name;
};

static const constexpr OperatorNamePair OperatorNames[] = {
    // clang-format off
    {"+", "add"}, // binary
    {"+", "pos"}, // unary
    {"-", "sub"}, // binary
    {"-", "neg"}, // unary
    {"*", "mul"},
    {"**", "pow"},
    {"/", "truediv"},
    {"//", "floordiv"},
    {"%", "mod"},
    {"@", "matmul"},
    {"&", "and"},
    {"|", "or"},
    {"^", "xor"},
    {"~", "invert"},
    {">>", "rshift"},
    {"<<", "lshift"},
    {"not", "not"},

    {">", "gt"},
    {">=", "ge"},
    {"<", "lt"},
    {"<=", "le"},
    {"!=", "ne"},
    {"==", "eq"},
    // clang-format on
};
} // namespace detail

enum { OperatorsCount = std::size(detail::OperatorNames) };

mlir::ArrayRef<detail::OperatorNamePair> getOperators();

class PyType : public mlir::Type::TypeBase<::plier::PyType, mlir::Type,
                                           ::plier::detail::PyTypeStorage> {
public:
  using Base::Base;

  static PyType get(mlir::MLIRContext *context, mlir::StringRef name);
  static PyType getUndefined(mlir::MLIRContext *context);

  mlir::StringRef getName() const;
};

class SliceType : public ::mlir::Type::TypeBase<SliceType, ::mlir::Type,
                                                ::mlir::TypeStorage> {
public:
  using Base::Base;

  static SliceType get(mlir::MLIRContext *context);
};

} // namespace plier
