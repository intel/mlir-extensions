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

#include "mlir-extensions/Dialect/plier/PlierOpsDialect.h.inc"
//#include "mlir-extensions/Dialect/plier/PlierOpsEnums.h.inc"
#define GET_OP_CLASSES
#include "mlir-extensions/Dialect/plier/PlierOps.h.inc"

namespace plier {
namespace attributes {
llvm::StringRef getFastmathName();
llvm::StringRef getJumpMarkersName();
llvm::StringRef getParallelName();
llvm::StringRef getMaxConcurrencyName();
llvm::StringRef getForceInlineName();
llvm::StringRef getOptLevelName();
llvm::StringRef getGpuRangeName();
} // namespace attributes

namespace detail {
struct PyTypeStorage;
struct LiteralTypeStorage;
struct TypeVarStorage;

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

enum { OperatorsCount = llvm::array_lengthof(detail::OperatorNames) };

mlir::ArrayRef<detail::OperatorNamePair> getOperators();

class PyType : public mlir::Type::TypeBase<::plier::PyType, mlir::Type,
                                           ::plier::detail::PyTypeStorage> {
public:
  using Base::Base;

  static PyType get(mlir::MLIRContext *context, mlir::StringRef name);
  static PyType getUndefined(mlir::MLIRContext *context);

  mlir::StringRef getName() const;
};

class LiteralType
    : public mlir::Type::TypeBase<::plier::LiteralType, mlir::Type,
                                  ::plier::detail::LiteralTypeStorage> {
public:
  using Base::Base;

  static LiteralType get(mlir::Attribute value);

  mlir::Attribute getValue() const;
};

class SliceType : public ::mlir::Type::TypeBase<SliceType, ::mlir::Type,
                                                ::mlir::TypeStorage> {
public:
  using Base::Base;

  static SliceType get(mlir::MLIRContext *context);
};

class TypeVar : public mlir::Type::TypeBase<::plier::TypeVar, mlir::Type,
                                            ::plier::detail::TypeVarStorage> {
public:
  using Base::Base;

  static TypeVar get(mlir::Type type);

  mlir::Type getType() const;
};

} // namespace plier
