// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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

#define GET_TYPEDEF_CLASSES
#include "imex/Dialect/plier/PlierOpsTypes.h.inc"

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

} // namespace plier
