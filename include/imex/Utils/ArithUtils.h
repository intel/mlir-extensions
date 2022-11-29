//===- PassUtils.h - Pass Utility Functions --------------------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines utility functions for writing passes.
//
//===----------------------------------------------------------------------===//

#ifndef _IMEX_ARITHUTILS_H_
#define _IMEX_ARITHUTILS_H_

#include "PassUtils.h"
#include <climits>
#include <mlir/Dialect/Arith/IR/Arith.h>

namespace imex {

/// type to indicate no comparison value
struct NoCmp {};

/// Generic expression template representing a binary operation, optionally
/// enriched with a comparison.
template <typename LHS, typename OP, typename RHS, typename CMP = NoCmp>
struct EasyExpr {
  typedef typename OP::ElType ElType;

  EasyExpr(LHS const &lhs, RHS const &rhs, CMP const &cmp = {})
      : _lhs(lhs), _rhs(rhs), _cmp(cmp) {}

  /// triggers evaluation of template expression(s)
  ElType get(const ::mlir::Location &loc, ::mlir::OpBuilder &builder) const {
    if constexpr (std::is_same_v<CMP, NoCmp>) {
      return OP::eval(loc, builder, _lhs.get(loc, builder),
                      _rhs.get(loc, builder));
    } else if constexpr (!std::is_same_v<CMP, NoCmp>) {
      return OP::eval(loc, builder, _lhs.get(loc, builder),
                      _rhs.get(loc, builder), _cmp.get(loc, builder));
    }
  }

  LHS const &_lhs;
  RHS const &_rhs;
  CMP const &_cmp;
};

/// Generic operation with an eval method which creates MLIR operations
/// the eval method is called by the binding template expression
/// Parameter arg is currently casted to CmpIPredicate (required by comparison
/// ops only)
template <typename T, typename OP, uint64_t arg = ULLONG_MAX> struct EasyOp {
  using ElType = T;
  /// eval generic binary ops
  static ElType eval(const ::mlir::Location &loc, ::mlir::OpBuilder &builder,
                     T const &l, T const &r) {
    if constexpr (arg < ULLONG_MAX) {
      return builder.create<OP>(loc, (::mlir::arith::CmpIPredicate)arg, l, r);
    } else if constexpr (arg >= ULLONG_MAX) {
      return builder.create<OP>(loc, l, r);
    }
  }
  /// eval comparison op
  template <typename A>
  static ElType eval(const ::mlir::Location &loc, ::mlir::OpBuilder &builder,
                     T const &l, T const &r, A const &a) {
    if constexpr (arg < ULLONG_MAX) {
      return builder.create<OP>(loc, arg, a, l, r);
    } else if constexpr (arg >= ULLONG_MAX) {
      return builder.create<OP>(loc, a, l, r);
    }
  }
};

/// addition of the expressions
template <typename LHS, typename RHS>
EasyExpr<LHS, EasyOp<typename LHS::ElType, ::mlir::arith::AddIOp>, RHS>
operator+(LHS const &l, RHS const &r) {
  return {l, r};
}

/// subtraction of the expressions
template <typename LHS, typename RHS>
EasyExpr<LHS, EasyOp<typename LHS::ElType, ::mlir::arith::SubIOp>, RHS>
operator-(LHS const &l, RHS const &r) {
  return {l, r};
}

// multiplication of the expressions
template <typename LHS, typename RHS>
EasyExpr<LHS, EasyOp<typename LHS::ElType, ::mlir::arith::MulIOp>, RHS>
operator*(LHS const &l, RHS const &r) {
  return {l, r};
}

/// division of the expressions
template <typename LHS, typename RHS>
EasyExpr<LHS, EasyOp<typename LHS::ElType, ::mlir::arith::DivSIOp>, RHS>
operator/(LHS const &l, RHS const &r) {
  return {l, r};
}

/// generic comparison of expressions
template <::mlir::arith::CmpIPredicate CP, typename LHS, typename RHS>
EasyExpr<LHS, EasyOp<typename LHS::ElType, ::mlir::arith::CmpIOp, (uint64_t)CP>,
         RHS>
easyCompare(LHS const &l, RHS const &r) {
  return {l, r};
}

/// equal
template <typename LHS, typename RHS> auto easyEQ(LHS const &l, RHS const &r) {
  return easyCompare<::mlir::arith::CmpIPredicate::eq>(l, r);
}
/// non-equal
template <typename LHS, typename RHS> auto easyNE(LHS const &l, RHS const &r) {
  return easyCompare<::mlir::arith::CmpIPredicate::ne>(l, r);
}
/// signed-lower-than comparison of expressions
template <typename LHS, typename RHS> auto easySLT(LHS const &l, RHS const &r) {
  return easyCompare<::mlir::arith::CmpIPredicate::slt>(l, r);
}
/// unsigned-lower-than comparison of expressions
template <typename LHS, typename RHS> auto easyULT(LHS const &l, RHS const &r) {
  return easyCompare<::mlir::arith::CmpIPredicate::ult>(l, r);
}
/// signed greater-than
template <typename LHS, typename RHS> auto easySGT(LHS const &l, RHS const &r) {
  return easyCompare<::mlir::arith::CmpIPredicate::sgt>(l, r);
}
/// unsigned greater-than
template <typename LHS, typename RHS> auto easyUGT(LHS const &l, RHS const &r) {
  return easyCompare<::mlir::arith::CmpIPredicate::ugt>(l, r);
}
/// signed greater-equal
template <typename LHS, typename RHS> auto easySGE(LHS const &l, RHS const &r) {
  return easyCompare<::mlir::arith::CmpIPredicate::sge>(l, r);
}
/// unsigned greater-equal
template <typename LHS, typename RHS> auto easyUGE(LHS const &l, RHS const &r) {
  return easyCompare<::mlir::arith::CmpIPredicate::uge>(l, r);
}

/// select lhs or rhs dependent on comparison
template <typename CMP, typename LHS, typename RHS>
EasyExpr<LHS, EasyOp<typename LHS::ElType, ::mlir::arith::SelectOp>, RHS, CMP>
easySelect(CMP const &cmp, LHS const &l, RHS const &r) {
  return {l, r, cmp};
}

/// Generic Value class, simply wrapping a mlir::Value
/// Get actual mlir::Value through get()
/// Also provides get(loc, builder) to allow mix of expressions and Values
template <typename T> struct EasyVal {
  using ElType = ::mlir::Value;
  ElType _value;

  /// Create Value by wrapping existing mlir::Value
  EasyVal(const ElType &value) : _value(value) {}
  /// Create Value from C++ value
  EasyVal(const ::mlir::Location &loc, ::mlir::OpBuilder &builder, T value)
      : _value(createInt<sizeof(T) * 8>(loc, builder, value)) {
    static_assert(std::is_integral_v<T>);
  }

  /// @return wrapped mlir::Value
  ElType get() const { return _value; };
  /// @return wrapped mlir::Value
  ElType get(const ::mlir::Location &, ::mlir::OpBuilder &) const {
    return _value;
  };
};

/// Special EasyVal representing an mlir::Index
struct EasyIdx : public EasyVal<int64_t> {
  /// Create Value by wrapping existing mlir::Value of index-type
  EasyIdx(const ElType &value) : EasyVal<int64_t>(value) {}
  /// Potentially cast to Index
  EasyIdx(const ::mlir::Location &loc, ::mlir::OpBuilder &builder,
          const ElType &value)
      : EasyVal<int64_t>(createIndexCast(loc, builder, value)) {}
  /// Create Value from C++ value
  EasyIdx(const ::mlir::Location &loc, ::mlir::OpBuilder &builder,
          int64_t value)
      : EasyVal<int64_t>(createIndex(loc, builder, value)) {}
};

} // namespace imex

#endif // _IMEX_ARITHUTILS_H_
