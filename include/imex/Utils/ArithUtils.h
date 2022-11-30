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

/// Generic Value class, simply wrapping a mlir::Value
/// Get actual mlir::Value through get()
template <typename T> struct EasyVal {
  using ElType = ::mlir::Value;
  using CType = T;
  ElType _value;
  const ::mlir::Location *_loc;
  ::mlir::OpBuilder *_builder;

  /// Create Value by wrapping existing mlir::Value
  EasyVal(const ::mlir::Location &loc, ::mlir::OpBuilder &builder,
          const ElType &value)
      : _value(value), _loc(&loc), _builder(&builder) {}
  /// Create Value from C++ value
  template <typename X = T, typename std::enable_if<
                                std::is_integral<X>::value>::type * = nullptr>
  EasyVal(const ::mlir::Location &loc, ::mlir::OpBuilder &builder, X value)
      : _value(createInt<sizeof(T) * 8>(loc, builder, value)), _loc(&loc),
        _builder(&builder) {
    static_assert(std::is_integral_v<T>);
  }

  /// @return wrapped mlir::Value
  ElType get() const { return _value; };

  /// addition of the expressions
  template <typename X = T, typename std::enable_if<
                                std::is_integral<X>::value>::type * = nullptr>
  EasyVal<X> operator+(EasyVal<X> const &r) const {
    return {*_loc, *_builder,
            _builder->create<::mlir::arith::AddIOp>(*_loc, _value, r.get())};
  }

  /// subtraction of the expressions
  template <typename X = T, typename std::enable_if<
                                std::is_integral<X>::value>::type * = nullptr>
  EasyVal<X> operator-(EasyVal<X> const &r) const {
    return {*_loc, *_builder,
            _builder->create<::mlir::arith::SubIOp>(*_loc, _value, r.get())};
  }

  // multiplication of the expressions
  template <typename X = T, typename std::enable_if<
                                std::is_integral<X>::value>::type * = nullptr>
  EasyVal<X> operator*(EasyVal<X> const &r) const {
    return {*_loc, *_builder,
            _builder->create<::mlir::arith::MulIOp>(*_loc, _value, r.get())};
  }

  /// division of the expressions
  template <typename X = T, typename std::enable_if<
                                std::is_integral<X>::value>::type * = nullptr>
  EasyVal<X> operator/(EasyVal<X> const &r) const {
    return {*_loc, *_builder,
            _builder->create<::mlir::arith::DivSIOp>(*_loc, _value, r.get())};
  }

  /// min of the expressions
  template <typename X = T, typename std::enable_if<
                                std::is_integral<X>::value>::type * = nullptr>
  EasyVal<X> min(EasyVal<X> const &r) const {
    return {*_loc, *_builder,
            _builder->create<::mlir::arith::MinSIOp>(*_loc, _value, r.get())};
  }

  /// max of the expressions
  template <typename X = T, typename std::enable_if<
                                std::is_integral<X>::value>::type * = nullptr>
  EasyVal<X> max(EasyVal<X> const &r) const {
    return {*_loc, *_builder,
            _builder->create<::mlir::arith::MaxSIOp>(*_loc, _value, r.get())};
  }

  /// generic comparison of expressions for integral types
  EasyVal<bool> easyCompare(::mlir::arith::CmpIPredicate cmp,
                            EasyVal<T> const &r) const {
    return {
        *_loc, *_builder,
        _builder->create<::mlir::arith::CmpIOp>(*_loc, cmp, _value, r.get())};
  }

  /// integral type equal
  template <typename X = T, typename std::enable_if<
                                std::is_integral<X>::value>::type * = nullptr>
  auto eq(EasyVal<X> const &r) const {
    return easyCompare(::mlir::arith::CmpIPredicate::eq, r);
  }
  /// integral type non-equal
  template <typename X = T, typename std::enable_if<
                                std::is_integral<X>::value>::type * = nullptr>
  auto ne(EasyVal<X> const &r) const {
    return easyCompare(::mlir::arith::CmpIPredicate::ne, r);
  }
  /// integral type signed-lower-than comparison of expressions
  template <typename X = T, typename std::enable_if<
                                std::is_integral<X>::value>::type * = nullptr>
  auto slt(EasyVal<X> const &r) const {
    return easyCompare(::mlir::arith::CmpIPredicate::slt, r);
  }
  /// integral type unsigned-lower-than comparison of expressions
  template <typename X = T, typename std::enable_if<
                                std::is_integral<X>::value>::type * = nullptr>
  auto ult(EasyVal<X> const &r) const {
    return easyCompare(::mlir::arith::CmpIPredicate::ult, r);
  }
  /// integral type signed-lower-equal comparison of expressions
  template <typename X = T, typename std::enable_if<
                                std::is_integral<X>::value>::type * = nullptr>
  auto sle(EasyVal<X> const &r) const {
    return easyCompare(::mlir::arith::CmpIPredicate::sle, r);
  }
  /// integral type unsigned-lower-equal comparison of expressions
  template <typename X = T, typename std::enable_if<
                                std::is_integral<X>::value>::type * = nullptr>
  auto ule(EasyVal<X> const &r) const {
    return easyCompare(::mlir::arith::CmpIPredicate::ule, r);
  }
  /// integral type signed greater-than
  template <typename X = T, typename std::enable_if<
                                std::is_integral<X>::value>::type * = nullptr>
  auto sgt(EasyVal<X> const &r) const {
    return easyCompare(::mlir::arith::CmpIPredicate::sgt, r);
  }
  /// integral type unsigned greater-than
  template <typename X = T, typename std::enable_if<
                                std::is_integral<X>::value>::type * = nullptr>
  auto ugt(EasyVal<X> const &r) const {
    return easyCompare(::mlir::arith::CmpIPredicate::ugt, r);
  }
  /// integral type signed greater-equal
  template <typename X = T, typename std::enable_if<
                                std::is_integral<X>::value>::type * = nullptr>
  auto sge(EasyVal<X> const &r) const {
    return easyCompare(::mlir::arith::CmpIPredicate::sge, r);
  }
  /// integral type unsigned greater-equal
  template <typename X = T, typename std::enable_if<
                                std::is_integral<X>::value>::type * = nullptr>
  auto uge(EasyVal<X> const &r) const {
    return easyCompare(::mlir::arith::CmpIPredicate::uge, r);
  }

  /// select lhs or rhs dependent on comparison
  template <
      typename LHS, typename RHS, typename X = T,
      typename std::enable_if<std::is_same<X, bool>::value>::type * = nullptr>
  EasyVal<typename LHS::CType> select(RHS const &l, LHS const &r) const {
    return {*_loc, *_builder,
            _builder->create<::mlir::arith::SelectOp>(*_loc, _value, l.get(),
                                                      r.get())};
  }
};

/// Special EasyVal representing an mlir::Index
struct EasyIdx : public EasyVal<int64_t> {
  /// Potentially cast to Index
  EasyIdx(const ::mlir::Location &loc, ::mlir::OpBuilder &builder,
          const ElType &value)
      : EasyVal<int64_t>(loc, builder, createIndexCast(loc, builder, value)) {}
  /// Create Value from C++ value
  EasyIdx(const ::mlir::Location &loc, ::mlir::OpBuilder &builder,
          int64_t value)
      : EasyVal<int64_t>(loc, builder, createIndex(loc, builder, value)) {}
  EasyIdx &operator=(const EasyVal<int64_t> &r) {
    _loc = r._loc;
    _builder = r._builder;
    _value = r.get();
    return *this;
  }
  EasyIdx &operator=(const EasyIdx &r) {
    _loc = r._loc;
    _builder = r._builder;
    _value = r.get();
    return *this;
  }
};

} // namespace imex

#endif // _IMEX_ARITHUTILS_H_
