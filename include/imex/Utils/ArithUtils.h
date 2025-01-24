//===- PassUtils.h - Pass Utility Functions --------------------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
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

  /// Create Index Value from OpFoldResult
  // This is provided as a template because otherwise we get warnings about ISO
  // C++ ambiguity
  // template <typename T, typename std::enable_if<std::is_same<
  //                           T, ::mlir::OpFoldResult>::value>::type * =
  //                           nullptr>
  EasyVal(const mlir::Location &loc, mlir::OpBuilder &builder,
          const ::mlir::OpFoldResult &value)
      : _value(value.is<::mlir::Value>()
                   ? value.get<::mlir::Value>()
                   : builder.create<::mlir::arith::ConstantOp>(
                         *_loc, ::mlir::cast<::mlir::IntegerAttr>(
                                    value.get<::mlir::Attribute>()))),
        _loc(&loc), _builder(&builder) {}

  /// Create Value from C++ value
  template <
      typename X = T,
      typename std::enable_if<std::is_integral<X>::value &&
                              !std::is_same<X, bool>::value>::type * = nullptr>
  EasyVal(const ::mlir::Location &loc, ::mlir::OpBuilder &builder, X value)
      : _value(createInt(loc, builder, value, sizeof(T) * 8)), _loc(&loc),
        _builder(&builder) {
    static_assert(std::is_integral_v<T>);
  }

  /// Create Value from C++ bool is_same
  template <typename X = T, typename std::enable_if<
                                std::is_same<X, bool>::value>::type * = nullptr>
  EasyVal(const ::mlir::Location &loc, ::mlir::OpBuilder &builder, bool value)
      : _value(createInt(loc, builder, value, 1)), _loc(&loc),
        _builder(&builder) {
    static_assert(std::is_same_v<T, bool>);
  }

  /// @return wrapped mlir::Value
  ElType get() const { return _value; };

  /// addition of the expressions
  template <typename X = T, typename std::enable_if<
                                std::is_integral<X>::value>::type * = nullptr>
  EasyVal<X> operator+(EasyVal<X> const &r) const {
    return {
        *_loc, *_builder,
        _builder->createOrFold<::mlir::arith::AddIOp>(*_loc, _value, r.get())};
  }

  /// subtraction of the expressions
  template <typename X = T, typename std::enable_if<
                                std::is_integral<X>::value>::type * = nullptr>
  EasyVal<X> operator-(EasyVal<X> const &r) const {
    return {
        *_loc, *_builder,
        _builder->createOrFold<::mlir::arith::SubIOp>(*_loc, _value, r.get())};
  }

  // multiplication of the expressions
  template <typename X = T, typename std::enable_if<
                                std::is_integral<X>::value>::type * = nullptr>
  EasyVal<X> operator*(EasyVal<X> const &r) const {
    return {
        *_loc, *_builder,
        _builder->createOrFold<::mlir::arith::MulIOp>(*_loc, _value, r.get())};
  }

  /// division of the expressions
  template <typename X = T, typename std::enable_if<
                                std::is_integral<X>::value>::type * = nullptr>
  EasyVal<X> operator/(EasyVal<X> const &r) const {
    return {
        *_loc, *_builder,
        _builder->createOrFold<::mlir::arith::DivSIOp>(*_loc, _value, r.get())};
  }

  /// modulo expressions
  template <typename X = T, typename std::enable_if<
                                std::is_integral<X>::value>::type * = nullptr>
  EasyVal<X> operator%(EasyVal<X> const &r) const {
    return {
        *_loc, *_builder,
        _builder->createOrFold<::mlir::arith::RemSIOp>(*_loc, _value, r.get())};
  }

  /// min of the expressions
  template <typename X = T, typename std::enable_if<
                                std::is_integral<X>::value>::type * = nullptr>
  EasyVal<X> min(EasyVal<X> const &r) const {
    return {
        *_loc, *_builder,
        _builder->createOrFold<::mlir::arith::MinSIOp>(*_loc, _value, r.get())};
  }

  /// max of the expressions
  template <typename X = T, typename std::enable_if<
                                std::is_integral<X>::value>::type * = nullptr>
  EasyVal<X> max(EasyVal<X> const &r) const {
    return {
        *_loc, *_builder,
        _builder->createOrFold<::mlir::arith::MaxSIOp>(*_loc, _value, r.get())};
  }

  /// generic comparison of expressions for integral types
  EasyVal<bool> easyCompare(::mlir::arith::CmpIPredicate cmp,
                            EasyVal<T> const &r) const {
    return {*_loc, *_builder,
            _builder->createOrFold<::mlir::arith::CmpIOp>(*_loc, cmp, _value,
                                                          r.get())};
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

  /// logical XOR
  template <typename X = T, typename std::enable_if<
                                std::is_same<X, bool>::value>::type * = nullptr>
  EasyVal<bool> lxor(EasyVal<X> const &r) const {
    static_assert(std::is_same_v<T, bool>);
    return {
        *_loc, *_builder,
        _builder->createOrFold<::mlir::arith::XOrIOp>(*_loc, _value, r.get())};
    // return EasyVal<bool>(*_loc, *_builder, true) - ((*this) * r);
  }

  /// logical or
  template <typename X = T, typename std::enable_if<
                                std::is_same<X, bool>::value>::type * = nullptr>
  EasyVal<bool> lor(EasyVal<X> const &r) const {
    static_assert(std::is_same_v<T, bool>);
    return {
        *_loc, *_builder,
        _builder->createOrFold<::mlir::arith::OrIOp>(*_loc, _value, r.get())};
  }

  /// logical and
  template <typename X = T, typename std::enable_if<
                                std::is_same<X, bool>::value>::type * = nullptr>
  EasyVal<bool> land(EasyVal<X> const &r) const {
    static_assert(std::is_same_v<T, bool>);
    return {
        *_loc, *_builder,
        _builder->createOrFold<::mlir::arith::AndIOp>(*_loc, _value, r.get())};
  }

  /// select lhs or rhs dependent on comparison
  template <
      typename LHS, typename RHS, typename X = T,
      typename std::enable_if<std::is_same<X, bool>::value>::type * = nullptr>
  auto select(RHS const &l, LHS const &r) const {
    if constexpr(std::is_integral<LHS>::value && std::is_integral<RHS>::value) {
      return EasyVal<LHS>(*_loc, *_builder,
              _builder->createOrFold<::mlir::arith::SelectOp>(*_loc, _value,
                                                              EasyVal<LHS>(*_loc, *_builder, l),
                                                              EasyVal<RHS>(*_loc, *_builder, r)));
    } else if constexpr(!(std::is_integral<LHS>::value && std::is_integral<RHS>::value)) {
      return LHS{*_loc, *_builder,
              _builder->createOrFold<::mlir::arith::SelectOp>(*_loc, _value,
                                                              l.get(), r.get())};
    }
  }
};

/// Special EasyVal representing an mlir::Index
/// Do not use constructors, use easyIdx(...) below.
using EasyIdx = EasyVal<int64_t>;

/// Create Index Value from MLIR value, potentially by casting
inline EasyIdx easyIdx(const ::mlir::Location &loc, ::mlir::OpBuilder &builder,
                       const ::mlir::Value &value) {
  return EasyIdx(loc, builder, createIndexCast(loc, builder, value));
}

/// Create Index Value from C++ value
inline EasyIdx easyIdx(const ::mlir::Location &loc, ::mlir::OpBuilder &builder,
                       int64_t value) {
  return EasyIdx(loc, builder, createIndex(loc, builder, value));
}

/// Create Index Value from OpFoldResult
// This is provided as a template because otherwise we get warnings about ISO
// C++ ambiguity
template <typename T, typename std::enable_if<std::is_same<
                          T, ::mlir::OpFoldResult>::value>::type * = nullptr>
inline EasyIdx easyIdx(const mlir::Location &loc, mlir::OpBuilder &builder,
                       const T &value) {
  return value.template is<::mlir::Value>()
             ? easyIdx(loc, builder, value.template get<::mlir::Value>())
             : easyIdx(loc, builder,
                       ::mlir::getConstantIntValue(value).value());
}

/// Special EasyVal representing an mlir::I64
/// Do not use constructors, use easyI64(...) below.
using EasyI64 = EasyVal<int64_t>;

/// Create I64 Value from MLIR value, potentially by casting
inline EasyI64 easyI64(const ::mlir::Location &loc, ::mlir::OpBuilder &builder,
                       const ::mlir::Value &value) {
  return EasyI64(loc, builder, createCast(loc, builder, value, builder.getI64Type()));
}

/// Create I64 Value from C++ value
inline EasyI64 easyI64(const ::mlir::Location &loc, ::mlir::OpBuilder &builder,
                       int64_t value) {
  return EasyI64(loc, builder, createInt(loc, builder, value, 64));
}

/// Create I64 Value from OpFoldResult
// This is provided as a template because otherwise we get warnings about ISO
// C++ ambiguity
template <typename T, typename std::enable_if<std::is_same<
                          T, ::mlir::OpFoldResult>::value>::type * = nullptr>
inline EasyI64 easyI64(const mlir::Location &loc, mlir::OpBuilder &builder,
                       const T &value) {
  return value.template is<::mlir::Value>()
             ? easyI64(loc, builder, value.template get<::mlir::Value>())
             : easyI64(loc, builder,
                       ::mlir::getConstantIntValue(value).value());
}

} // namespace imex

#endif // _IMEX_ARITHUTILS_H_
