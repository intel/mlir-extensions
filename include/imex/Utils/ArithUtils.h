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
#include <mlir/Dialect/Arith/IR/Arith.h>

namespace imex {

class EasyInt {
  const ::mlir::Location *_loc;
  ::mlir::OpBuilder *_builder;
  ::mlir::Value _value;

public:
  EasyInt(const ::mlir::Location &loc, ::mlir::OpBuilder &builder,
          const ::mlir::Value &value, bool idx = false)
      : _loc(&loc), _builder(&builder) {
    if (idx) {
      _value = createIndexCast(loc, builder, value);
    } else {
      _value = value;
    }
  }

  template <int W = 64>
  static EasyInt create(const ::mlir::Location &loc, ::mlir::OpBuilder &builder,
                        uint64_t value, bool idx = false) {
    if (idx) {
      return {loc, builder, createIndex(loc, builder, value)};
    } else {
      // if constexpr(::std::is_integral_v<T>) {
      return {loc, builder, createInt<W>(loc, builder, value)};
    }
    assert(!"Only ints supported as EasyValues");
    return {loc, builder, ::mlir::Value()};
  }

  ::mlir::Value get() const { return _value; }

  EasyInt operator+(const EasyInt &rhs) const {
    return {*_loc, *_builder,
            _builder->create<::mlir::arith::AddIOp>(*_loc, _value, rhs.get())};
  }
  EasyInt operator-(const EasyInt &rhs) const {
    return {*_loc, *_builder,
            _builder->create<::mlir::arith::SubIOp>(*_loc, _value, rhs.get())};
  }
  EasyInt operator*(const EasyInt &rhs) const {
    return {*_loc, *_builder,
            _builder->create<::mlir::arith::MulIOp>(*_loc, _value, rhs.get())};
  }
  EasyInt operator/(const EasyInt &rhs) const {
    return {*_loc, *_builder,
            _builder->create<::mlir::arith::DivSIOp>(*_loc, _value, rhs.get())};
  }
  EasyInt eq(const EasyInt &rhs) const {
    return {*_loc, *_builder,
            _builder->create<::mlir::arith::CmpIOp>(
                *_loc, ::mlir::arith::CmpIPredicate::eq, _value, rhs.get())};
  }
  EasyInt ne(const EasyInt &rhs) const {
    return {*_loc, *_builder,
            _builder->create<::mlir::arith::CmpIOp>(
                *_loc, ::mlir::arith::CmpIPredicate::ne, _value, rhs.get())};
  }
  EasyInt slt(const EasyInt &rhs) const {
    return {*_loc, *_builder,
            _builder->create<::mlir::arith::CmpIOp>(
                *_loc, ::mlir::arith::CmpIPredicate::slt, _value, rhs.get())};
  }
  EasyInt ult(const EasyInt &rhs) const {
    return {*_loc, *_builder,
            _builder->create<::mlir::arith::CmpIOp>(
                *_loc, ::mlir::arith::CmpIPredicate::ult, _value, rhs.get())};
  }
  EasyInt sgt(const EasyInt &rhs) const {
    return {*_loc, *_builder,
            _builder->create<::mlir::arith::CmpIOp>(
                *_loc, ::mlir::arith::CmpIPredicate::sgt, _value, rhs.get())};
  }
  EasyInt ugt(const EasyInt &rhs) const {
    return {*_loc, *_builder,
            _builder->create<::mlir::arith::CmpIOp>(
                *_loc, ::mlir::arith::CmpIPredicate::ugt, _value, rhs.get())};
  }
  EasyInt sge(const EasyInt &rhs) const {
    return {*_loc, *_builder,
            _builder->create<::mlir::arith::CmpIOp>(
                *_loc, ::mlir::arith::CmpIPredicate::sge, _value, rhs.get())};
  }
  EasyInt uge(const EasyInt &rhs) const {
    return {*_loc, *_builder,
            _builder->create<::mlir::arith::CmpIOp>(
                *_loc, ::mlir::arith::CmpIPredicate::uge, _value, rhs.get())};
  }
  EasyInt select(const EasyInt &lhs, const EasyInt &rhs) {
    return {*_loc, *_builder,
            _builder->create<::mlir::arith::SelectOp>(*_loc, _value, lhs.get(),
                                                      rhs.get())};
  }
};

} // namespace imex

#endif // _IMEX_ARITHUTILS_H_
