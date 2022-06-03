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

#include "mangle.hpp"

#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Types.h>

#include <cctype>

#include "mlir-extensions/Dialect/plier/dialect.hpp"

namespace {
static const constexpr auto PREFIX = "_Z";

template <unsigned Width, mlir::IntegerType::SignednessSemantics Sign,
          char Symbol>
bool mangleInt(llvm::raw_ostream &res, mlir::Type type) {
  if (auto i = type.dyn_cast<mlir::IntegerType>()) {
    if (i.getWidth() == Width && i.getSignedness() == Sign) {
      res << Symbol;
      return true;
    }
  }
  return false;
}

template <unsigned Width, char Symbol>
bool mangleFloat(llvm::raw_ostream &res, mlir::Type type) {
  if (auto i = type.dyn_cast<mlir::FloatType>()) {
    if (i.getWidth() == Width) {
      res << Symbol;
      return true;
    }
  }
  return false;
}

static void mangleMemrefImpl(llvm::raw_ostream &res, mlir::ShapedType type);

static bool mangleMemref(llvm::raw_ostream &res, mlir::Type type) {
  if (auto m = type.dyn_cast<mlir::ShapedType>()) {
    mangleMemrefImpl(res, m);
    return true;
  }
  return false;
}

static bool mangleNone(llvm::raw_ostream & /*res*/, mlir::Type type) {
  if (type.isa<mlir::NoneType>())
    return true; // Nothing

  return false;
}

using type_mangler_t = bool (*)(llvm::raw_ostream &, mlir::Type);

static const constexpr type_mangler_t typeManglers[] = {
    &mangleInt<1, mlir::IntegerType::Signed, 'b'>,
    &mangleInt<1, mlir::IntegerType::Unsigned, 'b'>,
    &mangleInt<1, mlir::IntegerType::Signless, 'b'>,

    &mangleInt<8, mlir::IntegerType::Signed, 'a'>,
    &mangleInt<8, mlir::IntegerType::Unsigned, 'h'>,
    &mangleInt<8, mlir::IntegerType::Signless, 'c'>,

    &mangleInt<16, mlir::IntegerType::Signed, 's'>,
    &mangleInt<16, mlir::IntegerType::Unsigned, 't'>,
    &mangleInt<16, mlir::IntegerType::Signless, 's'>,

    &mangleInt<32, mlir::IntegerType::Signed, 'i'>,
    &mangleInt<32, mlir::IntegerType::Unsigned, 'j'>,
    &mangleInt<32, mlir::IntegerType::Signless, 'i'>,

    &mangleInt<64, mlir::IntegerType::Signed, 'x'>,
    &mangleInt<64, mlir::IntegerType::Unsigned, 'm'>,
    &mangleInt<64, mlir::IntegerType::Signless, 'x'>,

    &mangleInt<128, mlir::IntegerType::Signed, 'n'>,
    &mangleInt<128, mlir::IntegerType::Unsigned, 'o'>,
    &mangleInt<128, mlir::IntegerType::Signless, 'n'>,

    &mangleFloat<32, 'f'>,
    &mangleFloat<64, 'd'>,
    &mangleFloat<80, 'e'>,
    &mangleFloat<128, 'g'>,

    &mangleMemref,

    &mangleNone,
};

static bool checkType(mlir::Type type) {
  llvm::raw_null_ostream ss;
  for (auto mangler : typeManglers)
    if (mangler(ss, type))
      return true;

  return false;
}

static bool isValidChar(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (c >= '0' && c <= '9') || (c == '_');
}

static std::string escapeString(llvm::StringRef str) {
  std::string ret;
  llvm::raw_string_ostream ss(ret);
  for (auto c : str) {
    if (isValidChar(c)) {
      ss << c;
    } else {
      ss << "$" << llvm::format_hex_no_prefix(static_cast<unsigned>(c), 2);
    }
  }
  ss.flush();
  return ret;
}

template <typename F>
static void mangleIdentImpl(llvm::raw_ostream &res, llvm::StringRef ident,
                            F &&templateParams) {
  assert(!ident.empty());
  llvm::SmallVector<llvm::StringRef> parts;
  ident.split(parts, '.');
  assert(!parts.empty());
  auto writePart = [&](auto part) {
    auto escaped = escapeString(part);
    if (std::isdigit(escaped.front())) {
      res << escaped.size() + 1 << '_' << escaped;
    } else {
      res << escaped.size() << escaped;
    }
  };
  if (parts.size() == 1) {
    writePart(parts.front());
    templateParams(res);
  } else {
    res << 'N';
    for (auto &part : parts) {
      writePart(part);
    }
    templateParams(res);
    res << 'E';
  }
}

static void mangleIdent(llvm::raw_ostream &res, llvm::StringRef ident) {
  auto dummy = [](auto &) {};
  mangleIdentImpl(res, ident, dummy);
}

template <typename F>
static void mangleIdent(llvm::raw_ostream &res, llvm::StringRef ident,
                        F &&templateParams) {
  auto wrapTemplate = [&](llvm::raw_ostream &s) {
    s << 'I';
    templateParams(s);
    s << 'E';
  };
  mangleIdentImpl(res, ident, wrapTemplate);
}

static void mangleType(llvm::raw_ostream &res, mlir::Type type) {
  for (auto m : typeManglers)
    if (m(res, type))
      return;

  llvm_unreachable("Cannot mangle type");
}

static void mangleMemrefImpl(llvm::raw_ostream &res, mlir::ShapedType type) {
  auto params = [&](llvm::raw_ostream &s) {
    mangleType(s, type.getElementType());
    s << "Li" << type.getRank() << "E";
    mangleIdent(s, "C");
  };
  mangleIdent(res, "array", params);
}

static void mangleTypes(llvm::raw_ostream &res, mlir::TypeRange types) {
  for (auto type : types)
    mangleType(res, type);
}

} // namespace

bool mangle(llvm::raw_ostream &res, llvm::StringRef ident,
            mlir::TypeRange types) {
  for (auto type : types)
    if (!checkType(type))
      return false;

  res << PREFIX;
  mangleIdent(res, ident);
  mangleTypes(res, types);
  return true;
}

std::string mangle(llvm::StringRef ident, mlir::TypeRange types) {
  std::string ret;
  llvm::raw_string_ostream ss(ret);
  if (!mangle(ss, ident, types))
    return {};

  ss.flush();
  return ret;
}
