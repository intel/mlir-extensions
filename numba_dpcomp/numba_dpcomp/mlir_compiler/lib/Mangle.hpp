// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <string>

namespace llvm {
class StringRef;
class raw_ostream;
} // namespace llvm

namespace mlir {
class TypeRange;
}

bool mangle(llvm::raw_ostream &res, llvm::StringRef ident,
            mlir::TypeRange types);

std::string mangle(llvm::StringRef ident, mlir::TypeRange types);
