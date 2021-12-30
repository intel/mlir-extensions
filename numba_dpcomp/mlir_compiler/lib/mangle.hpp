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
