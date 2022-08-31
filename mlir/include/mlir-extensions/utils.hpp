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

#include <utility>

#include <llvm/ADT/ScopeExit.h>

namespace llvm {
class Twine;
}

namespace imex {
[[noreturn]] void reportError(const llvm::Twine &msg);

template <typename T, typename H, typename F>
void scopedDiagHandler(T &ctx, H &&diag_handler, F &&func) {
  auto &diagEngine = ctx.getDiagEngine();
  auto diagId = diagEngine.registerHandler(std::forward<H>(diag_handler));
  auto diagGuard =
      llvm::make_scope_exit([&]() { diagEngine.eraseHandler(diagId); });
  func();
}
} // namespace imex
