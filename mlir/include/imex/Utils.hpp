// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
