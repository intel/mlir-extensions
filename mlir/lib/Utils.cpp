// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "imex/Utils.hpp"

#include <stdexcept>

#include "llvm/ADT/Twine.h"

void imex::reportError(const llvm::Twine &msg) {
  throw std::runtime_error(msg.str());
}
