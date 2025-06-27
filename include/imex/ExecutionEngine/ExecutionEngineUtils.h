//===- ExecutionEngineUtils.h -  Utilities -----------------------===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file includes utility fiunctions used by the runtime wrappers.
///
//===----------------------------------------------------------------------===//

#ifndef IMEX_EXECUTIONENGINE_UTILS_H
#define IMEX_EXECUTIONENGINE_UTILS_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

// Utilities for calculating statistics on a vector of floats
float calculateMin(const std::vector<float> &values);
float calculateMax(const std::vector<float> &values);
float calculateAverage(const std::vector<float> &values);
float calculateMedian(std::vector<float> &values);
float calculateStdDev(const std::vector<float> &values,
                      float mean = -std::numeric_limits<float>::max());
float calculateVariance(const std::vector<float> &values,
                        float mean = -std::numeric_limits<float>::max());
float calculateP95(std::vector<float> &values);
float calculateP5(std::vector<float> &values);
float calculateMiddleThirdAverage(std::vector<float> &values);

#endif // IMEX_EXECUTIONENGINE_UTILS_H
