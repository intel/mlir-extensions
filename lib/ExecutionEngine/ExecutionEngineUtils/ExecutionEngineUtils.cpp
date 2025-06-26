//===- ExecutionEngineUtils.cpp -  Utilities -----------------------===//
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

#include "imex/ExecutionEngine/ExecutionEngineUtils.h"

// Calculate the minimum of a vector of floats
float calculateMin(const std::vector<float> &values) {
  if (values.empty()) {
    return std::numeric_limits<float>::max(); // Return maximum float value if
                                              // the vector is empty
  }

  return *std::min_element(values.begin(), values.end());
}

// Calculate the maximum of a vector of floats
float calculateMax(const std::vector<float> &values) {
  if (values.empty()) {
    return std::numeric_limits<float>::min(); // Return minimum float value if
                                              // the vector is empty
  }

  return *std::max_element(values.begin(), values.end());
}

// Calculate the average of a vector of floats
float calculateAverage(const std::vector<float> &values) {
  if (values.empty()) {
    return 0.0f;
  }
  float sum = 0.0f;
  for (const auto &value : values) {
    sum += value;
  }
  return sum / values.size();
}

// Calculate the median of a vector of floats
float calculateMedian(std::vector<float> &values) {
  if (values.empty()) {
    return 0.0f;
  }
  // std::sort(values.begin(), values.end());
  float median = 0.0f;
  size_t n = values.size();
  size_t medianIndex = n / 2 + 1;
  std::nth_element(values.begin(), values.begin() + medianIndex, values.end());
  // If n is even, return the average of the two middle elements
  // If n is odd, return the middle element
  // Note: This is a more efficient way to calculate median without sorting
  if (n % 2 == 1) {
    median = values[medianIndex];
  } else {
    auto n_2_value = values[medianIndex];
    std::nth_element(values.begin(), values.begin() + medianIndex - 1,
                     values.end());
    auto n_2_minus_1_value = values[medianIndex - 1];
    median = (n_2_value + n_2_minus_1_value) / 2;
  }
  return median;
}

// Calculate standard deviation of a vector of floats
float calculateStdDev(const std::vector<float> &values, float mean) {
  if (values.empty()) {
    return 0.0f;
  }
  // If mean is -std::numeric_limits<float>::max(), calculate it from the values
  // This is useful for cases where the mean is not precomputed
  mean = (mean == -std::numeric_limits<float>::max()) ? calculateAverage(values)
                                                      : mean;
  float sum = 0.0f;
  for (const auto &value : values) {
    sum += (value - mean) * (value - mean);
  }
  return sqrt(sum / values.size());
}

// Calculate variance of a vector of floats
float calculateVariance(const std::vector<float> &values, float mean) {
  if (values.empty()) {
    return 0.0f;
  }
  // If mean is -std::numeric_limits<float>::max(), calculate it from the values
  // This is useful for cases where the mean is not precomputed
  mean = (mean == -std::numeric_limits<float>::max()) ? calculateAverage(values)
                                                      : mean;
  float sum = 0.0f;
  for (const auto &value : values) {
    sum += (value - mean) * (value - mean);
  }
  return sum / values.size();
}

// Calculate P95 of a vector of floats
float calculateP95(std::vector<float> &values) {
  if (values.empty()) {
    return 0.0f;
  }
  size_t p95Index = static_cast<size_t>(0.95 * values.size());
  if (p95Index >= values.size()) {
    p95Index = values.size() - 1;
  }
  std::nth_element(values.begin(), values.begin() + p95Index, values.end());
  return values[p95Index];
}

// Calculate P5 of a vector of floats
float calculateP5(std::vector<float> &values) {
  if (values.empty()) {
    return 0.0f;
  }
  size_t p5Index = static_cast<size_t>(0.05 * values.size());
  if (p5Index >= values.size()) {
    p5Index = values.size() - 1;
  }
  std::nth_element(values.begin(), values.begin() + p5Index, values.end());
  return values[p5Index];
}

// Calculate average of middle 1/3 of a vector of floats, it ignores the first
// and last third of the sorted values
float calculateMiddleThirdAverage(std::vector<float> &values) {
  if (values.empty()) {
    return 0.0f;
  }
  size_t n = values.size();
  if (n < 3) {
    return calculateMedian(values);
  }
  std::sort(values.begin(), values.end());
  size_t start = n / 3;
  size_t end = n - start;
  float sum = 0.0f;
  for (size_t i = start; i < end; ++i) {
    sum += values[i];
  }
  return sum / (end - start);
}
