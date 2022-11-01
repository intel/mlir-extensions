// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "FilterStringParser.hpp"

#include <array>
#include <charconv>

static std::optional<std::array<std::string_view, 3>>
tokenize(std::string_view filterString) {
  std::array<std::string_view, 3> tokens;
  auto pos = filterString.find(":");
  if (pos == std::string_view::npos)
    return {};

  tokens[0] = filterString.substr(0, pos);
  filterString = filterString.substr(pos + 1);

  pos = filterString.find(":");
  if (pos == std::string_view::npos)
    return {};

  tokens[1] = filterString.substr(0, pos);
  filterString = filterString.substr(pos + 1);

  pos = filterString.find(":");
  if (pos != std::string_view::npos)
    return {};

  tokens[2] = filterString;
  return tokens;
}

std::optional<DeviceDesc> parseFilterString(std::string_view filterString) {
  auto tk = tokenize(filterString);
  if (!tk)
    return {};

  auto &tokens = *tk;

  DeviceDesc ret;
  auto idx = tokens[2];
  auto begin = idx.data();
  auto end = begin + idx.size();
  auto [ptr, ec] = std::from_chars(begin, end, ret.index);
  if (ec != std::errc() || ptr != end)
    return {};

  ret.backend = tokens[0];
  ret.name = tokens[1];
  return ret;
}
