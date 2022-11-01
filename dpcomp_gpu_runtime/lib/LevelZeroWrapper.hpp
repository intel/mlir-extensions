// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <level_zero/ze_api.h>

namespace ze {
namespace detail {

inline void checkResult(ze_result_t res, const char *func) {
  if (res != ZE_RESULT_SUCCESS)
    throw std::runtime_error(std::string(func) +
                             " failed: " + std::to_string(res));
}

#define CHECK_ZE_RESULT(expr) ze::detail::checkResult((expr), #expr)

template <typename Func, typename InfoFunc>
auto addInfo(Func &&func, InfoFunc &&infoFunc) {
  try {
    return func();
  } catch (const std::runtime_error &e) {
    throw std::runtime_error(std::string(e.what()) + infoFunc());
  }
}

template <typename T> auto wrapZeTypeHelper(T &&arg);

template <typename T, ze_result_t (*deleter)(T)> struct DeleterImpl {
  void operator()(T obj) const { deleter(obj); }
};

template <typename T, typename CreatorT, CreatorT Creator, typename... Args>
inline auto createImpl(Args &&...args) {
  T ret;
  CHECK_ZE_RESULT(Creator(wrapZeTypeHelper(std::forward<Args>(args))..., &ret));
  return ret;
}

template <typename T, ze_result_t (*Deleter)(T)>
struct Type : public std::unique_ptr<std::remove_pointer_t<T>,
                                     DeleterImpl<T, Deleter>> {
  using std::unique_ptr<std::remove_pointer_t<T>,
                        DeleterImpl<T, Deleter>>::unique_ptr;

  explicit operator T() const { return this->get(); }
};

template <typename SrcT, ze_structure_type_t Type> struct DescWrapper {
  DescWrapper(const SrcT &src) : desc(src) { desc.stype = Type; }

  operator SrcT *() { return &desc; }

  SrcT desc;
};

template <ze_structure_type_t Type, typename SrcT>
auto makeDescWrapper(const SrcT &arg) {
  return DescWrapper<SrcT, Type>(arg);
}
} // namespace detail

auto wrapZeType(ze_driver_handle_t src) { return src; }

auto wrapZeType(ze_device_handle_t src) { return src; }

auto wrapZeType(ze_module_handle_t src) { return src; }

auto wrapZeType(int32_t src) { return src; }
auto wrapZeType(uint32_t src) { return src; }
auto wrapZeType(std::nullptr_t src) { return src; }

auto wrapZeType(const ze_device_properties_t &src) {
  return detail::makeDescWrapper<ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES>(src);
}

auto wrapZeType(const ze_context_desc_t &src) {
  return detail::makeDescWrapper<ZE_STRUCTURE_TYPE_CONTEXT_DESC>(src);
}

auto wrapZeType(const ze_command_queue_desc_t &src) {
  return detail::makeDescWrapper<ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC>(src);
}

auto wrapZeType(const ze_module_desc_t &src) {
  return detail::makeDescWrapper<ZE_STRUCTURE_TYPE_MODULE_DESC>(src);
}

auto wrapZeType(const ze_kernel_desc_t &src) {
  return detail::makeDescWrapper<ZE_STRUCTURE_TYPE_KERNEL_DESC>(src);
}

auto wrapZeType(const ze_event_pool_desc_t &src) {
  return detail::makeDescWrapper<ZE_STRUCTURE_TYPE_EVENT_POOL_DESC>(src);
}

auto wrapZeType(const ze_event_desc_t &src) {
  return detail::makeDescWrapper<ZE_STRUCTURE_TYPE_EVENT_DESC>(src);
}

struct Context : public detail::Type<ze_context_handle_t, zeContextDestroy> {
  using detail::Type<ze_context_handle_t, zeContextDestroy>::Type;

  template <typename... Args> static auto create(Args &&...args) {
    return Context(
        detail::createImpl<ze_context_handle_t, decltype(&zeContextCreate),
                           zeContextCreate, Args...>(
            std::forward<Args>(args)...));
  }
};

Context::pointer wrapZeType(const Context &src) { return src.get(); }

struct CommandList
    : public detail::Type<ze_command_list_handle_t, zeCommandListDestroy> {
  using detail::Type<ze_command_list_handle_t, zeCommandListDestroy>::Type;

  template <typename... Args> static auto createImmediate(Args &&...args) {
    return CommandList(
        detail::createImpl<ze_command_list_handle_t,
                           decltype(&zeCommandListCreateImmediate),
                           zeCommandListCreateImmediate, Args...>(
            std::forward<Args>(args)...));
  }
};

CommandList::pointer wrapZeType(const CommandList &src) { return src.get(); }

struct BuildLog : public detail::Type<ze_module_build_log_handle_t,
                                      zeModuleBuildLogDestroy> {
  using detail::Type<ze_module_build_log_handle_t,
                     zeModuleBuildLogDestroy>::Type;
};

struct Module : public detail::Type<ze_module_handle_t, zeModuleDestroy> {
  using detail::Type<ze_module_handle_t, zeModuleDestroy>::Type;

  template <typename... Args> static auto create(Args &&...args) {
    ze_module_handle_t mod;
    ze_module_build_log_handle_t log;
    detail::addInfo(
        [&]() {
          CHECK_ZE_RESULT(zeModuleCreate(
              detail::wrapZeTypeHelper(std::forward<Args>(args))..., &mod,
              &log));
        },
        [&]() {
          BuildLog bl(log);
          size_t size = 0;
          CHECK_ZE_RESULT(zeModuleBuildLogGetString(bl.get(), &size, nullptr));
          std::string log;
          log.resize(size + 1);
          log[0] = '\n';
          // TODO: need C++17 for std::string mutable data
          CHECK_ZE_RESULT(zeModuleBuildLogGetString(
              bl.get(), &size, const_cast<char *>(log.data()) + 1));
          if (log.back() == '\0')
            log.pop_back();
          return log;
        });
    return std::make_pair(Module(mod), BuildLog(log));
  }
};

struct Kernel : public detail::Type<ze_kernel_handle_t, zeKernelDestroy> {
  using detail::Type<ze_kernel_handle_t, zeKernelDestroy>::Type;

  template <typename... Args> static auto create(Args &&...args) {
    return Kernel(
        detail::createImpl<ze_kernel_handle_t, decltype(&zeKernelCreate),
                           zeKernelCreate, Args...>(
            std::forward<Args>(args)...));
  }
};

struct EventPool
    : public detail::Type<ze_event_pool_handle_t, zeEventPoolDestroy> {
  using detail::Type<ze_event_pool_handle_t, zeEventPoolDestroy>::Type;

  template <typename... Args> static auto create(Args &&...args) {
    return EventPool(
        detail::createImpl<ze_event_pool_handle_t, decltype(&zeEventPoolCreate),
                           zeEventPoolCreate, Args...>(
            std::forward<Args>(args)...));
  }
};

EventPool::pointer wrapZeType(const EventPool &src) { return src.get(); }

struct Event : public detail::Type<ze_event_handle_t, zeEventDestroy> {
  using detail::Type<ze_event_handle_t, zeEventDestroy>::Type;

  template <typename... Args> static auto create(Args &&...args) {
    return Event(detail::createImpl<ze_event_handle_t, decltype(&zeEventCreate),
                                    zeEventCreate, Args...>(
        std::forward<Args>(args)...));
  }
};

Event::pointer wrapZeType(const Event &src) { return src.get(); }

namespace detail {
template <typename T> auto wrapZeTypeHelper(T &&arg) {
  return ::ze::wrapZeType(std::forward<T>(arg));
}
} // namespace detail
} // namespace ze
