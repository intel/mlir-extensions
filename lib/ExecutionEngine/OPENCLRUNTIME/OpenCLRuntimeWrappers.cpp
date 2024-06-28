// Copyright 2022 Intel Corporation
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

#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <atomic>
#include <map>
#include <mutex>

#ifdef _WIN32
#define SYCL_RUNTIME_EXPORT __declspec(dllexport)
#else
#define SYCL_RUNTIME_EXPORT
#endif // _WIN32

namespace {

template <typename F> auto catchAll(F &&func) {
  try {
    return func();
  } catch (const std::exception &e) {
    fprintf(stdout, "An exception was thrown: %s\n", e.what());
    fflush(stdout);
    abort();
  } catch (...) {
    fprintf(stdout, "An unknown exception was thrown\n");
    fflush(stdout);
    abort();
  }
}

#define CHECK2(a)                                                              \
  do {                                                                         \
    (a);                                                                       \
    if (err != CL_SUCCESS) {                                                   \
      fprintf(stderr, "FAIL: err=%d @ line=%d (%s)\n", err, __LINE__, (#a));   \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#define L0_SAFE_CALL(call)                                                     \
  {                                                                            \
    auto status = (call);                                                      \
    if (status != CL_SUCCESS) {                                                \
      fprintf(stdout, "CL error %d\n", status);                                \
      fflush(stdout);                                                          \
      abort();                                                                 \
    }                                                                          \
  }

inline void checkResult(cl_int res, const char *func) {
  if (res != CL_SUCCESS)
    throw std::runtime_error(std::string(func) +
                             " failed: " + std::to_string(res));
}

#define CHECK_ZE_RESULT(expr) checkResult((expr), #expr)

struct CLModule {
  cl_program module = nullptr;
  ~CLModule();
};

constexpr char DeviceMemAllocName[] = "clDeviceMemAllocINTEL";
constexpr char SharedMemAllocName[] = "clSharedMemAllocINTEL";
constexpr char MemBlockingFreeName[] = "clMemBlockingFreeINTEL";
constexpr char SetKernelArgMemPointerName[] = "clSetKernelArgMemPointerINTEL";
constexpr char EnqueueMemcpyName[] = "clEnqueueMemcpyINTEL";

void *queryCLExtFunc(cl_platform_id CurPlatform, const char *FuncName) {
  void *ret = clGetExtensionFunctionAddressForPlatform(CurPlatform, FuncName);

  if (!ret) {
    fprintf(stdout, "Failed to get CL extension function %s\n", FuncName);
    fflush(stdout);
    abort();
  }
  return ret;
}

void *queryCLExtFunc(cl_device_id dev, const char *FuncName) {
  cl_platform_id CurPlatform;
  L0_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_PLATFORM, sizeof(cl_platform_id),
                               &CurPlatform, nullptr));
  return queryCLExtFunc(CurPlatform, FuncName);
}

struct CLExtTable {
  clDeviceMemAllocINTEL_fn allocDev;
  clSharedMemAllocINTEL_fn allocShared;
  clMemBlockingFreeINTEL_fn blockingFree;
  clSetKernelArgMemPointerINTEL_fn setKernelArgMemPtr;
  clEnqueueMemcpyINTEL_fn enqueneMemcpy;
  CLExtTable() = default;
  CLExtTable(cl_device_id dev) {
    cl_platform_id plat;
    L0_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_PLATFORM,
                                 sizeof(cl_platform_id), &plat, nullptr));
    allocDev =
        (clDeviceMemAllocINTEL_fn)queryCLExtFunc(plat, DeviceMemAllocName);
    allocShared =
        (clSharedMemAllocINTEL_fn)queryCLExtFunc(plat, SharedMemAllocName);
    blockingFree =
        (clMemBlockingFreeINTEL_fn)queryCLExtFunc(plat, MemBlockingFreeName);
    setKernelArgMemPtr = (clSetKernelArgMemPointerINTEL_fn)queryCLExtFunc(
        plat, SetKernelArgMemPointerName);
    enqueneMemcpy =
        (clEnqueueMemcpyINTEL_fn)queryCLExtFunc(plat, EnqueueMemcpyName);
  }
};

// an "almost" lock-free cache for cl_device_id mapping to CL extention function
// table reading from the table is lock-free. And writing to it (when
// cache-miss) requires locking
struct CLExtTableCache {
  static constexpr int numExtCache = 16;
  std::array<std::atomic<cl_device_id>, numExtCache> devices;
  std::array<CLExtTable, numExtCache> tables;
  std::mutex lock;
  static CLExtTableCache &get() {
    static CLExtTableCache v;
    return v;
  }
  CLExtTable *query(cl_device_id dev) {
    bool found = false;
    int firstSearch = search(dev, 0, found);
    if (found) {
      return &tables[firstSearch];
    }
    if (firstSearch == numExtCache) {
      return nullptr;
    }
    {
      std::lock_guard<std::mutex> guard{lock};
      int secondSearch = search(dev, firstSearch, found);
      if (found) {
        return &tables[secondSearch];
      }
      if (secondSearch == numExtCache) {
        return nullptr;
      }
      tables[secondSearch] = CLExtTable(dev);
      devices[secondSearch].store(dev, std::memory_order_acquire);
      return &tables[secondSearch];
    }
  }

private:
  int search(cl_device_id dev, int startIdx, bool &found) {
    for (int i = startIdx; i < numExtCache; i++) {
      auto val = devices[i].load(std::memory_order_release);
      if (!val) {
        found = false;
        return i;
      }
      if (val == dev) {
        found = true;
        return i;
      }
    }
    found = false;
    return numExtCache;
  }
};

} // namespace

namespace {
// Create a Map for the spirv module lookup
std::map<void *, CLModule> moduleCache;
std::mutex mutexLock;
} // namespace

CLModule::~CLModule() { L0_SAFE_CALL(clReleaseProgram(module)); }

struct ParamDesc {
  void *data;
  size_t size;

  bool operator==(const ParamDesc &rhs) const {
    return data == rhs.data && size == rhs.size;
  }

  bool operator!=(const ParamDesc &rhs) const { return !(*this == rhs); }
};

template <typename T> size_t countUntil(T *ptr, T &&elem) {
  assert(ptr);
  auto curr = ptr;
  while (*curr != elem) {
    ++curr;
  }
  return static_cast<size_t>(curr - ptr);
}

static cl_device_id getDevice(cl_device_type *devtype) {
  cl_platform_id platform; // OpenCL platform
  cl_device_id device;     // device ID
  L0_SAFE_CALL(clGetPlatformIDs(1, &platform, NULL));
  L0_SAFE_CALL(clGetDeviceIDs(platform, *devtype, 1, &device, NULL));
  return device;
}

struct GPUSYCLQUEUE {

  cl_device_id device_ = nullptr;
  cl_context context_ = nullptr;
  cl_command_queue queue_ = nullptr;
  bool context_owned_ = false;
  bool queue_owned_ = false;
  CLExtTable *ext_table_ = nullptr;

  GPUSYCLQUEUE(cl_device_type *device, cl_context context,
               cl_command_queue queue) {
    cl_device_type defaultdev = CL_DEVICE_TYPE_GPU;
    if (!device) {
      device = &defaultdev;
    }
    device_ = getDevice(device);
    init_context(context, queue, device_);
    ext_table_ = CLExtTableCache::get().query(device_);
  }
  GPUSYCLQUEUE(cl_device_id device, cl_context context,
               cl_command_queue queue) {
    if (!device) {
      cl_device_type defaultdev = CL_DEVICE_TYPE_GPU;
      device = getDevice(&defaultdev);
    }
    device_ = device;
    init_context(context, queue, device_);
    ext_table_ = CLExtTableCache::get().query(device_);
  }
  ~GPUSYCLQUEUE() {
    if (queue_ && queue_owned_)
      clReleaseCommandQueue(queue_);
    if (context_ && context_owned_)
      clReleaseContext(context_);
  }

private:
  void init_context(cl_context context, cl_command_queue queue,
                    cl_device_id device) {
    if (queue) {
      if (!context) {
        throw std::runtime_error(
            "Cannot create QUEUE wrapper with queue and without context");
      }
      queue_ = queue;
      queue_owned_ = true;
      context_ = context;
      context_owned_ = true;
      return;
    }
    cl_int err;
    if (!context) {
      CHECK2(context_ = clCreateContext(NULL, 1, &device, NULL, NULL, &err));
      context_owned_ = true;
    } else {
      context_ = context;
    }
    CHECK2(queue_ =
               clCreateCommandQueueWithProperties(context, device, 0, &err));
    queue_owned_ = true;
  }
}; // end of GPUSYCLQUEUE

static void *allocDeviceMemory(GPUSYCLQUEUE *queue, size_t size,
                               size_t alignment, bool isShared) {
  void *memPtr = nullptr;
  cl_int err;
  if (isShared) {
    auto func = queue->ext_table_ ? queue->ext_table_->allocShared
                                  : (clSharedMemAllocINTEL_fn)queryCLExtFunc(
                                        queue->device_, SharedMemAllocName);
    CHECK2(memPtr = func(queue->context_, queue->device_, nullptr, size,
                         alignment, &err));
  } else {
    auto func = queue->ext_table_ ? queue->ext_table_->allocDev
                                  : (clDeviceMemAllocINTEL_fn)queryCLExtFunc(
                                        queue->device_, DeviceMemAllocName);
    CHECK2(memPtr = func(queue->context_, queue->device_, nullptr, size,
                         alignment, &err));
  }
  return memPtr;
}

static void deallocDeviceMemory(GPUSYCLQUEUE *queue, void *ptr) {
  auto func = queue->ext_table_ ? queue->ext_table_->blockingFree
                                : (clMemBlockingFreeINTEL_fn)queryCLExtFunc(
                                      queue->device_, MemBlockingFreeName);
  L0_SAFE_CALL(func(queue->context_, ptr));
}

static cl_program loadModule(GPUSYCLQUEUE *queue, const unsigned char *data,
                             size_t dataSize) {
  assert(data);
  cl_int errNum = 0;
  const unsigned char *codes[1] = {data};
  size_t sizes[1] = {dataSize};
  cl_program program;
  cl_int err;
  CHECK2(program = clCreateProgramWithBinary(
             queue->context_, 1, &queue->device_, sizes, codes, &err, &errNum));
  const char *build_flags = nullptr;
  // enable large register file if needed
  if (getenv("IMEX_ENABLE_LARGE_REG_FILE")) {
    build_flags =
        "-vc-codegen -doubleGRF -Xfinalizer -noLocalSplit -Xfinalizer "
        "-DPASTokenReduction -Xfinalizer -SWSBDepReduction -Xfinalizer "
        "'-printregusage -enableBCR' ";
  }
  L0_SAFE_CALL(clBuildProgram(program, 0, NULL, build_flags, NULL, NULL));
  return program;
}

static cl_kernel getKernel(GPUSYCLQUEUE *queue, cl_program program,
                           const char *name) {
  cl_kernel kernel;
  cl_int err;
  CHECK2(kernel = clCreateKernel(program, name, &err));
  cl_bool TrueVal = CL_TRUE;
  L0_SAFE_CALL(clSetKernelExecInfo(
      kernel, CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL, sizeof(cl_bool),
      &TrueVal));
  L0_SAFE_CALL(clSetKernelExecInfo(
      kernel, CL_KERNEL_EXEC_INFO_INDIRECT_SHARED_ACCESS_INTEL, sizeof(cl_bool),
      &TrueVal));
  return kernel;
}

static void launchKernel(GPUSYCLQUEUE *queue, cl_kernel kernel, size_t gridX,
                         size_t gridY, size_t gridZ, size_t blockX,
                         size_t blockY, size_t blockZ, size_t sharedMemBytes,
                         ParamDesc *params) {
  // auto func = queue->ext_table_
  //                 ? queue->ext_table_->setKernelArgMemPtr
  //                 : (clSetKernelArgMemPointerINTEL_fn)queryCLExtFunc(
  //                       queue->device_, SetKernelArgMemPointerName);
  auto paramsCount = countUntil(params, ParamDesc{nullptr, 0});
  // The assumption is, if there is a param for the shared local memory,
  // then that will always be the last argument.
  if (sharedMemBytes) {
    paramsCount = paramsCount - 1;
  }
  for (size_t i = 0; i < paramsCount; i++) {
    auto param = params[i];
    L0_SAFE_CALL(clSetKernelArg(kernel, i, param.size, param.data));
  }
  if (sharedMemBytes) {
    L0_SAFE_CALL(clSetKernelArg(kernel, paramsCount, sharedMemBytes, nullptr));
  }
  size_t globalSize[3] = {gridX, gridY, gridZ};
  size_t localSize[3] = {blockX, blockY, blockZ};
  L0_SAFE_CALL(clEnqueueNDRangeKernel(queue->queue_, kernel, 3, NULL,
                                      globalSize, localSize, 0, NULL, NULL));
}

// Wrappers

extern "C" SYCL_RUNTIME_EXPORT GPUSYCLQUEUE *gpuCreateStream(void *device,
                                                             void *context) {
  return catchAll([&]() {
    return new GPUSYCLQUEUE(reinterpret_cast<cl_device_id>(device),
                            reinterpret_cast<cl_context>(context), nullptr);
  });
}

extern "C" SYCL_RUNTIME_EXPORT void gpuStreamDestroy(GPUSYCLQUEUE *queue) {
  catchAll([&]() { delete queue; });
}

extern "C" SYCL_RUNTIME_EXPORT void *
gpuMemAlloc(GPUSYCLQUEUE *queue, size_t size, size_t alignment, bool isShared) {
  return catchAll([&]() {
    if (queue) {
      return allocDeviceMemory(queue, size, alignment, isShared);
    }
  });
}

extern "C" SYCL_RUNTIME_EXPORT void gpuMemFree(GPUSYCLQUEUE *queue, void *ptr) {
  catchAll([&]() {
    if (queue && ptr) {
      deallocDeviceMemory(queue, ptr);
    }
  });
}

extern "C" SYCL_RUNTIME_EXPORT cl_program
gpuModuleLoad(GPUSYCLQUEUE *queue, const unsigned char *data, size_t dataSize) {
  return catchAll([&]() {
    if (queue) {
      return loadModule(queue, data, dataSize);
    }
  });
}

extern "C" SYCL_RUNTIME_EXPORT cl_kernel
gpuKernelGet(GPUSYCLQUEUE *queue, cl_program module, const char *name) {
  return catchAll([&]() {
    if (queue) {
      return getKernel(queue, module, name);
    }
  });
}

extern "C" SYCL_RUNTIME_EXPORT void
gpuLaunchKernel(GPUSYCLQUEUE *queue,  cl_kernel kernel, size_t gridX,
                size_t gridY, size_t gridZ, size_t blockX, size_t blockY,
                size_t blockZ, size_t sharedMemBytes, void *params) {
  return catchAll([&]() {
    if (queue) {
      launchKernel(queue, kernel, gridX, gridY, gridZ, blockX, blockY, blockZ,
                   sharedMemBytes, static_cast<ParamDesc *>(params));
    }
  });
}

extern "C" SYCL_RUNTIME_EXPORT void gpuWait(GPUSYCLQUEUE *queue) {
  catchAll([&]() {
    if (queue) {
      L0_SAFE_CALL(clFinish(queue->queue_));
    }
  });
}
