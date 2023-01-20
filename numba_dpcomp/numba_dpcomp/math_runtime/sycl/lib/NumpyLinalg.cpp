// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <mutex>
#include <string_view>
#include <unordered_map>

#include "Common.hpp"
#include "dpcomp-math-sycl-runtime_export.h"

#ifdef IMEX_USE_SYCL_MKL
#include "CL/sycl.hpp"
#include "oneapi/mkl.hpp"
#endif

/// Stream interface, must be in sync with gpu runtime.
/// TODO: move to common place.
class StreamInterface {
public:
  virtual ~StreamInterface() = default;
  virtual std::string_view getDeviceName() = 0;
};

// using namespace cl;

namespace {

#ifdef IMEX_USE_SYCL_MKL

struct QueueMap {
  std::unordered_map<std::string, cl::sycl::queue> map;
  std::mutex m;
  cl::sycl::queue getQueue(std::string device) {
    std::lock_guard<std::mutex> guard(m);
    auto device_queue_iter = map.find(device);
    if (device_queue_iter == map.end()) {
      try {
        sycl::device d{sycl::ext::oneapi::filter_selector(device.c_str())};
        device_queue_iter = map.insert({device, sycl::queue(d)}).first;
      } catch (const sycl::exception &e) {
        std::vector<cl::sycl::device> devices = cl::sycl::device::get_devices();
        std::string known_devices = "";
        for (auto &&d : devices)
          known_devices += d.get_info<cl::sycl::info::device::name>() + ",";
        fatal_failure("Failed to find device mathcing name '%s'.\n"
                      "Error message is '%s'\n."
                      "Known devices ara: '%s'",
                      device.c_str(), e.what(), known_devices.c_str());
      }
    }

    return device_queue_iter->second;
  }
};

static std::unique_ptr<QueueMap> qMapPtr;

template <typename T>
using GemmFunc = sycl::event (*)(cl::sycl::queue &, oneapi::mkl::transpose,
                                 oneapi::mkl::transpose, std::int64_t,
                                 std::int64_t, std::int64_t, T, const T *,
                                 std::int64_t, const T *, std::int64_t, T, T *,
                                 std::int64_t,
                                 const std::vector<cl::sycl::event> &);

template <typename T>
static void deviceGemm(void *stream, const Memref<2, T> *a,
                       const Memref<2, T> *b, Memref<2, T> *c, T alpha,
                       T beta) {
  auto streamIface = static_cast<StreamInterface *>(stream);

  auto isContiguous = [](const Memref<2, T> *arr, char arr_name) {
    if (arr->strides[0] != 1 && arr->strides[1] != 1) {
      fatal_failure(
          "mkl gemm suports only arrays contiguous on inner dimension.\n"
          "stride for at least one dimension should be equal to 1.\n"
          "'%c' parameter is not contiguous. '%c' strides are %d and %d.\n",
          arr_name, arr_name, int(arr->strides[0]), int(arr->strides[1]));
    }
  };

  isContiguous(a, 'a');
  isContiguous(b, 'b');
  isContiguous(c, 'c');

  constexpr auto colmGemm =
      static_cast<GemmFunc<T>>(oneapi::mkl::blas::column_major::gemm);
  constexpr auto rowmGemm =
      static_cast<GemmFunc<T>>(oneapi::mkl::blas::row_major::gemm);

  auto isRowm = [](const Memref<2, T> *arr) { return arr->strides[1] == 1; };

  auto Gemm = isRowm(c) ? rowmGemm : colmGemm;
  auto transA = isRowm(a) == isRowm(c) ? oneapi::mkl::transpose::N
                                       : oneapi::mkl::transpose::T;
  auto transB = isRowm(b) == isRowm(c) ? oneapi::mkl::transpose::N
                                       : oneapi::mkl::transpose::T;

  auto m = static_cast<std::int64_t>(a->dims[0]);
  auto n = static_cast<std::int64_t>(b->dims[1]);
  auto k = static_cast<std::int64_t>(a->dims[1]);

  auto lda =
      static_cast<std::int64_t>(isRowm(a) ? a->strides[0] : a->strides[1]);
  auto ldb =
      static_cast<std::int64_t>(isRowm(b) ? b->strides[0] : b->strides[1]);
  auto ldc =
      static_cast<std::int64_t>(isRowm(c) ? c->strides[0] : c->strides[1]);

  auto aData = getMemrefData(a);
  auto bData = getMemrefData(b);
  auto cData = getMemrefData(c);

  auto queue = qMapPtr->getQueue(std::string(streamIface->getDeviceName()));

  Gemm(queue,  /*queue*/
       transA, /*transa*/
       transB, /*transb*/
       m,      /*m*/
       n,      /*n*/
       k,      /*k*/
       alpha,  /*alpha*/
       aData,  /*a*/
       lda,    /*lda*/
       bData,  /*b*/
       ldb,    /*ldb*/
       beta,   /*beta*/
       cData,  /*c*/
       ldc,    /*ldc*/
       {}      /*dependencies*/
       )
      .wait();
}
#endif

void initMap() {
#ifdef IMEX_USE_SYCL_MKL
  qMapPtr.reset(new QueueMap());
#endif
}

void finilizeMap() {
#ifdef IMEX_USE_SYCL_MKL
  qMapPtr.reset();
#endif
}

} // namespace

extern "C" {

#ifdef IMEX_USE_SYCL_MKL
#define GEMM_VARIANT(T, Suff)                                                  \
  DPCOMP_MATH_SYCL_RUNTIME_EXPORT void mkl_gemm_##Suff##_device(               \
      void *stream, const Memref<2, T> *a, const Memref<2, T> *b,              \
      Memref<2, T> *c) {                                                       \
    deviceGemm<T>(stream, a, b, c, 1, 0);                                      \
  }

GEMM_VARIANT(float, float32)
GEMM_VARIANT(double, float64)
#undef GEMM_VARIANT
#endif

// Not thread safe
DPCOMP_MATH_SYCL_RUNTIME_EXPORT void dpcompMathRuntimeInit() { initMap(); }

DPCOMP_MATH_SYCL_RUNTIME_EXPORT void dpcompMathRuntimeFinalize() {
  finilizeMap();
}
}
