// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include "Common.hpp"

#ifdef IMEX_USE_DPNP
#include <dpnp_iface.hpp>
#endif

#ifdef IMEX_USE_MKL
#include "mkl.h"
#endif

#define fatal_failure(format, ...)                                             \
  do {                                                                         \
    fprintf(stderr, format, ##__VA_ARGS__);                                    \
    fflush(stderr);                                                            \
    abort();                                                                   \
  } while (0)

namespace {

template <typename T>
void eigImpl(Memref<2, const T> *input, Memref<1, T> *vals,
             Memref<2, T> *vecs) {
#ifdef IMEX_USE_DPNP
  dpnp_eig_c<T, T>(input->data, vals->data, vecs->data, input->dims[0]);
#else
  (void)input;
  (void)vals;
  (void)vecs;
  // direct MKL call or another implementation?
  fprintf(stderr, "Math runtime was compiled without DPNP support\n");
  fflush(stderr);
  abort();
#endif
}

#ifdef IMEX_USE_MKL
template <typename T>
using GemmFunc = void(const CBLAS_LAYOUT, const CBLAS_TRANSPOSE,
                      const CBLAS_TRANSPOSE, const MKL_INT, const MKL_INT,
                      const MKL_INT, const T, const T *, const MKL_INT,
                      const T *, const MKL_INT, const T, T *, const MKL_INT);

template <typename T>
void cpu_gemm(GemmFunc<T> Gemm, const Memref<2, T> *a, const Memref<2, T> *b,
              Memref<2, T> *c, T alpha, T beta) {
  auto is_contiguous = [](const Memref<2, T> *arr, char arr_name) {
    if (arr->strides[0] != 1 && arr->strides[1] != 1) {
      fatal_failure(
          "mkl gemm suports only arrays contiguous on inner dimension.\n"
          "stride for at least one dimension should be equal to 1.\n"
          "'%c' parameter is not contiguous. '%c' strides are %d and %d.",
          arr_name, arr_name, int(arr->strides[0]), int(arr->strides[1]));
    }
  };

  is_contiguous(a, 'a');
  is_contiguous(b, 'b');
  is_contiguous(c, 'c');

  auto is_rowm = [](const Memref<2, T> *arr) { return arr->strides[1] == 1; };

  auto layout = is_rowm(c) ? CblasRowMajor : CblasColMajor;
  auto transA = is_rowm(a) == is_rowm(c) ? CblasNoTrans : CblasTrans;
  auto transB = is_rowm(b) == is_rowm(c) ? CblasNoTrans : CblasTrans;

  auto m = static_cast<MKL_INT>(a->dims[0]);
  auto n = static_cast<MKL_INT>(b->dims[1]);
  auto k = static_cast<MKL_INT>(a->dims[1]);

  auto lda = static_cast<MKL_INT>(is_rowm(a) ? a->strides[0] : a->strides[1]);
  auto ldb = static_cast<MKL_INT>(is_rowm(b) ? b->strides[0] : b->strides[1]);
  auto ldc = static_cast<MKL_INT>(is_rowm(c) ? c->strides[0] : c->strides[1]);

  Gemm(layout,  /*layout*/
       transA,  /*transa*/
       transB,  /*transb*/
       m,       /*m*/
       n,       /*n*/
       k,       /*k*/
       alpha,   /*alpha*/
       a->data, /*a*/
       lda,     /*lda*/
       b->data, /*b*/
       ldb,     /*ldb*/
       beta,    /*beta*/
       c->data, /*c*/
       ldc      /*ldc*/
  );
}
#endif
} // namespace

extern "C" {

#define EIG_VARIANT(T, Suff)                                                   \
  DPCOMP_MATH_RUNTIME_EXPORT void dpcompLinalgEig_##Suff(                      \
      Memref<2, const T> *input, Memref<1, T> *vals, Memref<2, T> *vecs) {     \
    eigImpl(input, vals, vecs);                                                \
  }

EIG_VARIANT(float, float32)
EIG_VARIANT(double, float64)

#undef EIG_VARIANT

#ifdef IMEX_USE_MKL
#define MKL_CALL(f, ...) f(__VA_ARGS__)
#define MKL_GEMM(Prefix) cblas_##Prefix##gemm
#else
static inline void ALL_UNUSED(int dummy, ...) { (void)dummy; }
#define MKL_GEMM(Prefix) 0
#define MKL_CALL(f, ...)                                                       \
  ALL_UNUSED(0, __VA_ARGS__);                                                  \
  fatal_failure("Math runtime was compiled without MKL support\n");
#endif

#define GEMM_VARIANT(T, Prefix, Suff)                                          \
  DPCOMP_MATH_RUNTIME_EXPORT void mkl_gemm_##Suff(                             \
      const Memref<2, T> *a, const Memref<2, T> *b, Memref<2, T> *c) {         \
    MKL_CALL(cpu_gemm<T>, MKL_GEMM(Prefix), a, b, c, 1, 0);                    \
  }

GEMM_VARIANT(float, s, float32)
GEMM_VARIANT(double, d, float64)

#undef GEMM_VARIANT
#undef MKL_CALL
}
