// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include "Common.hpp"

#ifdef IMEX_USE_DPNP
#include <dpnp_iface.hpp>
#endif

// #include "CL/sycl.hpp"
// #include "oneapi/mkl.hpp"
#include "mkl.h"

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

// template<typename T>
// void sycl_gemm(cl::sycl::queue* queue, Memref<2, const T> *a, Memref<2, const T> *b, Memref<2, T> *c, float alpha, float beta)
// {
//   auto transA = a->strides[1] == sizeof(T) ? onemkl::transpose::N : onemkl::transpose::T;
//   auto transB = a->strides[1] == sizeof(T) ? onemkl::transpose::N : onemkl::transpose::T;

//   auto lda = transA == onemkl::transpose::N ? a->strides[0]/sizeof(T): a->strides[1]/sizeof(T);
//   auto ldb = transB == onemkl::transpose::N ? b->strides[0]/sizeof(T): b->strides[1]/sizeof(T);
//   auto ldc = b->strides[0]/sizeof(T);

//   oneapi::mkl::blas::row_major::gemm(
//     queue, /*queue*/
//     transA, /*transa*/
//     transB, /*transb*/
//     a->dims[0], /*m*/
//     b->dims[1], /*n*/
//     a->dims[1], /*k*/
//     alpha, /*alpha*/
//     a->data, /*a*/
//     lda, /*lda*/
//     b->data, /*b*/
//     ldb, /*ldb*/
//     beta, /*beta*/
//     c->data, /*c*/
//     ldc, /*ldc*/
//     {} /*dependencies*/
//     ).wait();
// }


template<typename T> using GemmFunc = void(
  const CBLAS_LAYOUT,
  const CBLAS_TRANSPOSE,
  const CBLAS_TRANSPOSE,
  const MKL_INT,
  const MKL_INT,
  const MKL_INT,
  const T,
  const T*,
  const MKL_INT,
  const T*,
  const MKL_INT,
  const T,
  T*,
  const MKL_INT
);

template<typename T>
// void cpu_gemm(GemmFunc<T> Gemm, Memref<2, const T> *a, Memref<2, const T> *b, Memref<2, T> *c, float alpha, float beta)
void cpu_gemm(Memref<2, const T> a, Memref<2, const T> b, Memref<2, T> c, float alpha, float beta)
{
  // auto transA = a->strides[1] == sizeof(T) ? CblasNoTrans : CblasTrans;
  // auto transB = a->strides[1] == sizeof(T) ? CblasNoTrans : CblasTrans;

  // auto lda = transA == CblasNoTrans ? a->strides[1]/sizeof(T): a->strides[0]/sizeof(T);
  // auto ldb = transB == CblasNoTrans ? b->strides[1]/sizeof(T): b->strides[0]/sizeof(T);
  // auto ldc = b->strides[1]/sizeof(T);

  printf("%d %d %d %d\n", int(a.dims[0]), int(a.dims[1]), int(a.strides[0]), int(a.strides[1]));
  printf("%d %d %d %d\n", int(b.dims[0]), int(b.dims[1]), int(b.strides[0]), int(b.strides[1]));
  printf("%d %d %d %d\n", int(c.dims[0]), int(c.dims[1]), int(c.strides[0]), int(c.strides[1]));
  // printf("%d %d %d %d\n", b->dims[0], b->dims[1], b->strides[0], b->strides[1]);
  // printf("%d %d %d %d\n", c->dims[0], c->dims[1], c->strides[0], c->strides[1]);

  // for (int i = 0; i < a->dims[0]; ++i)
  // {
  //   for (int j = 0; j < b->dims[1]; ++j)
  //   {
  //     T val = 0;
  //     for (int k = 0; k < a->dims[1]; ++k)
  //     {
  //       val += a->data[i + ]
  //     }
  //   }
  // }
  printf("Hello\n");
  // Gemm(
  //   CblasRowMajor, /*layout*/
  //   transA, /*transa*/
  //   transB, /*transb*/
  //   a->dims[0], /*m*/
  //   b->dims[1], /*n*/
  //   a->dims[1], /*k*/
  //   alpha, /*alpha*/
  //   a->data, /*a*/
  //   a->dims[1], /*lda*/
  //   b->data, /*b*/
  //   b->dims[1], /*ldb*/
  //   beta, /*beta*/
  //   c->data, /*c*/
  //   c->dims[1] /*ldc*/
  //   );
}
// template<typename T>
// void cpu_gemm(Memref<2, const T> *a, Memref<2, const T> *b, Memref<2, T> *c, float alpha, float beta)
// {
//   auto transA = a->strides[1] == sizeof(T) ? CblasNoTrans : CblasTrans;
//   auto transB = a->strides[1] == sizeof(T) ? CblasNoTrans : CblasTrans;

//   auto lda = transA == CblasNoTrans ? a->strides[0]/sizeof(T): a->strides[1]/sizeof(T);
//   auto ldb = transB == CblasNoTrans ? b->strides[0]/sizeof(T): b->strides[1]/sizeof(T);
//   auto ldc = b->strides[0]/sizeof(T);

//   // Gemm(
//   //   CblasRowMajor, /*layout*/
//   //   transA, /*transa*/
//   //   transB, /*transb*/
//   //   a->dims[0], /*m*/
//   //   b->dims[1], /*n*/
//   //   a->dims[1], /*k*/
//   //   alpha, /*alpha*/
//   //   a->data, /*a*/
//   //   lda, /*lda*/
//   //   b->data, /*b*/
//   //   ldb, /*ldb*/
//   //   beta, /*beta*/
//   //   c->data, /*c*/
//   //   ldc /*ldc*/
//   //   );
// }
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

// #define GEMM_VARIANT(T, Prefix, Suff)                                         \
//   DPCOMP_MATH_RUNTIME_EXPORT void mkl_gemm_##Suff(                            \
//       Memref<2, const T> *a,                                                  \
//       Memref<2, const T> *b,                                                  \
//       Memref<2, T> *c) {                                                      \
//     cpu_gemm<T>(cblas_##Prefix##gemm, a, b, c, 1, 0);                         \
//   }

#define GEMM_VARIANT(T, Prefix, Suff)                                         \
  DPCOMP_MATH_RUNTIME_EXPORT void mkl_gemm_##Suff(                            \
      Memref<2, const T> a,                                                  \
      Memref<2, const T> b,                                                  \
      Memref<2, T> c) {                                                      \
    cpu_gemm<T>(a, b, c, 1, 0);                                               \
  }


// #define GEMM_VARIANT(T, Prefix, Suff)                                         \
//   DPCOMP_MATH_RUNTIME_EXPORT void mkl_gemm_##Suff(                            \
//       Memref<2, const T> *a,                                                  \
//       Memref<2, const T> *b) {                                                \
//     cpu_gemm<T>(cblas_##Prefix##gemm, a, 0, 0, 1, 0);                         \
//   }

// #define GEMM_VARIANT(T, Prefix, Suff)                                         \
//   DPCOMP_MATH_RUNTIME_EXPORT void mkl_gemm_##Suff(                            \
//       Memref<2, const T> *a,                                                  \
//       Memref<2, const T> *b,                                                  \
//       float alpha,                                                            \
//       float beta,                                                             \
//       Memref<2, T> *c) {                                                      \
//     cpu_gemm<T>(a, b, c, alpha, beta);                                        \
//   }

GEMM_VARIANT(float, s, float32)
GEMM_VARIANT(double, d, float64)
}
