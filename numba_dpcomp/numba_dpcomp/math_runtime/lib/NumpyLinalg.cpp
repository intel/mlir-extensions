// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include "Common.hpp"

#ifdef IMEX_USE_DPNP
#include <dpnp_iface.hpp>
#endif

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
}
