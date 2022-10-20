// Copyright 2021 Intel Corporation
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

#include <stdio.h>

#include "common.hpp"

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
