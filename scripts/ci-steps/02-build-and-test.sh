#!/bin/bash

set -ex

cd ${BUILD_HOME}
source /opt/intel/oneapi/setvars.sh
pip install psutil
# Currently, IMEX has tests that rely on both upstream SYCL and IMEX internal runtimes. This is the reason for having both IMEX_ENABLE_SYCL_RUNTIME and MLIR_ENABLE_SYCL_RUNNER
./scripts/compile.sh -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DIMEX_ENABLE_PVC_TARGET=1 -DIMEX_ENABLE_SYCL_RUNTIME=1 -DIMEX_ENABLE_L0_RUNTIME=1 -DLLVM_LIT_ARGS="-a -j 4 --debug --timeout=1800" -DMLIR_ENABLE_SYCL_RUNNER=1
cmake --build build --target check-imex | tee build/tests.txt
