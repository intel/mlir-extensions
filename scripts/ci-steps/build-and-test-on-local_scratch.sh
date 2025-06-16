#!/bin/bash

set -ex -o pipefail

mkdir -p ${BUILD_HOME}
cp -r * ${BUILD_HOME}

#rm -rf ${HOME}/intel
#cd ${BUILD_HOME}
#curl -sSLO https://registrationcenter-download.intel.com/akdlm/IRC_NAS/ab743006-b006-4c4a-96ae-b536a90efc82/intel-deep-learning-essentials-2025.1.1.25_offline.sh
#sh ./intel-deep-learning-essentials-2025.1.1.25_offline.sh -a --silent --eula accept --install-dir ${BUILD_HOME}/oneapi2025.1 --log-dir ${BUILD_HOME}/oneapi-log
#source ${BUILD_HOME}/oneapi2025.1/setvars.sh

cd ${BUILD_HOME}
curl -sSLO https://github.com/conda-forge/miniforge/releases/download/25.3.0-3/Miniforge3-25.3.0-3-Linux-x86_64.sh
sh ./Miniforge3-25.3.0-3-Linux-x86_64.sh -b -p ${BUILD_HOME}/miniforge -u
source ${BUILD_HOME}/miniforge/bin/activate
conda install -y -c conda-forge pkgconfig level-zero level-zero-devel
source /swtools/intel-gpu/intel_gpu_vars.sh
source /swtools/intel/2025.1/oneapi-vars.sh 
pip install psutil

cd ${BUILD_HOME}
git clone https://github.com/llvm/llvm-project
export LLVM_SHA=$(cat build_tools/llvm_version.txt)
cd llvm-project
git checkout $LLVM_SHA
git apply ../build_tools/patches/*.patch

# Currently, IMEX has tests that rely on both upstream SYCL and IMEX internal runtimes. This is the reason for having both IMEX_ENABLE_SYCL_RUNTIME and MLIR_ENABLE_SYCL_RUNNER
cd ${BUILD_HOME}
./scripts/compile.sh -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DIMEX_ENABLE_BMG_TARGET=1 -DIMEX_ENABLE_SYCL_RUNTIME=1 -DIMEX_ENABLE_L0_RUNTIME=1 -DLLVM_LIT_ARGS="-a -j 4 --debug --timeout=1800" -DMLIR_ENABLE_SYCL_RUNNER=1 -DLEVEL_ZERO_DIR=${CONDA_PREFIX}
cmake --build build --target check-imex | tee ${GITHUB_WORKSPACE}/tests.txt

#cd ${BUILD_HOME}
#rm -rf scripts/xetile-test-gen/GEMM_reports scripts/xetile-test-gen/Generated_GEMM
#cd scripts/xetile-test-gen
#pip install pandas argparse openpyxl
#./run_tests.sh --gen_default_cases=1 --validate=1 --verbose=1 --llvm_build_dir=../../build

#cp build/tests.txt ${GITHUB_WORKSPACE}/
