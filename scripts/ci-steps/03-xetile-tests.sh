#!/bin/bash

set -ex

cd ${BUILD_HOME}
rm -rf scripts/xetile-test-gen/GEMM_reports scripts/xetile-test-gen/Generated_GEMM
cd scripts/xetile-test-gen
pip install pandas argparse openpyxl
./run_tests.sh --gen_default_cases=1 --validate=1 --verbose=1 --llvm_build_dir=../../build
