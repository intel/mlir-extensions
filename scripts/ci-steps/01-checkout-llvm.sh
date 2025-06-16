#!/bin/bash

set -ex

cd ${BUILD_HOME}
git clone https://github.com/llvm/llvm-project
export LLVM_SHA=$(cat build_tools/llvm_version.txt)
cd llvm-project
git checkout $LLVM_SHA
git apply ../build_tools/patches/*.patch
