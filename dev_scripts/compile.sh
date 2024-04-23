#!/bin/sh

set -e
set -vx

cd $(dirname "$0")/..
mlir_dir=$(pwd -P)

cd ..
if test -d llvm-project
then
    cd llvm-project
    git clean -fd
else
    git clone https://github.com/llvm/llvm-project.git
    cd llvm-project
fi

git reset --hard HEAD
git checkout $(cat $mlir_dir/build_tools/llvm_version.txt)
git apply $mlir_dir/build_tools/patches/*
cmake -G Ninja -B build -S llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="X86" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_EXTERNAL_PROJECTS="Imex" \
   -DLLVM_EXTERNAL_IMEX_SOURCE_DIR=$mlir_dir

cmake --build build --target check-imex

