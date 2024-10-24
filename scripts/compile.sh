#!/bin/sh

set -e
set -vx

cd $(dirname "$0")/..

cmake -S . -B build -GNinja  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_USE_LINKER=gold -DLLVM_ENABLE_ZSTD=OFF "$@"
