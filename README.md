# dpcomp

This is the MLIR based compiler backend

## Building

For build LLVM and MLIR paths should be specified, for example

cmake . -DLLVM_DIR="../llvm-project/lib/cmake/llvm" -DMLIR_DIR="../llvm-project/lib/cmake/mlir"

cmake --build .
