# dpcomp - the MLIR based compiler backend

# numba_dpcomp - Python package with custom jit decorators that are provide a compilation with dpcomp

## Usage

Simple scenario of using is to replace jit decorators from Numba with the same from numba_dpcomp, for example replace "from numba import njit" with "from numba_dpcomp import njit" in Python script.

## Build environment

```bash
conda create -n dpcompenv python=3.7 numba=0.53 scipy pybind11 tbb tbb-devel cmake pytest
```

LLVM is required.

For now only one of latest version of LLVM is supported, the sha of it commit can be found in llvm-sha.txt in ther root of repository.

## Build

For build LLVM and MLIR paths should be specified.

### Windows

```bash
cmake -A x64 . -DLLVM_DIR="..\llvm\lib\cmake\llvm" -DMLIR_DIR="..\llvm\lib\cmake\mlir"
cmake --build . --config Release
```

### Linux/MacOS

```bash
cmake . -DLLVM_DIR="../llvm/lib/cmake/llvm" -DMLIR_DIR="../llvm/lib/cmake/mlir"
cmake --build .
```

## Run tests

For now it is necessary to add directory with mlir_compiler module to PYTHONPATH environment variable:

`...\dpcomp\numba_dpcomp\mlir_compiler\Release` - for Windows

`.../dpcomp/numba_dpcomp/mlir_compiler` - for Linux/MacOS

Use Pytest from root of repository to run tests.

```bash
pytest
```
