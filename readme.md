# Building MLIR backend

MLIR backend is not yet integrated into Numba build process

1. Follow usual numba build instructions (using release llvm)
2. Install pybind11
3. Build llvm from specific commit required for the backend (numba/mlir-compiler/llvm-sha.txt)
4. Build backend using cmake (numba/mlir-compiler/CMakeLists.txt) using compiled llvm
5. Add dir with compiled backend to PYTHONPATH

# Running MLIR backend tests

`python runtests.py numba.mlir.tests`

# Useful env variables

* `NUMBA_MLIR_ENABLE=1` - enable/diasable MLIR backed (default - 1)
* `NUMBA_MLIR_PRINT_IR=1` - dump MLIR IR to stdout before and after each pass (default - 0)
