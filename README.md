## Adding a new dialect
```sh
# enter root directory of mlir-extension
cd mlir-extensions
python scripts/add_dialect.py <name-of-new-dialect>
```
This will
* generate directories in the appropriate directories (include/mlir/Dialect and lib/dialect)
* Extend/Create cmake infrastrcuture with defaults
* Create stub soruce files for IR/Ops

## Adding a new Conversion
* Add directories include/mlir/Conversion/<conversion-name> and lib/Conversion/<conversion-name>
* Add declarations to header include/mlir/Conversion/<conversion-name>/<conversion-name>.h
* Put cpp definitions (implementations) to lib/Conversion/<conversion-name>/<conversion-name>.cpp
* Add new conversion-dir to lib/Conversion/CMakeLists.txt
* Copy lib/Conversion/PTensorToLinalg/CMakeLists.txt to lib/Conversion/<conversion-name>/ and adjust as needed

## How to build
**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with
CMake so that it installs `FileCheck` to the chosen installation prefix.
```sh
mkdir build
cd build
CC=gcc-9 CXX=g++-9 MLIR_DIR=<llvm-c38ef550de81631641cb1485e0641d1d2227dce4> cmake ..
make -j 12
```

## License
This code is made available under the Apache License 2.0 with LLVM Exceptions. See the `LICENSE.txt` file for more details.
