## Setting up development environment
```sh
conda create -n imex-dev -c intel -c defaults -c conda-forge pip">=21.2.4" pre-commit cmake clang-format tbb-devel
conda activate imex-dev
pre-commit install -f -c ./scripts/.pre-commit-config.yaml
```
Or make sure you have a working c++ compiler, python, clang-format and cmake>3.16 in your path.

## Building
### Linux
**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with
CMake so that it installs `FileCheck` to the chosen installation prefix.

#### Convenience Building LLVM/MLIR + IMEX
```sh
python scripts/build_locally.py \
    --working-dir $local-working-dir-for-tmp-files \
    --llvm-install $llvm-target-dir
```

#### Building Bare Metal With Existing LLVM/MLIR
```sh
mkdir build
cd build
CC=gcc-9 CXX=g++-9 MLIR_DIR=$llvm-c38ef550de81631641cb1485e0641d1d2227dce4 cmake ..
make -j 12
```

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

Also add your dialect to `lib/Conversion/IMEXPassDetail.h`.

Now fill in what's marked with FIXME

## Adding a new Conversion
```sh
# enter root directory of mlir-extension
cd mlir-extensions
python scripts/add_conversion.py $name-of-source-dialect $name-of-target-dialect
```
This will
* Let $conversion-name name be "$name-of-source-dialectTo$name-of-target-dialect"
* Add directories include/mlir/Conversion/<conversion-name> and lib/Conversion/<conversion-name>
* Add declarations to header include/mlir/Conversion/<conversion-name>/<conversion-name>.h
* Put cpp definition stubs to lib/Conversion/<conversion-name>/<conversion-name>.cpp
* Add new conversion-dir to lib/Conversion/CMakeLists.txt
* Add conversion to include/imex/Conversion/IMEXPasses.td and include/imex/Conversion/IMEXPasses.h
* Create a basic lib/Conversion/<conversion-name>/CMakeLists.txt

Now fill in what's marked with FIXME
* Pattern rewriters
* Populating lists with patterns
* Passes

## License
This code is made available under the Apache License 2.0 with LLVM Exceptions. See the `LICENSE.txt` file for more details.
