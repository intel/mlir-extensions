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

The script `build_locally.py` can build both LLVM/MLIR and IMEX for you.

```sh
python scripts/build_locally.py \
    --working-dir $local-working-dir-to-build-llvm \
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
* generate directories `IR` and `Transforms` in the directories (`include/mlir/Dialect`
  and `lib/dialect`)
* Extend/Create cmake infrastructure with defaults
* Create stub source files for IR and transforms
  - `include/imex/Dialect/<name>/IR/<name>Ops.h`
  - `include/imex/Dialect/<name>/IR/<name>Ops.td`
  - `lib/Dialect/IR/<name>Ops.cpp`
  - `include/imex/Dialect/<name>/Transforms/Passes.h`
  - `include/imex/Dialect/<name>/Transforms/Passes.td`
  - `lib/Dialect/Transforms/PassDetail.h`

Add your dialect and its transforms/passes to appropriate places in
- `include/imex/InitIMEXDialects.h`
- `include/imex/InitIMEXPasses.h`
- `lib/Conversion/IMEXPassDetail.h`

Fill in what's marked with FIXME

## Adding a new Conversion
```sh
# enter root directory of mlir-extension
cd mlir-extensions
python scripts/add_conversion.py $name-of-source-dialect $name-of-target-dialect
```
This will
* Let $conversion-name name be "$name-of-source-dialectTo$name-of-target-dialect"
* Add directories `include/mlir/Conversion/<conversion-name>` and `lib/Conversion/<conversion-name>`
* Extend/Create cmake infrastructure with defaults
* Add declarations to header `include/mlir/Conversion/<conversion-name>/<conversion-name>.h`
* Put cpp definition stubs to `lib/Conversion/<conversion-name>/<conversion-name>.cpp`
* Add conversion to `include/imex/Conversion/IMEXPasses.td and include/imex/Conversion/IMEXPasses.h`

Now fill in what's marked with FIXME
* Pattern rewriters
* Populating lists with patterns
* Passes

## License
This code is made available under the Apache License 2.0 with LLVM Exceptions. See the `LICENSE.txt` file for more details.
