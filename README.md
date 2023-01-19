# Intel® Extension for MLIR
Intel® Extension for MLIR (IMEX) is a collection of MLIR dialects and passes from Intel for improving upstream MLIR.
Current extension covers.
* Dialects and passes needed to lower and execute MLIR entry dialect (linalg, CFG, and etc) on Intel GPU.
* Wrapper libraries to inteface with level zero runtime and sycl runtime supporting Intel GPU.
* Other experimental dialects: PTensor, Dist

## Requirements for building and development
### For build
* CMake > 3.14.3 (3.18 if using Conda)
* Ninja
* doxygen (Optional for building docs)
### Additionals for development
* pre-commit
* clang-format
* lit (If set up as an out-of-tree build of LLVM)

### Example: Setting up requirements using Conda
```sh
conda create -n imex-dev -c intel -c defaults -c conda-forge pip">=21.2.4" pre-commit cmake clang-format lit doxygen

conda activate imex-dev
```

### Setting up pre-commit
```
pre-commit install -f -c .pre-commit-config.yaml
```

## Building IMEX
IMEX supports three different ways of building depending on how LLVM is set up.
### Option 1: Build IMEX together with LLVM source as an LLVM external project
IMEX can be treated like a sub-project of LLVM and built as part of LLVM by using an LLVM config option called LLVM_EXTERNAL_PROJECTS.
```
git clone https://github.com/intel/mlir-extensions.git
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout `cat ../mlir-extensions/build_tools/llvm_version.txt`
cmake -G Ninja -B build -S llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="X86" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_EXTERNAL_PROJECTS="Imex" \
   -DLLVM_EXTERNAL_IMEX_SOURCE_DIR=../mlir-extensions

cmake --build build --target check-imex
```
**Note**: `-DLLVM_INSTALL_UTILS=ON` is not needed for this build since all tests
will run using the `FileCheck` utility that is available in the build tree.
An external `lit` is not needed as well, since all tests will run using `llvm-lit`
in the build tree.

### Option 2: Build IMEX with a separately built and installed LLVM
**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with
CMake so that it installs `FileCheck` to the chosen installation prefix.

#### Using bundled build script

The script `build_tools/build_imex.py` can build both LLVM/MLIR and IMEX for you. Use
`build_imex.py -h` to look at all the options provided by the script. It is
advisable to use an external lit when building IMEX.

If you want the script to build LLVM and then IMEX, do as follows:

```sh
external_lit=`which lit`
python build_tools/build_imex.py                        \
    --working-dir $local-working-dir-to-build-llvm  \
    --external-lit ${external_lit}
```

To reuse a previously built LLVM, use the following steps:

Make sure your LLVM install is built from the git commit sha as stated in
'build_tools/llvm_version.txt'.

```sh
external_lit=`which lit`
python build_tools/build_imex.py                        \
    --working-dir $local-working-dir-to-build-llvm  \
    --llvm-install $llvm-target-dir                 \
    --external-lit ${external_lit}
```

#### Building Bare Metal With Existing LLVM/MLIR
Make sure your LLVM install is built from the git commit sha as stated in
'build_tools/llvm_version.txt'.
```sh
mkdir build
cd build
CC=gcc-9 CXX=g++-9 MLIR_DIR=<llvm-install-directory> cmake ..
make -j 12


For GPU support, pass the cmake variables to enable the required runtime libraries

CC=gcc-9 CXX=g++-9 MLIR_DIR=<llvm-install-directory> cmake .. -DSYCL_DIR=/PATH_TO/intel/oneapi/compiler/latest/linux/ -DLEVEL_ZERO_DIR=/PATH_TO/level-zero-install/ -DIMEX_ENABLE_L0_RUNTIME=1 -DIMEX_ENABLE_SYCL_RUNTIME=1
make -j 12
```

### Option 3: Build IMEX with LLVM build tree
This is similar to Option 2. Instead of building and installing LLVM. Just build LLVM and set "MLIR_DIR" to the sub-directory in the LLVM build tree that has generated file MLIRConfig.cmake. Rest of the step is the same as Option 2.

### Building docs
To build user documentation do
```sh
cd build
cmake --build . --target mlir-doc
```
It will render docs to the 'doc' directory.

To build code documentation use '-DIMEX_INCLUDE_DOCS' when configuring with cmake and do
```sh
cd build
cmake --build . --target doc_doxygen
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

Now, it's your turn to
* Add your dialect and its transforms/passes to appropriate places in
  - `include/imex/InitIMEXDialects.h`
  - `include/imex/InitIMEXPasses.h`
  - `lib/Conversion/IMEXPassDetail.h`
* Fill in what's marked with FIXME
* The documentation of the dialect should go into the `description` fields in `<name>Ops.td`. At build time the description
will be extracted and a file `doc/<name>.md` will be generated automatically. It will include descriptions of the dialect and operations in a standardized way.

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
* Add a pass def stub to `include/imex/Conversion/IMEXPasses.td and include/imex/Conversion/Passes.td`

You will now have to
* Fill in the above files what's marked with FIXME
* The documentation of the pass should go into the `description` field in `Passes.td`. At build time the description
will be extracted and a file `doc/Conversions.md` will be generated automatically.
* Write your Pattern rewriters


## Getting Level Zero loader (Optional, needed for GPU support with Level zero runtime)
```Bash
git clone https://github.com/oneapi-src/level-zero.git
cd level-zero
mkdir build
cd build
cmake ../level-zero -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../level-zero-install
ninja install
```

## Getting DPC++ compiler (Optional, needed for GPU support with Sycl runtime)
```
Install DPC++ compiler : Instructions here
https://www.intel.com/content/www/us/en/developer/articles/tool/oneapi-standalone-components.html#dpcpp-cpp

Once DPC++ is installed source the compiler vars:
source /PATH_TO/intel/oneapi/compiler/latest/env/vars.sh
```

## Run the lit tests
To run the FileCheck based tests, follow the following steps:

```sh
cd <to your IMEX build directory>
cmake --build . --target check-imex
```
Add '-v' to the above command-line to get verbose output.

## Profiling kernel execute time
### sycl event
```sh
export IMEX_ENABLE_PROFILING=ON
run the test
```
### trace tools
```sh
python {your_path}/imex_runner.py xxx -o test.mlir
mlir-translate test.mlir -mlir-to-llvmir -o test.ll
llc test.ll -filetype=obj -o test.o
clang++ test.o {path}/libmlir_runner_utils.so {path}/libmlir_c_runner_utils.so {path}/libsycl-runtime.so -no-pie -o test
ze_tracer ./test
```

## License
This code is made available under the Apache License 2.0 with LLVM Exceptions.
See the `LICENSE.txt` file for more details.
