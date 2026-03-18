# Intel® Extension for MLIR
Intel® Extension for MLIR (IMEX) is a collection of MLIR dialects and passes from Intel for supporting MLIR lowering to Intel silicon (CPU, GPU, …). Goal of this project is to support development of MLIR enhancements for upstream contribution, and to provide a sandbox for validation independent of front end frameworks. Current project scope includes:

* Dialects and passes needed to lower and execute MLIR entry dialect (linalg, CFG, and etc) on Intel GPU.
* Wrapper libraries to inteface with level zero runtime and sycl runtime supporting Intel GPU.
* Other experimental dialects: NDArray, Dist

## Requirements for building and development
### For build
* CMake >= 3.20.0
* Ninja
* doxygen (Optional for building docs)

### Additionals for development
* pre-commit
* clang-format
* lit (If building with option 2 below. https://pypi.org/project/lit/)

### For building GPU runtime (Optional)
#### Installing Intel® software for general purpose GPU capability
```
Instructions here
https://dgpu-docs.intel.com/installation-guides/index.html
```

#### Getting DPC++ compiler (For Sycl runtime)
```
Install DPC++ compiler : Instructions here
https://www.intel.com/content/www/us/en/developer/articles/tool/oneapi-standalone-components.html#dpcpp-cpp

Once DPC++ is installed source the compiler vars:
source /PATH_TO/intel/oneapi/compiler/latest/env/vars.sh
```

#### Getting Level Zero loader (For Level zero runtime and Sycl runtime)
* Build from source for non system-wide(local) install
```sh
git clone https://github.com/oneapi-src/level-zero.git
cd level-zero
cmake -G Ninja -B build -S . \
   -DCMAKE_BUILD_TYPE=Release \
   -DCMAKE_INSTALL_PREFIX=../level-zero-install
cmake --build build --target install
```
* Binary package for system-wide install: https://github.com/oneapi-src/level-zero/releases

### Example: Setting up requirements using Conda
```sh
conda create -n imex-dev -c conda-forge pip">=21.2.4" pre-commit cmake clang-format lit doxygen

conda activate imex-dev
```

### Setting up pre-commit
```sh
pre-commit install -f -c .pre-commit-config.yaml
```

## Building IMEX
IMEX supports three different ways of building depending on how LLVM is set up.
Option 1 is in-tree (Built as part of LLVM) and option 2 and 3 are out-of-tree (Built outside of LLVM)
### Option 1: Build IMEX as an LLVM external project (in-tree)
IMEX can be treated like a sub-project of LLVM and built as part of LLVM by using an LLVM config option called LLVM_EXTERNAL_PROJECTS.
```sh
git clone https://github.com/intel/mlir-extensions.git
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout `cat ../mlir-extensions/build_tools/llvm_version.txt`
git apply ../mlir-extensions/build_tools/patches/*
cmake -G Ninja -B build -S llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="X86;SPIRV" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_EXTERNAL_PROJECTS="Imex" \
   -DLLVM_EXTERNAL_IMEX_SOURCE_DIR=../mlir-extensions

# For GPU support pass these cmake variables for LLVM to enable L0 runtime
  -DMLIR_ENABLE_LEVELZERO_RUNNER=1 \
# Additional if using a non system wide Level Zero Loader built from source
  -DLEVEL_ZERO_DIR=/PATH_TO/level-zero-install \

# build and run imex tests
cmake --build build --target check-imex

```
**Note**: `-DLLVM_INSTALL_UTILS=ON` is not needed for this build since all tests
will run using the `FileCheck` utility that is available in the build tree.
An external `lit` is not needed as well, since all tests will run using `llvm-lit`
in the build tree.

### Option 2: Build IMEX with an installed LLVM (out-of-tree)
**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with
CMake so that it installs `FileCheck` to the chosen installation prefix.
Additonally, `lit` has to be installed separately as it does not install with
the rest of LLVM.

Make sure the installed LLVM is built from the git commit sha as stated in
`build_tools/llvm_version.txt`.
And has all LLVM patches in `build_tools/patches` applied.
For GPU support, make sure LLVM installation has SPIR-V backend and
L0 runtime.
```sh
# For GPU support pass these cmake variables when building LLVM to enable
# SPIR-V backend and L0 runtime
  -DLLVM_TARGETS_TO_BUILD="X86;SPIRV" \
  -DMLIR_ENABLE_LEVELZERO_RUNNER=1 \
```

```sh
cmake -G Ninja -B build -S . \
   -DMLIR_DIR=<PATH_TO_DIRECTORY_WITH_MLIRConfig.cmake> \
   -DLLVM_EXTERNAL_LIT=<PATH_TO_LIT> \
   -DCMAKE_BUILD_TYPE=Release

# build and run imex tests
cmake --build build --target check-imex
```

### Option 3: Build IMEX with LLVM build tree (out-of-tree)
This is similar to option 2. Instead of installed LLVM, LLVM build tree is used.

Make sure before building LLVM, checkout the git commit sha as stated in
`build_tools/llvm_version.txt`.
And apply all LLVM patches in `build_tools/patches`.
```sh
# For GPU support pass these cmake variables when building LLVM to enable
# SPIR-V backend and L0 runtime
  -DLLVM_TARGETS_TO_BUILD="X86;SPIRV" \
  -DMLIR_ENABLE_LEVELZERO_RUNNER=1 \
```

```sh
cmake -G Ninja -B build -S . \
   -DMLIR_DIR=<PATH_TO_DIRECTORY_WITH_MLIRConfig.cmake> \
   -DCMAKE_BUILD_TYPE=Release

# build and run imex tests
cmake --build build --target check-imex
```

### Run IMEX XeGPU SIMT-specific tests

If you want to make sure changes you're going to merge in the upstream LLVM don't break XeGPU testing you can run

```sh
./scripts/run_xegpu_simt_integration_tests.sh path_to_build_dir_of_your_llvm_repo
```

that will launch integration tests from Integration/Dialect/XeGPU/[SG, WG, SIMT] and Integration/Dialect/XeVM.
Make sure you have Intel GPU availaible to get tests running.
You can also run specific tests using option --test <pattern>.

### Building docs
To build user documentation do
```sh
cmake --build build --target mlir-doc
```
It will render docs to the 'doc' directory.

To build code documentation use '-DIMEX_INCLUDE_DOCS' when configuring with cmake and do
```sh
cd build
cmake --build build --target doc_doxygen
```

### Building Python bindings

IMEX can be built with Python bindings. The `imex_mlir` Python package functions similarly to the [MLIR Python bindings](https://mlir.llvm.org/docs/Bindings/Python/), providing access to the MLIR objects as well as IMEX dialects and passes. To enable the bindings:

- LLVM must be built with `-DMLIR_ENABLE_BINDINGS_PYTHON=1`.
- IMEX must be built with `-DIMEX_ENABLE_BINDINGS_PYTHON=1`.
- On systems with multiple Python implementations, you may need to specify the desired Python version, for example, `-DPython3_EXECUTABLE=$(which python3)`.
- pybind11 and nanobind are additional build dependencies.

The `imex_mlir` package is located in the `<IMEX_ROOT>/python_packages/` directory, where `<IMEX_ROOT>` refers to the IMEX build or installation directory. To use the Python bindings, include `<IMEX_ROOT>/python_packages/` in your `PYTHONPATH` environment variable. Usage examples can be found in the [Python test suite](test/python/).


## Build MLIR with Intel GPU support and run integration tests
```sh
git clone https://github.com/llvm/llvm-project.git
cd llvm-project

cmake -G Ninja -B build -S llvm \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_TARGETS_TO_BUILD="X86;SPIRV" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DMLIR_INCLUDE_INTEGRATION_TESTS=ON \
    -DMLIR_ENABLE_LEVELZERO_RUNNER=1

# build MLIR and run all tests including integration tests for Intel GPU.
cmake --build build --target check-mlir

# build MLIR and run only XeGPU dialect integration tests for Intel GPU.
cmake --build build --target check-mlir-integration-dialect-xegpu

# build MLIR and run only XeVM dialect integration tests for Intel GPU.
cmake --build build --target check-mlir-integration-dialect-xevm
```



## Running MLIR GPU code on Intel GPU
MLIR upstream has a high level pass [gpu-lower-to-xevm-pipeline](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/GPU/Pipelines/GPUToXeVMPipeline.cpp) that lowers GPU code for execution on Intel GPU.
The pass support subgroup or workgroup level gpu kernels with XeGPU dialect in addition to regular SIMT style gpu kernels.
```sh
# Pass option specify kernel abstaction level
--gpu-lower-to-xevm-pipeline="xegpu-op-level=lane"  # Lane/SIMT level kernel
--gpu-lower-to-xevm-pipeline="xegpu-op-level=subgroup"  # Subgroup level kernel
--gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup"  # Workgroup level kernel
```
For a simple example of SIMT GPU kernel without XeGPU dialect, check the upstream integration test [no-xegpu-ops.mlir](https://github.com/llvm/llvm-project/blob/main/mlir/test/Integration/Dialect/XeGPU/LANE/no-xegpu-ops.mlir).

To run the example, first make sure to build the required executables and shared libraries for GPU code generation and execution.

```sh
cmake --build build --target mlir-opt mlir-runner mlir_levelzero_runtime mlir_runner_utils
```

The first five lines of `no-xegpu-ops.mlir` list the commands to execute the file on Intel GPU. Here is the break down of the entire sequence. Running GPU code follows a two step process. First, apply `gpu-lower-to-xevm-pipeline` pass with MLIR pass driver `mlir-opt`. This step lowers code to LLVM dialect. Second, jit execute the lowered IR with MLIR jit execution engine for LLVM dialect `mlir-runner`.
The lowered IR contains reference to external functions in external libraries. For example,`libmlir_levelzero_runtime.so` is a shared library for interacting with level zero runtime, and `libmlir_runner_utils.so` is a shared library that provide some utility functions like printing to MLIR code. `mlir-runner` has an option `--shared-libs` that can be used multiple times to specify those external libraries. `mlir-runner` also asks user to provide return type of the main entry function. For example, `--entry-point-result=void` indicates the return type of "main" is void.
Everything put together, entire command will looks like the following.

```sh
mlir-opt <path_to_your_gpu_code.mlir> --gpu-lower-to-xevm-pipeline="xegpu-op-level=lane" \
    | mlir-runner \
       --shared-libs=<path_to_libmlir_levelzero_runtime.so> \
       --shared-libs=<path_to_libmlir_runner_utils.so> \
       --entry-point-result=void
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

Now, it's your turn to
* Add your dialect and its transforms/passes to appropriate places in
  - `include/imex/InitIMEXDialects.h`
  - `include/imex/InitIMEXPasses.h`
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



## Run the lit tests
To run the FileCheck based tests, follow the following steps:

```sh
cmake --build build --target check-imex
```
Add '-v' to the above command-line to get verbose output.

## Benchmarking
IMEX provides an initial set of benchmarks for studying its performance. To build these benchmarks, users need
to manually add `-DIMEX_ENABLE_BENCHMARK=ON` option when building the IMEX. The benchmark testcases and the
script for running them will be generated under the `build/benchmarks` folder.

Currently, IMEX provides benchmarks for the following 4 categories of operations:
| Operation                             | CPU    | GPU    |
| :---:                                 | :---:  | :---:  |
| elementwise (relu and silu)           | Yes    | Yes    |
| reduction (softmax)                   | Yes    | Yes    |
| transpose (transpose)                 | Yes    | Yes    |
| fusion (kInputFusion and kLoopFusion) | No     | Yes    |

These test cases are mainly implemented using linalg dialect, and the spriv test cases for
relu are also provided. Each testcase is named following the pattern of `opname_shape_dtype.mlir`

### How to run ?
For simplicity, the `bench_imex` script is provided to run the benchmark. It can take a mlir file or a folder as input.
for the later case, it will simply run all test cases inside the folder. In addition, it also has to choose a runtime
based on the option. It accepts one of the following three options:
- `-c` for cpu runtime
- `-l` for level-zero runtime (for INTEL GPU)
- `-s` for sycl runtime (for INTEL GPU)


#### Example
```sh
# run a specific test case on CPU
 ./bench_imex -c relu/cpu/relu_1x160x160x120_f16.mlir

# run a set of test cases on GPU using sycl runtime
 ./bench_imex -s relu/gpu/
```
> **NOTE**: if you are using `-c`, please use testcases under `cpu` subfolder; similarly, if you are using `-s` or `-l`,
> please use testcases under `gpu` subfolder. Otherwise, it may have unspecified errors or behaviors.


### How to customize the benchmark ?
IMEX benchmark suite is implemented using CMAKE template, and initially provides limited set of shapes extraced from some production models, e.g., BERT, and AlexNet.
- ReLU: 1x160x160x120, 50x640x20x15, 512x640x20x15
- SiLU: 1x1024x40x30, 50x20x3072, 512x640x20x15
- Softmax: 1x2000, 16x2000, 64x2000, 256x2000, 1024x2000
- Transpose: 128x136, 1024x1024, 16x96x96, 96x7x96
- Reduce: 32x16x512x512

Users can extend it to evaluate more shapes by editing the, e.g, `relu.shapes.in` file, in each subfolder, and then
rebuild the imex. User can also add new data types, but it is currently only limited to basic data types including
fp32, fp16, int32 etc.



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

## Dist/NDArray Misc
- Not using LoadOp. Instead, everything is a SubviewOp. Any size-1 dim must be annotated with static size 1.
  - right now we can only broadcast size-1 dims if their extent is statically known (to be 1)
- Generally, rank reduction of SubviewOp needs overhaul.
  - Right now, no rank reduction is happening, and appropriate affine maps are generated accordingly
  - Without dist-coalesce, repartitioning of 0d arrays coming from a subview do not work correctly. Only the owning process will have the right data.
  - Even if SubviewOp resulted in rank-reduced arrays, we cannot view into our local data since the element might be remote.
  - To follow existing mechanisms (e.g. target parts) we'd basically need to duplicate the entire array.
  - We probably need some special feature to hold duplicates of slices with only one element on the distributed axis.
- NDArray/dist tests can be run (without GPU tests etc) uwing `cmake --build . --target check-ndarray`

## License
This code is made available under the Apache License 2.0 with LLVM Exceptions.
See the `LICENSE.txt` file for more details.
