# Intel Extension for MLIR

# numba_dpcomp - MLIR-based python compiler

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
Compatible LLVM build can be taken from https://github.com/IntelPython/mlir-llvm-recipe artifacts from Azure build.
It also can be build manually. Instruction below suppose LLVM to be placed in mlir-llvm folder in CWD.

### Windows

LLVM build:
```bash
git clone https://github.com/IntelPython/mlir-llvm-recipe.git
cmake --version
set /p SHA=<mlir-llvm-recipe\\llvm-sha.txt
pip install ninja
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout %SHA%
"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
cmake ..\\llvm-project\\llvm -G "Ninja" -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="host" -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_INSTALL_PREFIX=..\\mlir-llvm
ninja install
```

dpcomp build:
```bash
git clone https://github.com/IntelPython/dpcomp.git
curl --output tbb.zip -L "https://github.com/oneapi-src/oneTBB/releases/download/v2021.1.1/oneapi-tbb-2021.1.1-win.zip"
mkdir tbb
tar -xf "tbb.zip" -C tbb --strip-components=1
<path_to_miniconda>\\Scripts\\activate
conda install numba=0.53 scipy pybind11 tbb=2021.1 pytest -c conda-forge
cmake --version
set LLVM_PATH=%cd%\\mlir-llvm
set TBB_PATH=%cd%\\tbb
cd dpcomp
python setup.py develop
```


### Linux/MacOS

LLVM build:
```bash
git clone https://github.com/IntelPython/mlir-llvm-recipe.git
export SHA=$(cat mlir-llvm-recipe/llvm-sha.txt)
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout $SHA
cmake --version
cmake ../llvm-project/llvm -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="host" -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_RTTI=ON -DCMAKE_INSTALL_PREFIX=../mlir-llvm
make -j$(nproc --all) all
cmake --install .
```

dpcomp build:
```bash
git clone https://github.com/IntelPython/dpcomp.git
wget -O tbb.tgz "https://github.com/oneapi-src/oneTBB/releases/download/v2021.1.1/oneapi-tbb-2021.1.1-lin.tgz"
mkdir tbb
tar -xf "tbb.tgz" -C tbb --strip-components=1
source <path_to_miniconda>/bin/activate
conda install -y numba=0.53 scipy pybind11 pytest -c conda-forge
cmake --version
chmod -R 777 mlir-llvm
export LLVM_PATH=$(pwd)/mlir-llvm
export TBB_PATH=$(pwd)/tbb
cd dpcomp
python setup.py develop
```

## Run python tests

Use Pytest from root of repository to run tests.

```bash
source <path_to_miniconda>/bin/activate
source $(pwd)/../tbb/env/vars.sh
pytest
```

To run tests in parallel and to prevent segfaults from terminating your test runner use you can use `pytest-xdist`
```bash
conda install pytest-xdist
pytest -n4
```

## Run lit/FileCheck tests

```bash
ctest -C Release
```
