# Intel Extension for MLIR (IMEX)

IMEX is a staging ground to try out new MLIR-based Dialects, tools and compilers. The solutions developed as part of IMEX may eventually be proposed for upstreamed to LLVM, moved out into standalone projects, or keep on getting developed in this repo.

The current state of the code base is highly experimental and mostly an engineering preview. The code is likely to change rapidly.

# Components

There are three main Dialects that are part of IMEX:

* PLIER: An IR for the Numba Python JIT compiler
* GPU_runtime: A runtime dialect for Intel Level-zero devices
* SPIRV: Forked from upstream SPIRV dialect that adds ops to represents compute kernels. (already getting upstreamed)

# Tools

* `numba_dpcomp`: A proof-of-concept Numba-based compiler built on top of PLIER and GPU_runtime dialects.
* `dpcomp-opt`: An `mlir-opt` like tool for IMEX.

# Building IMEX

IMEX is tied to a specific LLVM SHA. The SHA is stored in the `llvm-sha.txt` file. To build IMEX, you need to build LLVM with the MLIR project enabled for the specific SHA mentioned in `llvm-sha.txt. An example of the minimum CMake config needed to build the LLVM project as required by IMEX is shown below:

```bash
cmake <point-to-llvm-project\llvm> \
 -GNinja                           \
 -DLLVM_ENABLE_PROJECTS=mlir       \
 -DLLVM_ENABLE_RTTI=ON             \
 -DLLVM_USE_LINKER=gold            
```
Once you have built LLVM, you are ready to build IMEX. Another minimal CMake configuration is shown below:

```bash
cmake                                             \
-GNinja                                           \
-DCMAKE_PREFIX_PATH= <where-llvm-was-installed>   \
-DIMEX_ENABLE_IGPU_DIALECT=ON                     
```

# Build numba_mlir

TODO

