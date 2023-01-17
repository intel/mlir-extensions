# imex-runner.py

IMEX provides an pass driver similar to mlir-opt called imex-opt. imex-opt is an extended version of mlir-opt capable of running IMEX passes. Although imex-opt is useful for testing passes it lacks a couple of user friendly feature. First, imex-opt does not have a file format for defining pass pipeline. So pass pipeline setup is passed as a usually long sequence of command line arguments. Second, imex-opt is pass driver without any other features. There are several common usage cases where imex-opt is used as part of command line pipeline like using FileCheck for checking pass outputs or using mlir-cpu-runner for testing end-to-end execution. In such common cases, it would be nice to have tool that glues things together without users to type whole chain of tools connected by pipes.
imex-runner.py was designed as a tool that wraps imex-opt and other common tools to provide a simpler user experience.

## Features
## Pass pipeline file
Pass pipeline is a text file with extension .pp the allows user to save mlir pass pipeline in a text format. It is based on mlir's Textual Pass Pipeline Specification with some extensions.
The format support C++ style single line comment. Add supports listing one pass name per line. The format is converted by imex-runner.py in to mlir' Textual Pass Pipeline Specification by stripping out comments and auto inserting commas. Users can use indentation along with listing one pass per line for an easier to read layout. Whitespaces are stripped automatically.
### Example

```
// Easier for users to understand pass flow and hierarchy
builtin.module(inline
    convert-tensor-to-linalg // Some comment
    convert-elementwise-to-linalg
    // Some other comment
    arith-bufferize
    func.func(empty-tensor-to-alloc-tensor
          eliminate-empty-tensors
          scf-bufferize
          shape-bufferize
          linalg-bufferize
          bufferization-bufferize
          tensor-bufferize)
    func-bufferize)
```

The Pass pipeline file will be read by imex-runner.py and will be expanded into:

```
--pass-pipeline={builtin.module(inline,convert-tensor-to-linalg,convert-elementwise-to-linalg,arith-bufferize,func.func(empty-tensor-to-alloc-tensor,eliminate-empty-tensors,scf-bufferize,shape-bufferize,linalg-bufferize,bufferization-bufferize,tensor-bufferize),func-bufferize)}

```
## Supporting common fixed workflow
imex-runner.py supports several common workflows. The setup is running imex-opt followed by an optional runner(executing some mlir dialect on hardware) followed by an optional FileCheck utility. By opting in and out of the optional parts, user can cover common use cases like running passes with and without checking, end-to-end execution by running passes for lowering and executing though a runner with and without result checking with FileCheck.
## IMEX feature based conditional execution
imex-runner.py support conditional excution based on available features. Current supported features are vulkan-runner, l0-runtime, sycl-runtime.
For example, if you would like imex-runner.py to execute only if vulkan-runner is available, you can do the following.
```
imex-runner.py --requires=vulkan-runner <rest of options>
```
imex-runner.py will simply exit if vulkan-runner is not available.
