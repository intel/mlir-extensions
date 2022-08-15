# RFC: Intel GPU Dialect (I-GPU)

Nishant Patel 

## Summary

We propose the Intel Gpu Dialect as a custom Intel dialect to unlock the underlying runtimes (Level zero & Sycl) best suited for Intel Gpus.
This Dialect will be at a lower level than the upstream GPU Dialect. 

## Motivation

Upstream MLIR has a Dialect called the [GPU Dialect](https://mlir.llvm.org/docs/Dialects/GPU/) which provides middle-level abstractions for launching GPU kernels following a programming model similar to that of CUDA or OpenCL. The upstream dialect exposes all the operation required to launch the kernel on a GPU device. But, the drawback with the upstream dialect is that it is designed majorly catering to CUDA. One of the problem with upstream dialect is that it does not expose Stream as an operation in the dialect. For the upstream dialect, stream is created internally while lowering to LLVM runtime calls and it refers specifically to the CUDA stream. Therefore, the need to create an I-GPU Dialect arises. The I-GPU dialect is an extension of the upstream GPU dialect, with a single op called GetStreamOp. The GetStreamOp op is required to prepare the Intel GPU stream for underlying runtime (SYCL/L0). Stream, here, can be thought of as a structure which is similar to sycl queue in Sycl runtime and command queue in L0 runtime. Exposing stream as an operation is necessary to have more control over the underlying runtime and also enables us to take the stream as an input provided by some external user. 

## Proposal

We propose to have the I-GPU dialect with just one op called the GetStreamOp.

### Operation:

#### intel_gpu.get_stream (intel_gpu::GetStreamOp)

Intel GPU Stream Operation

Syntax:

```
operation ::= "intel_gpu.get_stream"() : () -> !intel_gpu.OpaqueType
```

The intel_gpu.get_stream() operation will be added by a pass called IntelGpuStreamOp after the dialect is in Gpu Dialect. After that the IR is lowered all the way down to LLVM.
This operation will be converted to runtime call in the LLVM generated code. The runtime routine will create a SYCL/L0 queue. Queue is used to schedule operations on the device (GPU). From the queue, one can extract the device and context. 


## Alternative

The alternative approch here is to add this operation in the upstream dialect and extend the upstream GPU dialect. But, we feel that having our own dialect will give us more command and control over how we want to design it. Nvidia and AMD both have their own custom dialect at a similar level. 

## Questions

Do we need more operations like context & devices? 




