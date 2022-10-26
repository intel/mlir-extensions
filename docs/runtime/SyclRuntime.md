# IMEX GPU RUNTIME

The imex gpu runtime will support two separate runtimes, namely, Sycl and Level Zero. The lowering passes will lower the mlir code all the way to llvm, which will have calls to these runtime wrappers. The eventual calls to sycl or level zero will be wrapped by a wrapper function which will be called from the lowering passes.

```
For example the lowering pass will have IR generated like:

%stream = llvm.call @gpuStreamCreate() : () -> !llvm.ptr<i8>

At runtime, gpuStreamCreate function (which is wrapped) will call the appropriate runtime function.

For example:

void *gpuCreateStream()
{
    FOR SYCL RUNTIME
        // Create a Sycl queue

    FOR L0 RUNTIME
        // Create a L0 queue
    }
}
```

The Wrapper function API's:

gpuCreateStream : This function creates a runtime data structure called Queue. Queue is the high level data structure that encapsulates a sycl queue for sycl runtime and level zero command queue for level zero runtime. The Queue structure for sycl runtime holds sycl::device, sycl::context and sycl::queue. The queue structure for level zero will hold ze::CommandQueue, ze::Device and ze::Context. Programs submit tasks to a device via the queue and monitor the queue for completion.

gpuStreamDestroy  This function, as the name suggests, destroys the above created Queue data structure at the end of the program when the device, context and queue are destroyed.

gpuMemAlloc:  This function allocates memory on the device (GPU) and returns a pointer to that allocated memory.

gpuMemFree: This function frees the memory on the device created by the above mem alloc function. This function is called at the end of the program when the memory is no longer needed.

gpuModuleLoad: This function loads the gpu module. GPU module can contain multiple gpu kernels. This function internally calls zeModuleCreate which compiles the spirv binary to be executed on the device.

gpuKernelGet: This function gets a specific kernel (based on the kernel name) within a gpu module. Kernel here is the computation to be executed on the device.

gpuLaunchKernel: This function launches a specific kernel within a gpu module. It submits a command group function object to the queue for asynchronous execution.

gpuWait: This function waits on the queue till the operations in the queue are completed.
