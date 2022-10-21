# RFC: GPUX Dialect (Extension of upstream GPU Dialect)

Core MLIR Team

## Summary

The GPUX dialect allows us to expose stream/device/context creation/destruction ops.These ops are required by the underlying runtimes (Level zero & Sycl) for explicit stream/context/device creation. Hence, we propose the GPUX dialect as an extension of the upstream GPU dialect.
The GPUX dialect will also have some of the upstream GPU dialect ops extended with an added argument for stream in those ops.

## Motivation

Upstream MLIR has a Dialect called the [GPU Dialect](https://mlir.llvm.org/docs/Dialects/GPU/) which provides middle-level abstractions for launching GPU kernels following a programming model similar to that of CUDA or OpenCL. The upstream dialect exposes all the operation required to launch the kernel on a GPU device. But, the drawback with the upstream dialect is it does not expose Stream as an operation in the dialect. For example, if the user wants to launch the kernel on a particular stream(provided by the user), there is no way to do that today. For the upstream dialect, stream is created internally while lowering to LLVM runtime calls and it refers specifically to the CUDA stream. Therefore, the need to create a GPUX Dialect arises. The GPUX dialect is an extension of the upstream GPU dialect with following ops - CreateDeviceOp, DestroyDeviceOp, CreateContextOp, DestroyContextOp, CreateStreamOp, DestroyStreamOp, LaunchFuncOp, AllocOp, DeallocOp, WaitOp, MemcpyOp & MemsetOp.

## Proposal

We propose to have the GPUX dialect with ops listed below:

### Operation:

#### gpux.create_device (gpux::CreateDeviceOp)

Create the GPU Device.

Syntax:

```
operation ::= gpux.create_device $device : (!gpux.OpaqueType) -> !gpux.DeviceType
```

The gpux.create_device op takes a pointer (opaque) to an underlying device and will return a sycl/l0 device handle.


#### gpux.create_context (gpux::CreateContextOp)

Create the Context.

Syntax:

```
operation ::= gpux.create_context $device : (!gpux.DeviceType) -> !gpux.ContextType
```

The gpux.create_context op takes a pointer to an underlying device and will return a sycl/l0 context handle.


#### gpux.create_stream (gpux::CreateStreamOp)

Create the GPU Stream.

Syntax:

```
operation ::= gpux.create_stream $device, $context: (!gpux.DeviceType, !gpux.ContextType) -> !gpux.StreamType
```

The gpux.create_stream operation will create a sycl/l0 queue with the device and context provided.


#### gpux.destroy_device (gpux::DestroyDeviceOp)

Destroy the Device.

Syntax:

```
operation ::= gpux.destroy_device $device : (!gpux.DeviceType) -> ()
```

The gpux.destroy_device operation will deallocate the passed in device pointer.


#### gpux.destroy_context (gpux::DestroyContextOp)

Destroy the Context.

Syntax:

```
operation ::= gpux.destroy_context $context : (!gpux.ContextType) -> ()
```

The gpux.destroy_context operation will deallocate the passed in context pointer.


#### gpux.destroy_stream (gpux::DestroyStreamOp)

Destroy the GPU Stream.

Syntax:

```
operation ::= gpux.destroy_stream $gpux_stream : (!gpux.StreamType) -> ()
```

The gpux.destroy_stream operation will deallocate the passed in device pointer.

#### gpux.alloc (gpux::AllocOp)

GPU memory allocation operation.

Syntax:

```
operation ::= gpux.alloc custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              ($gpux_stream, $dynamicSizes) ([ $symbolOperands^ ])? attr-dict : type($memref)
```

This op is an extension of upstream gpu.alloc op with one added argument for stream, using which the memory is allocated on a specific device.


#### gpux.dealloc (gpux::DeallocOp)

GPU memory deallocation operation.

Syntax:

```
operation ::= gpux.dealloc custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              ($gpux_stream, $memref) attr-dict : type($memref)
```

This op is an extension of upstream gpu.dealloc op with one added argument for stream, using which the memory is deallocated on a specific device.


#### gpux.launch_func (gpux::LaunchFuncOp)

Launches a function as a GPU kernel.

Syntax:

```
operation ::= gpux.launch_func custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $gpux_stream,
              $kernel,
              blocks in  ($gridSizeX, $gridSizeY, $gridSizeZ)
              threads in ($blockSizeX, $blockSizeY, $blockSizeZ)
              (dynamic_shared_memory_size $dynamicSharedMemorySize^)?
              custom<LaunchFuncOperands>($operands, type($operands)) attr-dict
```

This op is an extension of upstream gpu.launch_func op with one added argument for stream on which the kernel is enqueued or launched.

#### gpux.wait (gpux::WaitOp)

Wait for gpu ops in a particular stream to complete.

Syntax:

```
operation ::= gpu.wait custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $gpux_stream attr-dict
```

This op is an extension of upstream gpu.wait op with one added argument for stream on which to wait on.


#### gpux.memcpy (gpux::MemcpyOp)

GPU memcpy operation

Syntax:

```
operation ::= gpu.memcpy custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $gpux_stream, $dst, $src : type($dst), type($src) attr-dict
```

This op is an extension of upstream gpu.memcpy op with one added argument for stream on which this operation will be queued.


#### gpux.memset (gpux::MemsetOp)

GPU memset operation

Syntax:

```
operation ::= gpux.memset custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $gpux_stream, $dst, $value : type($dst), type($value) attr-dict
```

This op is an extension of upstream gpu.memset op with one added argument for stream on which this operation will be queued.

## Upstreaming Plans.

We intend to create an RFC for changing the upstream GPU dialect ops to include stream as an optional argument in the future. Till then, we plan to use this extended dialect.
