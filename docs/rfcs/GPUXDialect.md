# RFC: GPUX Dialect (Extension of upstream GPU Dialect)

Core MLIR Team

## Summary

We propose the GPUX dialect as an extension of the upstream GPU dialect. The GPUX dialect allows us to expose stream creation/destruction ops.These ops are required by the underlying runtimes (Level zero & Sycl) for explicit stream/context/device creation.
The GPUX dialect will also have some of the upstream GPU dialect ops extended with an added argument for stream in those ops.

## Motivation

Upstream MLIR has a Dialect called the [GPU Dialect](https://mlir.llvm.org/docs/Dialects/GPU/) which provides middle-level abstractions for launching GPU kernels following a programming model similar to that of CUDA or OpenCL. The upstream dialect exposes all the operation required to launch the kernel on a GPU device. But, the drawback with the upstream dialect is it does not expose Stream as an operation in the dialect. For example, if the user wants to launch the kernel on a particular stream(provided by the user), there is no way to do that today. For the upstream dialect, stream is created internally while lowering to LLVM runtime calls and it refers specifically to the CUDA stream. Therefore, the need to create a GPUX Dialect arises. The GPUX dialect is an extension of the upstream GPU dialect with following ops - CreateStreamOp, DestroyStreamOp, LaunchFuncOp, AllocOp, DeallocOp & WaitOp . The CreateStreamOp op allows for explicit Stream creation. The stream is a temporary stream and is internally creating a SYCL/L0 queue, with default context and device.

## Proposal

We propose to have the GPUX dialect with ops listed below:

### Operation:

#### gpux.create_stream (gpux::CreateStreamOp)

Create the GPU Stream.

Syntax:

```
operation ::= gpux.create_stream : () -> !gpux.StreamType
```

The gpux.create_stream() operation will be added by a custom pass only if user does not want to use their own stream.
This operation will be converted to runtime call for a temporary stream creation in the LLVM generated code. The runtime routine will create a SYCL/L0 queue, device and context based on the runtime.


#### gpux.destroy_stream (gpux::DestroyStreamOp)

Destroy the GPU Stream.

Syntax:

```
operation ::= gpux.destroy_stream $gpux_stream : (!gpux.StreamType) -> ()
```

The gpux.destroy_stream() operation will destroy the above created stream. If the user wants to use their stream, we expect the user to manage the lifetime of the stream.


#### gpux.alloc (gpux::AllocOp)

GPU memory allocation operation.

Syntax:

```
operation ::= gpux.alloc custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              ($gpux_stream, $dynamicSizes) ([ $symbolOperands^ ])? attr-dict : type($memref)
```

This op is an extension of upstream gpu.alloc op with one added argument for stream.


#### gpux.dealloc (gpux::DeallocOp)

GPU memory deallocation operation.

Syntax:

```
operation ::= gpux.dealloc custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              ($gpux_stream, $memref) attr-dict : type($memref)
```

This op is an extension of upstream gpu.dealloc op with one added argument for stream.


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

This op is an extension of upstream gpu.launch_func op with one added argument for stream.

#### gpux.wait (gpux::WaitOp)

Wait for gpu ops in a particular stream to complete.

Syntax:

```
operation ::= gpu.wait custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $gpux_stream attr-dict
```

The wait op will need to be provided with the stream to wait on. This op is an extension of upstream gpu.wait op with one added argument for stream.


## Example Usage in case where users do not provide the stream:

// IR before our custom transformation pass:

```

func.func @main() attributes {llvm.emit_c_interface} {
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 2.200000e+00 : f32
  %cst_0 = arith.constant 1.100000e+00 : f32
  %cst_1 = arith.constant 0.000000e+00 : f32
  %0 = gpu.alloc  () : memref<8xf32>
  %1 = gpu.alloc  () : memref<8xf32>
  %2 = gpu.alloc  () : memref<8xf32>
  gpu.launch_func  @main_kernel::@main_kernel blocks in (%c8, %c1, %c1) threads in (%c1, %c1, %c1) args(%0 : memref<8xf32>, %1 : memref<8xf32>, %2 : memref<8xf32>)
  %3 = memref.cast %2 : memref<8xf32> to memref<*xf32>
  call @printMemrefF32(%3) : (memref<*xf32>) -> ()
  gpu.dealloc(%0) : memref<8xf32>
  gpu.dealloc(%1) : memref<8xf32>
  gpu.dealloc(%2) : memref<8xf32>
  return
}

```

// IR after our custom transformation pass:

```

func.func @main() attributes {llvm.emit_c_interface} {
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 2.200000e+00 : f32
  %cst_0 = arith.constant 1.100000e+00 : f32
  %cst_1 = arith.constant 0.000000e+00 : f32
  %stream = gpux.create_stream () : () -> !gpux.StreamType
  %0 = gpux.alloc (%stream) : (!gpux.StreamType) -> memref<8xf32>
  %1 = gpux.alloc (%stream) : (!gpux.StreamType) -> memref<8xf32>
  %2 = gpux.alloc  (%stream) : (!gpux.StreamType) -> memref<8xf32>
  gpux.launch_func  @main_kernel::@main_kernel blocks in (%c8, %c1, %c1) threads in (%c1, %c1, %c1) args(%stream : !gpux.StreamType, %memref : memref<8xf32>, %memref_2 : memref<8xf32>, %memref_3 : memref<8xf32>)
  %3 = memref.cast %memref_3 : memref<8xf32> to memref<*xf32>
  call @printMemrefF32(%3) : (memref<*xf32>) -> ()
  gpux.dealloc(%stream, %0) : (!gpux.StreamType, memref<8xf32>) -> ()
  gpux.dealloc(%stream, %1) : (!gpux.StreamType, memref<8xf32>) -> ()
  gpux.dealloc(%stream, %2) : (!gpux.StreamType, memref<8xf32>) -> ()
  gpux.stream.destroy(%stream) : (!gpux.StreamType) -> ()
  return
}
```

In case the users dont want to use their own stream, our custom pass will create a temporary stream for the function, use that stream to launch kernel, and destroy it at the end of the function.

##  Usage case where users want to use their own stream:
If users want to use their own stream, we expect the users to write their own custom pass and change the IR to use the stream with our GPUX dialect.
Users can write a pass to change the function signature to take an additional argument for stream and use the GPUX dialect ops to launch the kernel with their own provided stream. We dont intend on providing that pass as the use cases can be varied and depend on users need. There can be two use cases: one is the upper level compiler decides compute region and assigns stream at the compile time, like TF. The other is the upper level compiler decide compute region, and leaves it to runtime to assign stream. Hence, we keep this open to allow the upper level compiler to write their own paths to decide stream usage, as GPUX is a low level dialect.


## Example Usage in case of a highler level dialect having a region with stream

```
Ptensor Region ( Ptensor_stream ) {
               t3 = Ptensor.add  t1, t2
}

After lowering it at the GPU dialect it would look like:

Device_func ptensor_add_x (t3, t1, t2) {
               t3 = linalg.add  t1, t2
}
Ptensor Region ( Ptensor_stream ) {
               GPU.launch_func @ptensor_add_x(), (t3, t1, t2)
}

At this point, we expect there is a pass for Ptensor compiler to lower Ptensor region.
With this pass, you can lower the GPU to GPUX, which allows upper-level compiler to specify
stream and hook that stream in gpux dialect ops that take stream as input.

Region_Func_1 (Ptensor_stream, t3, t1, t2)

      GPUX.launch_func GPUX_stream, @ptensor_add_x(), (t3, t1, t2)
}

Now the Ptensor compiler could call createConvertGPUXToLLVMPass(Runtime runtime), which will be a part of IMEX GPUX dialect implementation.
We expect createConvertGPUXToLLVMPass(Runtime runtime) to accept a parameter (enum as defined below) which indicate the compilation target l0 runtime or sycl runtime. Now at runtime, the compiled Region_Func_1 would expect a Ptensor_stream of type StreamType.

enum Runtime {
  sycl,
  level_zero
};

```

## StreamType (Custom Type):
We will have our custom StreamType defined which will hold a opaque pointer to a Struct. We expect the user to provide us a struct which holds queue, device and context. The Stream struct should adhere to one of these formats:

For sycl:
```
struct Stream {
  sycl::queue queue;
  sycl::device device;
  sycl::context context;
}
```

For Level Zero:
```
struct Stream {
  ze::CommandList queue;
  ze_device_handle_t device;
  ze::Context context;
}
```

At runtime the Opaque pointer (of StreamType) will be casted to appropriate runtime objects using the underlying data members.

## Features to be added in subsequent PR's (either in the dialect or via a pass operating on this dialect):
1. More ops can be added/extended to dialect going forward based on the use cases and requirements.
2. Currently, the stream creation and destruction happens for every function. A pass to manage stream will be added where the stream is created only once at the beginning of the program and used throughout.

## Upstreaming Plans.

We intend to create an RFC for changing the upstream GPU dialect ops to include stream as an optional argument in the future. Till then, we plan to use this extended dialect.

## Questions

Do we need more operations like context & devices to be exposed?
