// RUN: imex-opt -allow-unregistered-dialect --gpux-to-spirv -split-input-file %s | FileCheck %s

module attributes {gpu.container_module, spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Int64, Kernel], []>, #spirv.resource_limits<>>} {
  func.func @builtin() {
    %c0 = arith.constant 1 : index
    gpu.launch_func @kernels::@builtin_workgroup_size_x
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
    return
  }

  // CHECK-LABEL:  spirv.module @{{.*}}
  // CHECK: spirv.GlobalVariable [[VALUE:@.*]] built_in("WorkgroupSize")
  gpu.module @kernels {
    gpu.func @builtin_workgroup_size_x() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      // CHECK: [[ADDRESS:%.*]] = spirv.mlir.addressof [[VALUE]]
      // CHECK-NEXT: [[VEC:%.*]] = spirv.Load "Input" [[ADDRESS]] : vector<3xi64>
      // CHECK-NEXT: {{%.*}} = spirv.CompositeExtract [[VEC]]{{\[}}0 : i32{{\]}}
      %0 = gpu.block_dim x
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module, spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Int64, Kernel], []>, #spirv.resource_limits<>>} {
  func.func @builtin() {
    %c0 = arith.constant 1 : index
    gpu.launch_func @kernels::@builtin_workgroup_size_x
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
    return
  }

  // CHECK-LABEL:  spirv.module @{{.*}}
  // CHECK: spirv.GlobalVariable [[VALUE:@.*]] built_in("NumWorkgroups")
  gpu.module @kernels {
    gpu.func @builtin_workgroup_size_x() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      // CHECK: [[ADDRESS:%.*]] = spirv.mlir.addressof [[VALUE]]
      // CHECK-NEXT: [[VEC:%.*]] = spirv.Load "Input" [[ADDRESS]] : vector<3xi64>
      // CHECK-NEXT: {{%.*}} = spirv.CompositeExtract [[VEC]]{{\[}}0 : i32{{\]}}
      %0 = gpu.grid_dim x
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module, spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Int64, Kernel], []>, #spirv.resource_limits<>>} {
  func.func @builtin() {
    %c0 = arith.constant 1 : index
    gpu.launch_func @kernels::@builtin_workgroup_size_x
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
    return
  }

  // CHECK-LABEL:  spirv.module @{{.*}}
  // CHECK: spirv.GlobalVariable [[VALUE:@.*]] built_in("LocalInvocationId")
  gpu.module @kernels {
    gpu.func @builtin_workgroup_size_x() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      // CHECK: [[ADDRESS:%.*]] = spirv.mlir.addressof [[VALUE]]
      // CHECK-NEXT: [[VEC:%.*]] = spirv.Load "Input" [[ADDRESS]] : vector<3xi64>
      // CHECK-NEXT: {{%.*}} = spirv.CompositeExtract [[VEC]]{{\[}}0 : i32{{\]}}
      %0 = gpu.thread_id x
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module, spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Int64, Kernel], []>, #spirv.resource_limits<>>} {
  func.func @builtin() {
    %c0 = arith.constant 1 : index
    gpu.launch_func @kernels::@builtin_workgroup_size_x
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
    return
  }

  // CHECK-LABEL:  spirv.module @{{.*}}
  // CHECK: spirv.GlobalVariable [[VALUE:@.*]] built_in("WorkgroupId")
  gpu.module @kernels {
    gpu.func @builtin_workgroup_size_x() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      // CHECK: [[ADDRESS:%.*]] = spirv.mlir.addressof [[VALUE]]
      // CHECK-NEXT: [[VEC:%.*]] = spirv.Load "Input" [[ADDRESS]] : vector<3xi64>
      // CHECK-NEXT: {{%.*}} = spirv.CompositeExtract [[VEC]]{{\[}}0 : i32{{\]}}
      %0 = gpu.block_id x
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module, spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Int64, Kernel], []>, #spirv.resource_limits<>>} {
  func.func @builtin() {
    %c0 = arith.constant 1 : index
    gpu.launch_func @kernels::@builtin_workgroup_size_x
        blocks in (%c0, %c0, %c0) threads in (%c0, %c0, %c0)
    return
  }

  // CHECK-LABEL:  spirv.module @{{.*}}
  // CHECK: spirv.GlobalVariable [[VALUE:@.*]] built_in("GlobalInvocationId")
  gpu.module @kernels {
    gpu.func @builtin_workgroup_size_x() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      // CHECK: [[ADDRESS:%.*]] = spirv.mlir.addressof [[VALUE]]
      // CHECK-NEXT: [[VEC:%.*]] = spirv.Load "Input" [[ADDRESS]] : vector<3xi64>
      // CHECK-NEXT: {{%.*}} = spirv.CompositeExtract [[VEC]]{{\[}}0 : i32{{\]}}
      %0 = gpu.global_id x
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module, spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Int64, Kernel], []>, #spirv.resource_limits<>>} {
  // CHECK-LABEL:  spirv.module @{{.*}}
  // CHECK: spirv.GlobalVariable [[VALUE:@.*]] built_in("SubgroupId")
  gpu.module @kernels {
    gpu.func @builtin_subgroup_id() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      // CHECK: [[ADDRESS:%.*]] = spirv.mlir.addressof [[VALUE]]
      // CHECK-NEXT: {{%.*}} = spirv.Load "Input" [[ADDRESS]] : i32
      %0 = gpu.subgroup_id : index
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module, spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Int64, Kernel], []>, #spirv.resource_limits<>>} {
  // CHECK-LABEL:  spirv.module @{{.*}}
  // CHECK: spirv.GlobalVariable [[VALUE:@.*]] built_in("NumSubgroups")
  gpu.module @kernels {
    gpu.func @builtin_subgroup_id() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      // CHECK: [[ADDRESS:%.*]] = spirv.mlir.addressof [[VALUE]]
      // CHECK-NEXT: {{%.*}} = spirv.Load "Input" [[ADDRESS]] : i32
      %0 = gpu.num_subgroups : index
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module, spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Int64, Kernel], []>, #spirv.resource_limits<>>} {
  // CHECK-LABEL:  spirv.module @{{.*}}
  // CHECK: spirv.GlobalVariable [[VALUE:@.*]] built_in("SubgroupSize")
  gpu.module @kernels {
    gpu.func @builtin_subgroup_id() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      // CHECK: [[ADDRESS:%.*]] = spirv.mlir.addressof [[VALUE]]
      // CHECK-NEXT: {{%.*}} = spirv.Load "Input" [[ADDRESS]] : i32
      %0 = gpu.subgroup_size : index
      gpu.return
    }
  }
}
