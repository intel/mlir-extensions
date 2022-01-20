//*****************************************************************************
// Copyright 2022 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

//===----------------------------------------------------------------------===//
//
// This file defines the base for the DNNLGraph dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GPURUNTIME_BASE
#define GPURUNTIME_BASE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// GPU Runtime dialect base definition
//===----------------------------------------------------------------------===//

def GpuRuntime_Dialect : Dialect {
  let name = "gpu_runtime";
  let description = [{
      GpuRuntime Dialect for representing Gpu runtime ops for level zero runtime
    }];
  let cppNamespace = "gpu_runtime";
}

class GpuRuntime_Op<string mnemonic, list<OpTrait> traits = []> :
      Op<GpuRuntime_Dialect, mnemonic, traits>;

#endif // GPURUNTIME_BASE
