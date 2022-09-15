// Copyright 2022 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NTENSOR_OPS
#define NTENSOR_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

def NTensor_Dialect : Dialect {
  let name = "ntensor";
  let cppNamespace = "::imex::ntensor";

  let hasConstantMaterializer = 1;
  let useDefaultTypePrinterParser = 1;
}

class NTensor_OpBase<string mnemonic, list<Trait> traits = []>
    : Op<NTensor_Dialect, mnemonic, traits>;



#endif // NTENSOR_OPS
