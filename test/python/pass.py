# RUN: %python_executable %s | FileCheck %s

import gc
from imex_mlir.ir import Context, Location, Module
from imex_mlir.passmanager import PassManager
from imex_mlir.dialects import region


def run(f):
    print("\nTEST:", f.__name__)
    f()
    gc.collect()
    assert Context._get_live_count() == 0
    return f


# CHECK-LABEL: TEST: testDropRegion
# CHECK: func.func @main() -> i32 {
# CHECK-NEXT: [[C0:%.+]] = arith.constant 42 : i32
# CHECK-NEXT: return [[C0]] : i32
@run
def testDropRegion():
    ir = """module {
  func.func @main() -> i32 {
    %0 = region.env_region "my_region" -> i32 {
      %1 = arith.constant 42 : i32
      region.env_region_yield %1 : i32
    }
    return %0 : i32
  }
}
"""
    pipeline = "builtin.module(drop-regions)"
    with Context() as ctx, Location.unknown():
        region.register_dialect(ctx)
        module = Module.parse(ir)

    pm = PassManager.parse(pipeline, context=module.context)
    pm.run(module.operation)
    module.operation.print()
    assert module.operation.verify()
