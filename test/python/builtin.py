# RUN: %python_executable %s | FileCheck %s

import gc
from imex_mlir.ir import Context, InsertionPoint, Location, IntegerType
from imex_mlir.dialects import builtin, arith


def run(f):
    print("\nTEST:", f.__name__)
    f()
    gc.collect()
    assert Context._get_live_count() == 0
    return f


# CHECK-LABEL: TEST: testConstOp
# CHECK: module {
# CHECK-NEXT: [[C0:%.+]] = arith.constant 42 : i32
@run
def testConstOp():
    with Context(), Location.unknown():
        i32 = IntegerType.get_signless(32)
        module = builtin.ModuleOp()
        with InsertionPoint(module.body):
            arith.ConstantOp(i32, 42)
    module.print()
    assert module.verify()
