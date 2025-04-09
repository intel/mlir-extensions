# RUN: %python_executable %s | FileCheck %s

import gc
from imex_mlir.ir import Context, InsertionPoint, Location, Block
from imex_mlir.ir import IntegerType, StringAttr
from imex_mlir.dialects import builtin
from imex_mlir.dialects import arith
from imex_mlir.dialects import region


def run(f):
    print("\nTEST:", f.__name__)
    f()
    gc.collect()
    assert Context._get_live_count() == 0
    return f


# CHECK-LABEL: TEST: testRegionOp
# CHECK: module {
# CHECK:  [[C0:%.+]] = region.env_region "my_region" -> i32 {
# CHECK-NEXT: [[C1:%.+]] = arith.constant 42 : i32
# CHECK-NEXT: region.env_region_yield [[C1]] : i32
@run
def testRegionOp():
    with Context() as ctx, Location.unknown():
        region.register_dialect(ctx)
        i32 = IntegerType.get_signless(32)
        module = builtin.ModuleOp()
        with InsertionPoint(module.body):
            region_attr = StringAttr.get("my_region")
            results = [i32]
            args = []
            region_op = region.EnvironmentRegionOp(results, region_attr, args)
            block = Block.create_at_start(region_op.regions[0], [])
            with InsertionPoint(block):
                const_op = arith.ConstantOp(i32, 42)
                region.EnvironmentRegionYieldOp(const_op)

    module.print()
    assert module.verify()
