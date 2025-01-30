PYTHON=`which python`
ROOT=`git rev-parse --show-toplevel`
SCRIPT=$ROOT/build/bin/imex-runner.py
PIPELINE=$ROOT/test/Integration/Dialect/XeTile/xetile-to-func-vc.pp
LEVEL_ZERO=$ROOT/build/lib/liblevel-zero-runtime.so
IMEX_RUNNER_UTILS=$ROOT/build/lib/libimex_runner_utils.so
MLIR_RUNNER_UTILS=$ROOT/../llvm-project/build/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS=$ROOT/../llvm-project/build/lib/libmlir_c_runner_utils.so
echo "ROOT= $ROOT"
echo "SCRIPT= $SCRIPT"
TEST=$1
IMEX_ENABLE_LARGE_REG_FILE=1 $PYTHON $SCRIPT --requires=l0-runtime -a -i $TEST --pass-pipeline-file=$PIPELINE --runner imex-cpu-runner -e main --entry-point-result=void --shared-libs=$IMEX_RUNNER_UTILS,$MLIR_RUNNER_UTILS,$MLIR_C_RUNNER_UTILS,$LEVEL_ZERO
