conda activate test_env
chmod -R 777 /localdisk/nbpatel/mlir-llvm
export LLVM_PATH=/localdisk/nbpatel/mlir-llvm
export TBB_PATH=/localdisk/nbpatel/mlir-extensions/tbb
export LEVEL_ZERO_DIR=/localdisk/nbpatel/mlir-extensions/level-zero/level_zero_install
cd build
