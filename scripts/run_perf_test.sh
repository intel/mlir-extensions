#!/usr/bin/sh
HOME_DIR=/home/gta/actions-runner/_work/frameworks.ai.mlir.mlir-extensions/frameworks.ai.mlir.mlir-extensions/gpurefactorbuild
imexDir=$HOME_DIR/mlir-extensions
llvmDir=${HOME_DIR}/llvm-mlir/_mlir_install

#user config
perfPipeline=test/PlaidML/linalg-to-llvm.pp
perfDir=(test/PlaidML)

#gather test
testList=()
for dir in ${perfDir[@]}; do
    testDir=${imexDir}/${dir}
    excludeList=$(cat ${testDir}/lit.local.cfg)
    files=$(ls $testDir)
    for file in ${files[@]}; do
      if [[ ${file} =~ ".mlir" && ! ${excludeList[*]} =~ ${file} ]]; then
        testList+=(${dir}/$file)
      fi
    done
done

echo "Perf Test(SYCL)" > report.txt
#run test
export IMEX_ENABLE_PROFILING=ON
runTest() {
  result=$(python ${imexDir}/build/bin/imex-runner.py -e main -entry-point-result=void --pass-pipeline-file=${imexDir}/${perfPipeline} --shared-libs=${llvmDir}/lib/libmlir_runner_utils.so,${llvmDir}/lib/libmlir_c_runner_utils.so,${imexDir}/build/lib/libsycl-runtime.so -i $1)
  while IFS= read -r line; do
    if [[ $line =~ ([0-9]+.?[0-9]+ ms) ]]; then
      echo ${BASH_REMATCH[1]}
    fi
  done <<< $result >> report.txt
}
for test in ${testList[@]}; do
    echo ${test} >> report.txt
    runTest ${imexDir}/${test}
done
