#!/bin/bash

TEST_CSV=""
TEST_DIR=generated-gemm
REPORT_DIR=gemm-reports
SHEET_NAME=speedup-report.xlsx
LLVM=""
VALIDATE=0
RT=l0-runtime
CODE_VERSION=""
GEN_DEFAULT_CASES=0
VERBOSE=0
TIMEOUT=0
while (( $# >= 1 )); do
    case $1 in
        --test_csv=*)
            TEST_CSV="${1#*=}"
            ;;
        --test_dir=*)
            TEST_DIR="${1#*=}"
            ;;
        --report_dir=*)
            REPORT_DIR="${1#*=}"
            ;;
        --sheet_name=*)
            SHEET_NAME="${1#*=}"
            ;;
        --llvm_build_dir=*)
            LLVM="${1#*=}"
            ;;
        --validate=*)
            VALIDATE="${1#*=}"
            ;;
        --runtime=*)
            RT="${1#*=}"
            ;;
        --code_version=*)
            CODE_VERSION="${1#*=}"
            ;;
        --gen_default_cases=*)
            GEN_DEFAULT_CASES="${1#*=}"
            ;;
        --verbose=*)
            VERBOSE="${1#*=}"
            ;;
        --timeout=*)
            TIMEOUT="${1#*=}"
            ;;
        *)
            break
            ;;
    esac
    shift
done

if [ "$LLVM" = "" ]; then
    echo "Please specify the LLVM build directory as --llvm_build_dir=YOUR_PATH"
    exit 255
fi

if [ "$TEST_CSV" = "" ]; then
    echo "No csv file was specified in --test_csv"
fi

if [ $GEN_DEFAULT_CASES = 0 ]; then
    echo "Not generating additional default test cases"
elif [ $GEN_DEFAULT_CASES = 1 ]; then
    echo "Generating additional default test cases"
fi

if [ $VALIDATE = 0 ]; then
    echo "Running in profiling mode"
    export IMEX_ENABLE_PROFILING=1
elif [ $VALIDATE = 1 ]; then
    echo "Running in validation mode"
    unset IMEX_ENABLE_PROFILING
fi

if [[ -z "$CODE_VERSION" ]]; then
    CODE_VERSIONS=(baseline prefetch)
elif [[ "$CODE_VERSION" == "baseline" ]]; then
    CODE_VERSIONS=(baseline)
elif [[ "$CODE_VERSION" == "prefetch" ]]; then
    CODE_VERSIONS=(prefetch)
else 
    echo "Invalid code version, use either 'baseline' or 'prefetch'."
    exit 255
fi

export IMEX_ENABLE_LARGE_REG_FILE=1

mkdir -p $REPORT_DIR
RESULT_CMD=0
for CODE_VER in ${CODE_VERSIONS[@]}; do
    TEST_REPORT=$CODE_VER".txt"
    REPORT_PATH=$REPORT_DIR/$TEST_REPORT
    rm -f $REPORT_PATH
    echo $'\nTesting code version:' $CODE_VER
    python3 xetile_testgen.py --code_version=$CODE_VER --validate=$VALIDATE --print_debug=0 --test_csv=$TEST_CSV --default_tests=$GEN_DEFAULT_CASES --output_gemm_dir=$TEST_DIR
    CUR_TEST_DIR=$TEST_DIR/$CODE_VER
    if [[ -z "$TEST_CSV" ]]; then
        # if no test $TEST_CSV is given, list whatever in the $CUR_TEST_DIR
        TEST_NAMES=($(ls $CUR_TEST_DIR -1))
    else
        # else, run the test cases as the same order in the $TEST_CSV file
        if [[ "$VALIDATE" == "0" ]]; then
            TEST_NAMES=($(cat -n $TEST_CSV | tail -n +2 \
                | awk -F' ' '{printf "%s,%s\n", $1,$2}' \
                | grep -v '#' \
                | awk -F',' '{printf "sg_gemm_%s_%sx%sx%s_wgm%s_wgn%s_sgm%s_sgn%s_sgk%s.mlir ", $2, $3, $4, $5, $7, $8, $9, $10, $11}'))
            LINE_NOS=($(cat -n $TEST_CSV | tail -n +2 \
                | awk -F' ' '{printf "%s,%s\n", $1,$2}' \
                | grep -v '#' \
                | awk -F',' '{print $1}'))
        else
            TEST_NAMES=($(cat -n $TEST_CSV | tail -n +2 \
                | awk -F' ' '{printf "%s,%s\n", $1,$2}' \
                | grep -v '#' \
                | awk -F',' '$2 > 5000 || $3 > 5000 || $4 > 5000 {next}1' \
                | awk -F',' '{printf "sg_gemm_%s_%sx%sx%s_wgm%s_wgn%s_sgm%s_sgn%s_sgk%s.mlir ", $2, $3, $4, $5, $7, $8, $9, $10, $11}'))
            LINE_NOS=($(cat -n $TEST_CSV | tail -n +2 \
                | awk -F' ' '{printf "%s,%s\n", $1,$2}' \
                | grep -v '#' \
                | awk -F',' '$3 > 5000 || $4 > 5000 || $5 > 5000 {next}1' \
                | awk -F',' '{print $1}'))
        fi
        if [[ "${#TEST_NAMES[@]}" != "${#LINE_NOS[@]}" ]]; then
            echo "\${#TEST_NAME[@]} != \${#LINE_NOS[@]}. Aborting."
            exit 255
        fi
    fi
    
    set -o pipefail
    n=0
    for TEST_NAME in ${TEST_NAMES[@]}; do
        TEST_CASE=$CUR_TEST_DIR/$TEST_NAME
        if [ -f "$TEST_CASE" ]; then
            echo "${LINE_NOS[$n]}. Testing $(basename $TEST_CASE)" | tee -a $REPORT_PATH # report parsing uses this line
            if [[ "$RT" == "l0-runtime" ]]; then
                CMD="python3 $LLVM/bin/imex-runner.py --requires=l0-runtime -i $TEST_CASE  \
                --pass-pipeline-file=../../test/Integration/Dialect/XeTile/xetile-to-func-vc.pp \
                --runner imex-cpu-runner -e main --entry-point-result=void \
                --shared-libs=$LLVM/lib/libimex_runner_utils.so,$LLVM/lib/libmlir_runner_utils.so,$LLVM/lib/libmlir_c_runner_utils.so,$LLVM/lib/liblevel-zero-runtime.so"
            elif [[ "$RT" == "sycl-runtime" ]]; then
                CMD="python3 $LLVM/bin/imex-runner.py --requires=sycl-runtime -i $TEST_CASE  \
                --pass-pipeline-file=../../test/Integration/Dialect/XeTile/xetile-to-func-vc.pp \
                --runner imex-cpu-runner -e main --entry-point-result=void \
                --shared-libs=$LLVM/lib/libimex_runner_utils.so,$LLVM/lib/libmlir_runner_utils.so,$LLVM/lib/libmlir_c_runner_utils.so,$LLVM/lib/libsycl-runtime.so"
            fi
            if [ $VERBOSE -eq 1 ]; then
                echo $CMD | tee -a $REPORT_PATH
            fi
            timeout --signal=SIGRTMAX $TIMEOUT $CMD |& tee -a $REPORT_PATH
            tmp_res=$?
            if [ $tmp_res -ne 0 ]; then
                RESULT_CMD=1
            fi
            echo "" | tee -a $REPORT_PATH # new line
        fi
        n=$(($n+1))
    done
done

unset IMEX_ENABLE_LARGE_REG_FILE
unset IMEX_ENABLE_PROFILING

python3 report_to_excel.py  --reports_dir=$REPORT_DIR --sheet_name=$SHEET_NAME
exit $RESULT_CMD
