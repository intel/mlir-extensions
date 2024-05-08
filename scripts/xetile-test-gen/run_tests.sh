#!/bin/bash

REPORT_DIR=GEMM_reports
LLVM=""
TEST_CSV=""
VALIDATE=0
GEN_DEFAULT_CASES=0
VERBOSE=0
while (( $# >= 1 )); do
    case $1 in
        --test_csv=*)
            TEST_CSV="${1#*=}"
            ;;
        --validate=*)
            VALIDATE="${1#*=}"
            ;;
        --report_dir=*)
            REPORT_DIR="${1#*=}"
            ;;
        --gen_default_cases=*)
            GEN_DEFAULT_CASES="${1#*=}"
            ;;
        --verbose=*)
            VERBOSE="${1#*=}"
            ;;
        --llvm_build_dir=*)
            LLVM="${1#*=}"
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


export IMEX_ENABLE_LARGE_REG_FILE=1

TEST_DIR="Generated_GEMM"
mkdir -p $REPORT_DIR
for CODE_VERSION in baseline prefetch
do
    TEST_REPORT=$CODE_VERSION".txt"
    REPORT_PATH=$REPORT_DIR/$TEST_REPORT
    rm -f $REPORT_PATH
    echo $'\nTesting code version:' $CODE_VERSION
    python3 xetile_testgen.py --code_version=$CODE_VERSION --validate=$VALIDATE --print_debug=0 --test_csv=$TEST_CSV --default_tests=$GEN_DEFAULT_CASES
    CUR_TEST_DIR=$TEST_DIR/$CODE_VERSION
    return_value=0
    for TEST_CASE in $CUR_TEST_DIR/*
    do
        if [ -f "$TEST_CASE" ]
        then
            echo "Testing $(basename $TEST_CASE)" | tee -a $REPORT_PATH # report parsing uses this line
            CMD="python3 $LLVM/bin/imex-runner.py --requires=l0-runtime -i $TEST_CASE  \
                --pass-pipeline-file=../../test/Integration/Dialect/XeTile/xetile-to-llvm.pp \
                --runner imex-cpu-runner -e main --entry-point-result=void \
                --shared-libs=$LLVM/lib/libimex_runner_utils.so,$LLVM/lib/libmlir_runner_utils.so,$LLVM/lib/libmlir_c_runner_utils.so,$LLVM/lib/liblevel-zero-runtime.so; echo $?"
            if [ $VERBOSE -eq 1 ]; then
                echo $CMD | tee -a $REPORT_PATH
            fi
            eval $CMD |& tee -a $REPORT_PATH
            echo $return_value
            if [ $return_value -ne 0 ]; then
                exit 1
            fi
            echo "" | tee -a $REPORT_PATH # new line
        fi
    done
done

unset IMEX_ENABLE_LARGE_REG_FILE
unset IMEX_ENABLE_PROFILING

python3 report_to_excel.py  --reports_dir=$REPORT_DIR
