# XeTile IMEX-4 test generator and perf reporter
This directory contains test generating infrastructure that measures performance for a set of test cases and reports in excel format.
One needs a csv file with test cases in this format:
```
BatchSize,M,K,N,dtype,wgm,wgn,sgm,sgn,sgk
1,128,1000,2000,bf16,64,64,8,16,64
1,128,1001,2048,bf16,64,64,8,16,64
1,128,32768,256,bf16,64,64,8,16,64
1,16,1024,1024,bf16,64,64,8,16,64
```
_Note: wg* and sg* here mean work group and subgroup tile sizes, sgk means step size in the k-loop
Matrix A is (MxK), B is (KxN), C is (MxN)._

To get reports for different code versions, run `run_tests.sh`, you need to specify 4 parameters:
1. `--test_csv` - path to the `.csv` file with testcases in the format mentioned above.
2. `--validate` - 0 or 1, default is 0. Validation tests won't be profiled.
3. `--gen_default_cases` - 0 or 1, default is 0. Additionally generates hardcoded test cases for a quick check (e.g., 4kx4k, batched 1kx1k).
4. `--report_dir` - path to the directory where test reports in text format will be stored, default is `GEMM_reports`, directory is created if it doesn't exist.
5. `--verbose`: 0 or 1, default is 0. If set to 1, outputs a command for each test that was used to run it.
6. `--llvm_build_dir`: path to the LLVM directory where imex was built.
Example usage:
    Generate tests from csv:
    ```
    ./run_tests.sh --test_csv=input_shapes.csv --validate=1 --report_dir=GEMM_reports --llvm_build_dir=../../../llvm-project/build
    ```
    Generate only hardcoded tests (e.g., 4kx4k, 1kx1k) for a quick check:
    ```
    ./run_tests.sh --gen_default_cases=1 --validate=0 --verbose=1 --llvm_build_dir=../../../llvm-project/build
    ```

It executes the following workflow for various options:
1. `gen_xetile_from_shapes.py` reads the tests csv file, for each test case generates a corresponding `.mlir` file.
2. All generated files are executed in the profiling mode and the reported time measurements (and errors) are aggregated in a text file.
3. `report_to_excel.py` parses the text file and fills the excel spreadsheet, it will also contain additional columns with formulas for TFLOPS and Speedup (colored).

Currently, it generates GEMM code for the baseline implementation and the prefetch version.


---------------
`xetile_testgen.py` has 4 parameters that should be specified:
1. `--code_version` - string, currently test generator supports `baseline` and `prefetch`.
2. `--validate` - 0 or 1, controls CPU validation. When set to 0, no CPU validation will be performed and env variables `IMEX_ENABLE_PROFILING` will be set to report kernel time.
3. `--print_debug` - 0 or 1, prints some debug info regarding tile shapes and layouts.
4. `--test_csv` - path to the `.csv` file with testcases in the format mentioned above.
`--prefetch` can be true or false: whether to generate code with prefetch or not
-------------------

`report_to_excel.py` has 3 parameters:
1. `--reports_dir` - path to the directory with report text files, MUST contain baseline.txt if creating spreadsheet from scratch.
2. `--report_name` - path to the report that will be used to update the spreadsheet from `--sheet_name` by appending 3 columns to the right: `time,TFLOPS,speedup`.
3. `--sheet_name` - path to the spreadsheet to be updated (if `--report_name`` is specified), otherwise will be used as the name of the spreadsheet to build.

Example build from scratch based on reports from directory:
```
python3 report_to_excel.py --reports_dir=../mydir
```
Example update existing:
```
python3 report_to_excel.py --reports_name=../new_code_variant.txt --sheet_name=../existing_sheet.xlsx
```
