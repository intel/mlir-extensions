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
1. `--test_csv`: The path to the `.csv` file with testcases in the format mentioned above.
2. `--test_dir`: The path to the directory where all the generated test mlir file will reside. Default `generated-gemm`.
3. `--report_dir`: The path to the directory where test reports in text format will be stored, default is `gemm_reports`, directory is created if it doesn't exist.
4. `--sheet_name`: The name of the final output xlsx sheet collating all resutls.
5. `--llvm_build_dir`: The path to the LLVM directory where imex was built.
6. `--validate`: 0 or 1, the default is 0. Validation tests won't be profiled.
7. `--runtime`: The name of the runtime, either `l0-runtime` (default) or `sycl-runtime`.
8. `--code_version`: To separate `baseline` and `prefetch` runs, will run both if not specified.
9. `--gen_default_cases`: 0 or 1, the default is 0. Additionally generates hardcoded test cases for a quick check (e.g., 4kx4k, batched 1kx1k).
10. `--verbose`: 0 or 1, the default is 0. If set to 1, outputs a command for each test that was used to run it.
11. `--timeout`: A timeout can be set for longer tests, uses GNU `timeout`, see `timeout --help`.

### Example usage:

Generate tests from csv:

```bash
./run_tests.sh --test_csv=input_shapes.csv --validate=1 --report_dir=gemm_reports --llvm_build_dir=../../../llvm-project/build
```

Generate only hardcoded tests (e.g., 4kx4k, 1kx1k) for a quick check:
    
```bash
./run_tests.sh --gen_default_cases=1 --validate=0 --verbose=1 --llvm_build_dir=../../../llvm-project/build
```

Specify the final output sheet name, use a `sycl-runtime` and run only `baseline` code:

```bash
./run_tests.sh --gen_default_cases=1 --validate=0 --verbose=1 --llvm_build_dir=../../../llvm-project/build --sheet_name=result.xlsx --runtime=sycl-runtime --code_version=baseline
```

It executes the following workflow for various options:
1. `gen_xetile_from_shapes.py` reads the tests csv file, for each test case generates a corresponding `.mlir` file.
2. All generated files are executed in the profiling mode and the reported time measurements (and errors) are aggregated in a text file.
3. `report_to_excel.py` parses the text file and fills the excel spreadsheet, it will also contain additional columns with formulas for TFLOPS and Speedup (colored).

Currently, it generates GEMM code for the baseline implementation and the prefetch version.

-------------------

`xetile_testgen.py` has 4 parameters that should be specified:
1. `--code_version` - string, currently test generator supports `baseline` and `prefetch`.
2. `--validate` - 0 or 1, controls CPU validation. When set to 0, no CPU validation will be performed and env variables `IMEX_ENABLE_PROFILING` will be set to report kernel time.
3. `--print_debug` - 0 or 1, prints some debug info regarding tile shapes and layouts.
4. `--test_csv` - path to the `.csv` file with testcases in the format mentioned above.

-------------------

`report_to_excel.py` has 3 parameters:
1. `--reports_dir` - path to the directory with report text files, MUST contain baseline.txt if creating spreadsheet from scratch.
2. `--report_name` - path to the report that will be used to update the spreadsheet from `--sheet_name` by appending 3 columns to the right: `time,TFLOPS,speedup`.
3. `--sheet_name` - path to the spreadsheet to be updated (if `--report_name`` is specified), otherwise will be used as the name of the spreadsheet to build.

Example build from scratch based on reports from directory:

```bash
python3 report_to_excel.py --reports_dir=../mydir
```
Example update existing:

```bash
python3 report_to_excel.py --reports_name=../new_code_variant.txt --sheet_name=../existing_sheet.xlsx
```
