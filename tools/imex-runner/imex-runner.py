#===- imex-runner.py ----------------------------------------*- Python -*-===#
#
# Copyright 2022 Intel Corporation
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https:#llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===#
#
# This file defines a runner for imex.
#
#===----------------------------------------------------------------------===#

"""
Run a mlir file through imex-opt using a user provided pass pipeline
and run resulting output through mlir-runner.
The pass pipeline is provided either in a file or as a string; expected
syntax is similar to mlir-opt's syntax.

If provided in a file passes can be separted by newline or ',' and
trailing/leading whitespaces will be eliminated.

Example: the following specs are equivalent:
string or content of file:
`func.func(scf-bufferize,shape-bufferize)`
content of file:
```
func.func(scf-bufferize,
          shape-bufferize)
```

Pass pipelines privded as strings will nnot be modified.

All unknown arguments will be forwarded to mlir-runner. Currently there is no
option to forward user-provided args to imex-opt.

To use this in a lit test, you can do something similar to
`// RUN: %{python_executable} %{imex_tools_dir}/imex-runner.py -i %s --pass-pipeline-file=%p/ptensor.pp -e main -entry-point-result=void --shared-libs=%{mlir_shlib_dir}/libmlir_c_runner_utils%shlibext --shared-libs=%{mlir_shlib_dir}/libmlir_runner_utils%shlibext | FileCheck %s`
"""

import os, sys, re
from os.path import join as jp
import argparse
import subprocess

parser = argparse.ArgumentParser(
    description="run imex-opt and pipe into mlir-runner"
)
parser.add_argument("--input-file", "-i", default=None, help="input MLIR file")
parser.add_argument("--pass-pipeline-file", "-f", default=None, help="file defining pass pipeline")
parser.add_argument("--pass-pipeline", "-p", default=None, help="pass pipeline (string)")

args, unknown = parser.parse_known_args()

ppipeline = None
if args.pass_pipeline_file:
    with open(args.pass_pipeline_file, 'r') as infile:
        for l in infile:
            ppipeline = ','.join([ppipeline, l]) if ppipeline else l
        # get rid of whitespaces, tabs and newlines
        ppipeline = re.sub(r"[\s\n\t]*", "", ppipeline)
        # get rid of duplicate ','s
        ppipeline = re.sub(r",+", ",", ppipeline)
        # get rid of trailing ','s
        ppipeline = re.sub(r",\)", ")", ppipeline)

# build imex-opt command
cmd = ["imex-opt"]
if ppipeline:
    cmd.append(f'--pass-pipeline={ppipeline}')
elif args.pass_pipeline:
    cmd.append(f'--pass-pipeline={args.pass_pipeline}')
if args.input_file:
    cmd.append(args.input_file)
# run and feed into pipe
p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE)
# build mlir-opt command: all unknown args will be passed to imex-opt
cmd = ["mlir-cpu-runner"] + unknown
# get stdout from imex-opt and pipe into mlir-opt
p2 = subprocess.Popen(cmd, stdin=p1.stdout)
p1.wait()
p2.wait()
