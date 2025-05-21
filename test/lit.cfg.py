# -*- Python -*-

import os
import platform
import re
import sys
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'IMEX'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.mlir']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.imex_obj_root, 'test')

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))
config.substitutions.append(('%mlir_lib_dir', config.mlir_runner_utils_dir))
config.substitutions.append(('%imex_tools_dir', config.imex_tools_dir))
config.substitutions.append(('%mlir_runner_utils', config.mlir_runner_utils))
config.substitutions.append(('%mlir_c_runner_utils', config.mlir_c_runner_utils))
if config.enable_vulkan_runner:
    config.substitutions.append(('%vulkan_runtime_wrappers', config.vulkan_runtime_wrappers))
config.substitutions.append(('%imex_runner', config.imex_runner))
config.substitutions.append(('%python_executable', config.python_executable))
if config.imex_enable_sycl_runtime:
    config.substitutions.append(('%mlir_sycl_runtime', config.mlir_sycl_runtime))
    config.substitutions.append(('%sycl_runtime', config.sycl_runtime))
if config.imex_enable_l0_runtime:
    config.substitutions.append(('%levelzero_runtime', config.levelzero_runtime))
if config.imex_enable_igpu:
    config.substitutions.append(('%igpu_fp64', config.igpu_fp64))
config.substitutions.append(('%irunner_utils', config.imex_runner_utils))

llvm_config.with_system_environment(
    ['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ['Inputs', 'Examples', 'Gen', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt']

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.imex_obj_root, 'test')

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', os.path.normpath(config.llvm_tools_dir), append_path=True)
llvm_config.with_environment('PATH', os.path.normpath(config.imex_tools_dir), append_path=True)

tool_dirs = [os.path.normpath(config.imex_tools_dir),
             os.path.normpath(config.llvm_tools_dir)]
tools = [
    'imex-opt',
    'imex-runner.py'
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

llvm_config.with_environment(
    "PYTHONPATH",
    [
        os.path.join(config.imex_obj_root, "python_packages"),
    ],
    append_path=True,
)
