# -*- Python -*-

import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'DPCOMP_OPT'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.mlir']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%PYTHON', config.python_executable))

llvm_config.with_system_environment(
    ['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

#llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ['runlit.py', 'lit.cfg.py', 'CMakeLists.txt']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.dpcomp_obj_root, 'test')
config.dpcomp_tools_dir = os.path.join(config.dpcomp_obj_root, 'tools')

tool_dirs = [
    os.path.join(config.dpcomp_tools_dir,'FileCheck'),
    os.path.join(config.dpcomp_tools_dir,'dpcomp-opt'),
]

tools = [
    'FileCheck',
    'dpcomp-opt',
]

llvm_config.add_tool_substitutions(tools, tool_dirs)
