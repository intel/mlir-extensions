# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import subprocess
from setuptools import find_packages, setup

IS_WIN = False
IS_LIN = False
IS_MAC = False

if 'linux' in sys.platform:
    IS_LIN = True
elif sys.platform == 'darwin':
    IS_MAC = True
elif sys.platform in ['win32', 'cygwin']:
    IS_WIN = True
else:
    assert False, sys.platform + ' not supported'

# CMAKE =======================================================================

LLVM_PATH = os.environ['LLVM_PATH']
LLVM_DIR = os.path.join(LLVM_PATH, "lib", "cmake", "llvm")
MLIR_DIR = os.path.join(LLVM_PATH, "lib", "cmake", "mlir")
TBB_DIR = os.path.join(os.environ['TBB_PATH'], "lib", "cmake", "tbb")
CMAKE_INSTALL_PREFIX = os.path.join(os.getcwd(), "numba_dpcomp")

cmake_build_dir = os.path.join(os.getcwd(), "cmake_build")
cmake_cmd = [
"cmake",
]

cmake_cmd += ["-GNinja"]

cmake_cmd += [
"..",
'-DCMAKE_BUILD_TYPE=Release',
"-DLLVM_DIR=" + LLVM_DIR,
"-DMLIR_DIR=" + MLIR_DIR,
"-DTBB_DIR=" + TBB_DIR,
"-DCMAKE_INSTALL_PREFIX=" + CMAKE_INSTALL_PREFIX,
]

# DPNP
try:
    from dpnp import get_include as dpnp_get_include
    DPNP_LIBRARY_DIR = os.path.join(dpnp_get_include(), '..', '..')
    DPNP_INCLUDE_DIR = dpnp_get_include()
    cmake_cmd += [
                  '-DDPNP_LIBRARY_DIR=' + DPNP_LIBRARY_DIR,
                  '-DDPNP_INCLUDE_DIR=' + DPNP_INCLUDE_DIR,
                  '-DDPNP_ENABLE=1',
                 ]
    print("Found DPNP at", DPNP_LIBRARY_DIR)
except ImportError:
    print("DPNP not found")


# GPU/L0
LEVEL_ZERO_DIR = os.getenv('LEVEL_ZERO_DIR', None)
if LEVEL_ZERO_DIR is None:
    print('LEVEL_ZERO_DIR is not set')
else:
    print('LEVEL_ZERO_DIR is', LEVEL_ZERO_DIR)
    cmake_cmd += [
                  '-DLEVEL_ZERO_DIR=' + LEVEL_ZERO_DIR,
                  '-DGPU_ENABLE=1',
                 ]

ENABLE_CMAKE_TESTS = os.getenv('DPCOMP_ENABLE_CMAKE_TESTS', False)
if ENABLE_CMAKE_TESTS:
    cmake_cmd += ['-DBUILD_TESTING=1']

try:
    os.mkdir(cmake_build_dir)
except FileExistsError:
    pass

subprocess.check_call(cmake_cmd, stderr=subprocess.STDOUT, shell=False, cwd=cmake_build_dir)
subprocess.check_call(["cmake", "--build", ".", "--config", "Release"], cwd=cmake_build_dir)
subprocess.check_call(["cmake", "--install", ".", "--config", "Release"], cwd=cmake_build_dir)

# =============================================================================

packages = find_packages(include=["numba_dpcomp"])

metadata = dict(
    name="numba-dpcomp",
    version="0.0.1",
    packages=packages,
    # install_requires=['numba==0.54'],
    include_package_data=True,
)

setup(**metadata)
