# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys
import subprocess
from setuptools import find_packages, setup
import versioneer
import numpy

IS_WIN = False
IS_LIN = False
IS_MAC = False

if "linux" in sys.platform:
    IS_LIN = True
elif sys.platform == "darwin":
    IS_MAC = True
elif sys.platform in ["win32", "cygwin"]:
    IS_WIN = True
else:
    assert False, sys.platform + " not supported"

# CMAKE =======================================================================


def buildSyclMathRuntime(root_path, build_prefix, install_prefix, use_mkl, use_sycl):
    IMEX_USE_MKL = use_mkl
    IMEX_USE_SYCL = use_sycl
    MATH_SYCL_RUNTIME_INSTALL_PATH = install_prefix

    cmake_build_dir_parent = os.path.join(build_prefix)
    cmake_build_dir = os.path.join(build_prefix, "sycl/math_runtime")
    sycl_math_runtime_path = os.path.join(root_path, "numba_dpcomp/math_runtime/sycl")
    cmake_cmd = [
        "cmake",
        sycl_math_runtime_path,
    ]

    cmake_cmd += ["-GNinja"]

    cmake_cmd += [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DMATH_SYCL_RUNTIME_INSTALL_PATH=" + MATH_SYCL_RUNTIME_INSTALL_PATH,
    ]

    if IMEX_USE_MKL is not None:
        cmake_cmd += ["-DIMEX_USE_MKL=" + IMEX_USE_MKL]

    if IMEX_USE_SYCL is not None:
        cmake_cmd += ["-DIMEX_USE_SYCL=" + IMEX_USE_SYCL]

    try:
        os.makedirs(cmake_build_dir)
    except FileExistsError:
        pass

    env = os.environ.copy()

    env["CC"] = "icx"
    env["CXX"] = "dpcpp"
    print(cmake_cmd)
    subprocess.check_call(
        cmake_cmd, stderr=subprocess.STDOUT, shell=False, cwd=cmake_build_dir, env=env
    )
    subprocess.check_call(
        ["cmake", "--build", ".", "--config", "Release"], cwd=cmake_build_dir
    )
    subprocess.check_call(
        ["cmake", "--install", ".", "--config", "Release"], cwd=cmake_build_dir
    )

    return ["dpcomp-math-sycl-runtime"], [MATH_SYCL_RUNTIME_INSTALL_PATH]


if int(os.environ.get("DPCOMP_SETUP_RUN_CMAKE", 1)):
    root_dir = os.path.dirname(os.path.abspath(__file__))

    LLVM_PATH = os.environ["LLVM_PATH"]
    LLVM_DIR = os.path.join(LLVM_PATH, "lib", "cmake", "llvm")
    MLIR_DIR = os.path.join(LLVM_PATH, "lib", "cmake", "mlir")
    TBB_DIR = os.path.join(os.environ["TBB_PATH"], "lib", "cmake", "tbb")
    IMEX_USE_MKL = os.environ.get("IMEX_USE_MKL")
    IMEX_USE_SYCL = os.environ.get("IMEX_USE_SYCL")
    CMAKE_INSTALL_PREFIX = os.path.join(root_dir, "..")

    cmake_build_dir = os.path.join(CMAKE_INSTALL_PREFIX, "dpcomp_cmake_build")

    extraLibs = []
    extraLibsPath = []

    install_dir = os.path.join(CMAKE_INSTALL_PREFIX, "numba_dpcomp/numba_dpcomp")
    buildSyclMathRuntime(
        root_dir, cmake_build_dir, install_dir, IMEX_USE_MKL, IMEX_USE_SYCL
    )

    cmake_cmd = [
        "cmake",
        CMAKE_INSTALL_PREFIX,
    ]

    cmake_cmd += ["-GNinja"]

    NUMPY_INCLUDE_DIR = numpy.get_include()

    cmake_cmd += [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLVM_DIR=" + LLVM_DIR,
        "-DMLIR_DIR=" + MLIR_DIR,
        "-DTBB_DIR=" + TBB_DIR,
        "-DCMAKE_INSTALL_PREFIX=" + CMAKE_INSTALL_PREFIX,
        "-DPython3_NumPy_INCLUDE_DIRS=" + NUMPY_INCLUDE_DIR,
        "-DPython3_FIND_STRATEGY=LOCATION",
        "-DIMEX_ENABLE_NUMBA_FE=ON",
        "-DIMEX_ENABLE_TBB_SUPPORT=ON",
        "-DLLVM_ENABLE_ZSTD=OFF",
    ]

    if IMEX_USE_MKL is not None:
        cmake_cmd += ["-DIMEX_USE_MKL=" + IMEX_USE_MKL]

    # DPNP
    try:
        from dpnp import get_include as dpnp_get_include

        DPNP_LIBRARY_DIR = os.path.join(dpnp_get_include(), "..", "..")
        DPNP_INCLUDE_DIR = dpnp_get_include()
        cmake_cmd += [
            "-DDPNP_LIBRARY_DIR=" + DPNP_LIBRARY_DIR,
            "-DDPNP_INCLUDE_DIR=" + DPNP_INCLUDE_DIR,
            "-DIMEX_USE_DPNP=ON",
        ]
        print("Found DPNP at", DPNP_LIBRARY_DIR)
    except ImportError:
        print("DPNP not found")

    # GPU/L0
    LEVEL_ZERO_DIR = os.getenv("LEVEL_ZERO_DIR", None)
    if LEVEL_ZERO_DIR is None:
        print("LEVEL_ZERO_DIR is not set")
    else:
        print("LEVEL_ZERO_DIR is", LEVEL_ZERO_DIR)
        cmake_cmd += [
            "-DIMEX_ENABLE_IGPU_DIALECT=ON",
        ]

    try:
        os.mkdir(cmake_build_dir)
    except FileExistsError:
        pass

    subprocess.check_call(
        cmake_cmd, stderr=subprocess.STDOUT, shell=False, cwd=cmake_build_dir
    )
    subprocess.check_call(
        ["cmake", "--build", ".", "--config", "Release"], cwd=cmake_build_dir
    )
    subprocess.check_call(
        ["cmake", "--install", ".", "--config", "Release"], cwd=cmake_build_dir
    )

# =============================================================================

packages = find_packages(where=root_dir, include=["numba_dpcomp", "numba_dpcomp.*"])

metadata = dict(
    name="numba-dpcomp",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=packages,
    install_requires=["numba>=0.56,<0.57"],
    include_package_data=True,
)

setup(**metadata)
