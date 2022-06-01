# Copyright 2020-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
import sys
import tempfile
import uuid
import warnings
import argparse
import shutil


def _get_llvm_sha():
    """Reads the SHA value from llvm-sha.txt

    Raises:
        RuntimeError: If the SHA was not read properly from the llvm-sha.txt
    Returns:
        str: The SHA value read from llvm-sha.txt

    """
    sha = ""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(base_dir + "/llvm-sha.txt", "r") as fd:
        sha = fd.readline().strip()
    if len(sha) == 0:
        raise RuntimeError("sha cannot be empty")
    return sha


def _sha_matched(llvm_install_dir, llvm_sha):
    """Checks if the SHA in llvm-sha.txt matches the SHA used to build LLVM.

    Args:
        llvm_install_dir (str): Path to an LLVM installation directory
        llvm_sha (str): LLVM commit SHA value required by us to build IMEX

    Raises:
        RuntimeError: When the SHA could not be found in the LLVM installation
        direcotry.

    Returns:
        bool: True if the two SHAs matched, else False
    """
    sha = None
    with open(llvm_install_dir + "/include/llvm/Support/VCSRevision.h", "r") as fd:
        for l in fd:
            if l.find("LLVM_REVISION") > -1:
                sha = l.split()[2].strip('"')
                break
    if sha is None:
        raise RuntimeError("Should have found the SHA in a valid LLVM install")
    if sha != llvm_sha:
        warnings.warn("Expected SHA " + llvm_sha + " Actual SHA " + sha)
    return sha == llvm_sha


def _configure_llvm_build(
    llvm_src_dir,
    cmake_exec,
    mlir_install_prefix,
    build_type="Release",
    enable_assertions=True,
    c_compiler=None,
    cxx_compiler=None,
):
    try:
        temp_uuid = str(uuid.uuid1())
        llvm_build_dir = llvm_src_dir + "/_build_" + temp_uuid
        os.mkdir(llvm_build_dir)
        llvm_build_dir = os.path.abspath(llvm_build_dir)
    except FileExistsError:
        raise RuntimeError("Could not create build directory.")
    except FileNotFoundError:
        raise RuntimeError(
            f"Could not create build direcotry {llvm_build_dir}", llvm_build_dir,
        )

    if "linux" in sys.platform:
        if c_compiler is None:
            c_compiler = "gcc"
        if cxx_compiler is None:
            cxx_compiler = "g++"
    else:
        raise RuntimeError("Unsupported platform")

    build_system = None
    if sys.platform in ["linux", "win32", "cygwin"]:
        build_system = "Ninja"
    else:
        assert False, sys.platform + " not supported"

    cmake_config_args = cmake_exec + (
        [
            "-G",
            build_system,
            "-DCMAKE_BUILD_TYPE=" + build_type,
            "-DCMAKE_C_COMPILER:PATH=" + c_compiler,
            "-DCMAKE_CXX_COMPILER:PATH=" + cxx_compiler,
            "-DLLVM_ENABLE_PROJECTS=mlir",
            "-DLLVM_BUILD_EXAMPLES=ON",
            #'-DLLVM_TARGETS_TO_BUILD="host"',
            "-DLLVM_ENABLE_ASSERTIONS=" + "ON" if enable_assertions else "OFF",
            "-DLLVM_ENABLE_RTTI=ON",
            "-DLLVM_USE_LINKER=gold",
            "-DCMAKE_INSTALL_PREFIX=" + os.path.abspath(mlir_install_prefix),
            os.path.abspath(llvm_src_dir + "/llvm"),
        ]
    )
    # Configure
    subprocess.check_call(
        cmake_config_args, shell=False, cwd=llvm_build_dir, env=os.environ
    )
    return llvm_build_dir


def _build_llvm(
    llvm_src_dir,
    mlir_install_dir=None,
    build_type="Release",
    enable_assertions=True,
    c_compiler=None,
    cxx_compiler=None,
    cmake_executable=None,
    save_llvm_build_dir=False,
):
    """Builds LLVMs MLIR project

    Args:
        llvm_src_dir (str): Path to an LLVM source directory
        mlir_install_dir (str): Path where LLVM build should be installed
        build_type (str, optional): Type of Cmake build. Defaults to "Release".
        enable_assertions (bool, optional): Whether Cmake should enable
        assertions in the LLVM sources. Defaults to True.
        c_compiler (str, optional): C compiler to be used. Defaults to None.
        cxx_compiler (str, optional): C++ compiler to be used. Defaults to None.
        cmake_executable (str, optional): Cmake to be used. Defaults to None.

    Raises:
        RuntimeError: If the LLVM source directory does not exist.
        RuntimeError: If the build directory could not be created.
        RuntimeError: If running on any system other than Linux.

    Returns:
        str: Path to the LLVM/MLIR installation
    """
    # Check if the LLVM source directory exists
    if not os.path.exists(llvm_src_dir):
        raise RuntimeError(
            f"llvm_src_dir specified as {llvm_src_dir} does not exist.",
            os.path.abspath(llvm_src_dir),
        )
    cmake_exec = [cmake_executable if cmake_executable else "cmake"]
    mlir_install_prefix = (
        mlir_install_dir if mlir_install_dir else llvm_src_dir + "/mlir_install"
    )
    llvm_build_dir = _configure_llvm_build(
        build_type=build_type,
        llvm_src_dir=llvm_src_dir,
        mlir_install_prefix=mlir_install_prefix,
        cmake_exec=cmake_exec,
        enable_assertions=enable_assertions,
        c_compiler=c_compiler,
        cxx_compiler=cxx_compiler,
    )

    try:
        # Build
        cmake_build_args = cmake_exec + ["--build", "."]
        subprocess.check_call(
            cmake_build_args, shell=False, cwd=llvm_build_dir, env=os.environ
        )
        # Install
        cmake_install_args = cmake_exec + ["--install", "."]
        subprocess.check_call(
            cmake_install_args, shell=False, cwd=llvm_build_dir, env=os.environ
        )
    finally:
        if not save_llvm_build_dir:
            print("Removing the llvm build direcotry")
            shutil.rmtree(llvm_build_dir)

    return mlir_install_prefix


def _get_llvm(llvm_sha=None, working_dir=None):
    """Checks out the llvm git repo.

    The llvm git repo is cloned either to a sub-direcotry inside the provided
    "working_dir" or into the system tmp folder. If a commit SHA is provided,
    then the commit is checked out into a new branch. The location of the cloned
    git repo is returned to caller.

    Args:
        working_dir (str, optional): Path where the LLVM repo is to be cloned.
        If None, then defaults to system temp directory.
        Defaults to None.
        llvm_sha (str, optional): LLVM SHA that should be checkedout after
        cloning. Defaults to None.

    Raises:
        RuntimeError: If the working directory does not exist.
        RuntimeError: If the build directory already exists.
        RuntimeError: If the build directory could not be created.
        RuntimeError: If git cloning or checkout failed.

    Returns:
        str: Path to the "llvm-project" sub directory inside the cloned directory.
    """

    if working_dir and not os.path.exists(working_dir):
        raise RuntimeError(
            f"working_dir specified as {working_dir} does not exist.", working_dir,
        )

    git_checkout_dir = working_dir

    if not git_checkout_dir:
        try:
            temp_uuid = str(uuid.uuid1())
            git_checkout_dir = tempfile.gettempdir() + "/_llvm-repo_" + temp_uuid
            os.mkdir(git_checkout_dir)
        except FileExistsError:
            raise RuntimeError("Found existing build directory.")
        except FileNotFoundError:
            raise RuntimeError(
                f"Could not create build direcotry {git_checkout_dir}",
                git_checkout_dir,
            )

    try:
        subprocess.check_call(
            ["git", "clone", "https://github.com/llvm/llvm-project"],
            shell=False,
            env=os.environ,
            cwd=git_checkout_dir,
        )
    except subprocess.CalledProcessError:
        raise RuntimeError("Could not clone llvm-project")

    if llvm_sha:
        try:
            subprocess.check_call(
                ["git", "checkout", llvm_sha.replace("\n", "")],
                shell=False,
                env=os.environ,
                cwd=git_checkout_dir + "/llvm-project",
            )
        except subprocess.CalledProcessError:
            raise RuntimeError(f"Could not checkout the sha: {llvm_sha}", llvm_sha)

    return git_checkout_dir + "/llvm-project"


def _checkout_sha_in_source_dir(llvm_source_dir, llvm_sha):
    """Checks out a specific commit SHA in the LLVM cloned repo. Runs git fetch
    if required.

    Args:
        llvm_source_dir (str): Path to a cloned LLVM repo.
        llvm_sha (str): LLVM commit SHA to be checked out.

    Raises:
        RuntimeError: If the LLVM source directory does not exist.
        RuntimeError: If the git checkout failed.
    """
    if not os.path.exists(llvm_source_dir):
        raise RuntimeError(
            f"llvm-sources specified as {llvm_source_dir} does not exist",
            llvm_source_dir,
        )
    try:
        subprocess.check_call(
            ["git", "fetch", "--all"], shell=False, env=os.environ, cwd=llvm_source_dir,
        )
        subprocess.check_call(
            ["git", "checkout", llvm_sha.replace("\n", "")],
            shell=False,
            env=os.environ,
            cwd=llvm_source_dir,
        )
    except subprocess.CalledProcessError:
        raise RuntimeError(f"Could not checkout the sha: {llvm_sha}", llvm_sha)


def _build_imex(
    build_dir,
    imex_source_dir,
    llvm_install_dir,
    imex_install_dir=None,
    build_type="Release",
    c_compiler=None,
    cxx_compiler=None,
    cmake_executable=None,
    enable_tests=False,
    enable_dpnp=False,
    enable_igpu=False,
    enable_numba_fe=False,
    with_tbb=None,
    enable_numba_hotfix=False,
):
    """Builds Intel MLIR extensions (IMEX).

    Args:
        build_dir (str): Path where to build IMEX.
        imex_source_dir (str): Path to IMEX sources.
        llvm_install_dir (str): path to an LLVM installation that has the MLIR
        project.
        imex_install_dir (str, optional): Path where IMEX is to be installed.
        Defaults to None.
        build_type (str, optional): Cmake build type. Defaults to "Release".
        c_compiler (_type_, optional): Path to C compiler. Defaults to None.
        cxx_compiler (_type_, optional): Path to C++ compiler. Defaults to None.
        cmake_executable (_type_, optional): Path to Cmake. Defaults to None.
        enable_tests (bool, optional): Set to build IMEX unit tests. Defaults to
        False.
        enable_dpnp (bool, optional): Set to build with dpnp calls enabled.
        Defaults to False.
        enable_igpu (bool, optional): Set to build the Intel GPU dialect.
        Defaults to False.
        enable_numba_fe (bool, optional): Set to build the Numba frontend.
        Defaults to False.
        with_tbb (str, optional): Set to path where a TBB cmake config exists.
        Defaults to None.
        enable_numba_hotfix (bool, optional): Set to build with hotfix for Numba.
        Defaults to False.

    Raises:
        RuntimeError: If the LLVM install directory is not found.
        RuntimeError: If a platform other than Linux
    """
    if not os.path.exists(llvm_install_dir):
        raise RuntimeError(
            f"llvm-install specified as {llvm_install_dir} does not exist",
            llvm_install_dir,
        )
    if not _sha_matched(llvm_install_dir, llvm_sha):
        warnings.warn(
            "LLVM installation in "
            + g_llvm_install_dir
            + "has a different SHA than the required sha specified in "
            + "llvm-sha.txt.Rebuild LLVM with require SHA."
        )
    build_system = None
    if sys.platform in ["linux", "win32", "cygwin"]:
        build_system = "Ninja"
    else:
        assert False, sys.platform + " not supported"

    imex_install_prefix = (
        imex_install_dir
        if imex_install_dir
        else os.path.abspath(imex_source_dir) + "/_install"
    )

    if "linux" in sys.platform:
        if c_compiler is None:
            c_compiler = "gcc"
        if cxx_compiler is None:
            cxx_compiler = "g++"
    else:
        raise RuntimeError("Unsupported platform")

    cmake_args = [cmake_executable if cmake_executable else "cmake"]
    cmake_config_args = cmake_args + (
        [
            "-G",
            build_system,
            "-DCMAKE_BUILD_TYPE=" + build_type,
            "-DCMAKE_C_COMPILER:PATH=" + c_compiler,
            "-DCMAKE_CXX_COMPILER:PATH=" + cxx_compiler,
            "-DCMAKE_PREFIX_PATH=" + os.path.abspath(llvm_install_dir),
            "-DCMAKE_INSTALL_PREFIX=" + os.path.abspath(imex_install_prefix),
            "-DIMEX_USE_DPNP=" + "ON" if enable_dpnp else "OFF",
            "-DIMEX_ENABLE_IGPU_DIALECT=" + "ON" if enable_igpu else "OFF",
            "-DIMEX_ENABLE_TESTS=" + "ON" if enable_tests else "OFF",
            "-DIMEX_ENABLE_NUMBA_FE=" + "ON" if enable_numba_fe else "OFF",
            "-DIMEX_ENABLE_NUMBA_HOTFIX=" + "ON" if enable_numba_hotfix else "OFF",
            "-DCMAKE_VERBOSE_MAKEFILE=ON",
            "--debug-trycompile",
            os.path.abspath(imex_source_dir),
        ]
    )

    if with_tbb is not None:
        if os.path.exists(with_tbb):
            cmake_config_args.append("-DTBB_DIR=" + with_tbb)
            cmake_config_args.append("-DIMEX_ENABLE_TBB_SUPPORT=ON")
        else:
            warnings.warn("Provided TBB directory path does not exist.")

    build_dir = os.path.abspath(build_dir)
    # Configure
    subprocess.check_call(cmake_config_args, shell=False, cwd=build_dir, env=os.environ)
    # Build
    cmake_build_args = cmake_args + ["--build", "."]
    subprocess.check_call(cmake_build_args, shell=False, cwd=build_dir, env=os.environ)
    # Install
    cmake_install_args = cmake_args + ["--install", "."]
    subprocess.check_call(
        cmake_install_args, shell=False, cwd=build_dir, env=os.environ
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Driver to build intel-mlir-extensions for in-place installation"
    )
    llvm_builder = parser.add_argument_group(title="LLVM build arguments")
    imex_builder = parser.add_argument_group(title="IMEX build arguments")

    llvm_builder.add_argument(
        "--verbose", action="store_true", help="Turns on CMAKE_MAKE_VERBOSE",
    )
    llvm_builder.add_argument(
        "--c-compiler", type=str, help="Name of C compiler", default=None
    )
    llvm_builder.add_argument(
        "--cxx-compiler", help="Name of C++ compiler", default=None
    )
    llvm_builder.add_argument(
        "--llvm-install",
        help="Path to an existing LLVM installation",
        dest="llvm_install",
        type=str,
    )
    llvm_builder.add_argument(
        "--build-type",
        default="Release",
        type=str,
        dest="build_type",
        help="Set the build mode for LLVM and imex",
    )
    llvm_builder.add_argument(
        "--disable-assertions",
        action="store_false",
        help="Set to disable assertions in LLVM build",
        dest="assertion_setting",
    )
    llvm_builder.add_argument(
        "--llvm-sources",
        help="Path to local LLVM git repo clone",
        dest="llvm_sources",
        type=str,
    )
    llvm_builder.add_argument(
        "--working-dir",
        help="Working directory where LLVM repo is cloned, built, installed",
        dest="working_dir",
        type=str,
        default=None,
    )
    llvm_builder.add_argument(
        "--cmake-executable",
        type=str,
        help="Path to cmake executable",
        dest="cmake_executable",
    )
    llvm_builder.add_argument(
        "--save-llvm-builddir",
        help="Set to True to save the LLVM build directory",
        action="store_true",
        dest="save_llvm_build_dir",
    )

    imex_builder.add_argument(
        "--imex-enable-igpu",
        action="store_true",
        help="Enables building the iGPU dialect.",
    )
    imex_builder.add_argument(
        "--imex-enable-tests",
        action="store_true",
        help="Enables building the ctest unit tests",
    )
    imex_builder.add_argument(
        "--imex-enable-numba",
        action="store_true",
        help="Enables building the Numba front end for IMEX",
    )
    imex_builder.add_argument(
        "--imex-tbb-dir",
        help="A directory where the TBBConfig.cmake or tbb-config.cmake is stored",
        dest="tbb_dir",
        type=str,
        default=None,
    )
    imex_builder.add_argument(
        "--imex-enable-numba-hotfix",
        action="store_true",
        help="Enables building IMEX with Numba hotfix",
    )
    imex_builder.add_argument(
        "--imex-enable-dpnp",
        action="store_true",
        help="Enables building IMEX with dpnp",
    )
    imex_builder.add_argument(
        "--imex-clean-build",
        action="store_true",
        help="Removes any existing build directory when building IMEX",
    )

    args = parser.parse_args()

    # Check if one of llvm_install or llvm_sources is provided.
    # Both flags are optional, but they cannot both be set.
    g_llvm_install_dir = getattr(args, "llvm_install", None)
    g_llvm_source_dir = getattr(args, "llvm_sources", None)
    g_working_dir = getattr(args, "working_dir", None)
    g_tbb_dir = getattr(args, "tbb_dir", None)
    # Get the llvm SHA as hard coded in llvm-sha.txt
    llvm_sha = _get_llvm_sha()

    # If a working directory is provided, check if it was previously used to
    # build IMEX. We check for the "llvm_project" (source directory of llvm) and
    # "_mlir_install" (install directory for llvm)
    if g_working_dir is not None:
        if (
            os.path.exists(g_working_dir + "/llvm-project")
            and g_llvm_source_dir is None
        ):
            g_llvm_source_dir = os.path.abspath(g_working_dir) + "/llvm-project"

        if (
            os.path.exists(g_working_dir + "/_mlir_install")
            and g_llvm_install_dir is None
        ):
            g_llvm_install_dir = os.path.abspath(g_working_dir) + "/_mlir_install"
    elif g_working_dir is None:
        warnings.warn(
            "No working directory specified. Going to use system temp "
            + "directory to build LLVM"
        )
        g_working_dir = tempfile.gettempdir()

    # At this point the working directory cannot be None
    assert g_working_dir is not None

    # If a previously installed LLVM is to be used, then skip LLVM build. We
    # still need to check if the build meets the LLVM_SHA requirement.
    if g_llvm_install_dir:
        if not os.path.exists(g_llvm_install_dir):
            raise RuntimeError(
                f"llvm-install specified as {g_llvm_install_dir} does not exist",
                g_llvm_install_dir,
            )
        if not _sha_matched(g_llvm_install_dir, llvm_sha):
            warnings.warn(
                "LLVM installation in "
                + g_llvm_install_dir
                + "has a different SHA than the required sha specified in "
                + "llvm-sha.txt.Rebuild LLVM with require SHA."
            )
            if g_llvm_source_dir is not None:
                _checkout_sha_in_source_dir(g_llvm_source_dir, llvm_sha)
                g_llvm_install_dir = _build_llvm(
                    build_type=args.build_type,
                    llvm_src_dir=g_llvm_source_dir,
                    mlir_install_dir=g_working_dir + "/_mlir_install",
                    enable_assertions=args.assertion_setting,
                    c_compiler=args.c_compiler,
                    cxx_compiler=args.cxx_compiler,
                    cmake_executable=args.cmake_executable,
                    save_llvm_build_dir=args.save_llvm_build_dir,
                )
            else:
                raise RuntimeError(
                    "Provide an LLVM source location to build and install "
                    + "LLVM with required SHA."
                )
    elif g_llvm_source_dir:
        _checkout_sha_in_source_dir(g_llvm_source_dir, llvm_sha)
        g_llvm_install_dir = _build_llvm(
            build_type=args.build_type,
            llvm_src_dir=g_llvm_source_dir,
            mlir_install_dir=g_working_dir + "/_mlir_install",
            enable_assertions=args.assertion_setting,
            c_compiler=args.c_compiler,
            cxx_compiler=args.cxx_compiler,
            cmake_executable=args.cmake_executable,
            save_llvm_build_dir=args.save_llvm_build_dir,
        )
    else:
        llvm_src = _get_llvm(working_dir=args.working_dir, llvm_sha=llvm_sha)
        g_llvm_install_dir = _build_llvm(
            build_type=args.build_type,
            llvm_src_dir=llvm_src,
            mlir_install_dir=g_working_dir + "/_mlir_install",
            enable_assertions=args.assertion_setting,
            c_compiler=args.c_compiler,
            cxx_compiler=args.cxx_compiler,
            cmake_executable=args.cmake_executable,
            save_llvm_build_dir=args.save_llvm_build_dir,
        )

    # Now we are ready to build IMEX
    setup_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    imex_build_dir = os.path.abspath(setup_dir + "/_build")
    if os.path.exists(imex_build_dir):
        if not args.imex_clean_build:
            raise RuntimeError(
                "IMEX build directory already exists. Use --imex-clean-build "
                + "to remove build directory"
            )
        else:
            shutil.rmtree(imex_build_dir)
    os.mkdir(imex_build_dir)
    _build_imex(
        build_dir=imex_build_dir,
        imex_source_dir=setup_dir,
        llvm_install_dir=g_llvm_install_dir,
        enable_dpnp=args.imex_enable_dpnp,
        enable_numba_fe=args.imex_enable_numba,
        enable_numba_hotfix=args.imex_enable_numba_hotfix,
        enable_igpu=args.imex_enable_igpu,
        with_tbb=g_tbb_dir,
        enable_tests=args.imex_enable_tests,
    )

    # TODO
    # Export out llvm-build-dir, source-dir, other configs to a YAML file
    # save the file both in current dir and working dir

    # Add a default/auto option where if a config file is present in __file__
    # location just validate for sha and rebuild.
