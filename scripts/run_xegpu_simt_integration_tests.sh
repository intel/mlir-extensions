#!/bin/bash

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to print section headers
print_section() {
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}========================================${NC}\n"
}

# Function to print usage
print_usage() {
    print_error "Usage: $0 [options] <llvm-project-path> [imex-project-path]"
    echo ""
    echo "Arguments:"
    echo "  <llvm-project-path>    Path to llvm-project repository or pre-built installation"
    echo "  [imex-project-path]    Optional path to IMEX repository"
    echo "                         (default: current directory if it's IMEX root, or parent of script directory)"
    echo ""
    echo "Options:"
    echo "  -t, --test <pattern>   Test name pattern (regex) to pass to LIT --filter"
    echo "                         Example: -t 'load_nd.*f16' or -t 'transpose'"
    echo "  --upstream-tests       Run upstream MLIR XeGPU integration tests (LANE/SG/WG)"
    echo "                         directly via mlir-opt/mlir-runner (default: enabled)"
    echo "  --no-upstream-tests    Skip the upstream MLIR XeGPU integration tests"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/llvm-project"
    echo "  $0 -t 'load_nd.*f16' /path/to/llvm-project"
    echo "  $0 --test 'transpose' /path/to/llvm-project /path/to/imex"
    echo "  $0 --no-upstream-tests /path/to/llvm-project"
}

# Parse command-line options
TEST_NAME_FILTER=""
RUN_UPSTREAM_TESTS=true
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--test)
            TEST_NAME_FILTER="$2"
            shift 2
            ;;
        --upstream-tests)
            RUN_UPSTREAM_TESTS=true
            shift
            ;;
        --no-upstream-tests)
            RUN_UPSTREAM_TESTS=false
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        -*)
            print_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Restore positional parameters
set -- "${POSITIONAL_ARGS[@]}"

# Check if correct number of arguments provided
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    print_usage
    exit 1
fi

LLVM_PROJECT_PATH="$1"

# Determine IMEX project path
if [ "$#" -eq 2 ]; then
    # User provided IMEX path
    IMEX_PROJECT_PATH="$2"
else
    # Try current directory first, then script location (parent of scripts/ folder)
    if [ -f "$(pwd)/build_tools/llvm_version.txt" ] && [ -d "$(pwd)/test/Integration/Dialect/XeGPU" ]; then
        IMEX_PROJECT_PATH="$(pwd)"
    else
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        # Script is in scripts/ subfolder, so go up one level to get IMEX root
        IMEX_PROJECT_PATH="$(dirname "$SCRIPT_DIR")"
    fi
fi

print_section "Starting XeGPU Integration Tests Setup"

# Validate IMEX project path
print_info "Validating IMEX project path: $IMEX_PROJECT_PATH"
if [ ! -d "$IMEX_PROJECT_PATH" ]; then
    print_error "IMEX project path does not exist: $IMEX_PROJECT_PATH"
    exit 1
fi

if [ ! -d "$IMEX_PROJECT_PATH/.git" ]; then
    print_error "IMEX project is not a git repository: $IMEX_PROJECT_PATH"
    exit 1
fi

if [ ! -d "$IMEX_PROJECT_PATH/test/Integration/Dialect/XeGPU" ]; then
    print_error "Not a valid IMEX project (test/Integration/Dialect/XeGPU not found): $IMEX_PROJECT_PATH"
    exit 1
fi

if [ ! -f "$IMEX_PROJECT_PATH/build_tools/llvm_version.txt" ]; then
    print_error "Not a valid IMEX project (build_tools/llvm_version.txt not found): $IMEX_PROJECT_PATH"
    exit 1
fi

print_success "IMEX project path validated: $IMEX_PROJECT_PATH"

# Validate LLVM project path and detect type (source repo vs pre-built)
print_info "Validating LLVM project path: $LLVM_PROJECT_PATH"
if [ ! -d "$LLVM_PROJECT_PATH" ]; then
    print_error "LLVM project path does not exist: $LLVM_PROJECT_PATH"
    exit 1
fi

# Detect if this is a source repository or pre-built LLVM
USE_PREBUILT_LLVM=false
MLIR_CMAKE_DIR=""

if [ -d "$LLVM_PROJECT_PATH/.git" ]; then
    # This is a source repository
    print_info "Detected LLVM source repository"
    if [ ! -d "$LLVM_PROJECT_PATH/mlir" ]; then
        print_error "Not a valid LLVM project (mlir directory not found): $LLVM_PROJECT_PATH"
        exit 1
    fi
    USE_PREBUILT_LLVM=false
elif [ -f "$LLVM_PROJECT_PATH/lib/cmake/mlir/MLIRConfig.cmake" ]; then
    # This is a pre-built LLVM installation
    print_info "Detected pre-built LLVM installation"
    MLIR_CMAKE_DIR="$LLVM_PROJECT_PATH/lib/cmake/mlir"
    USE_PREBUILT_LLVM=true
    print_info "MLIRConfig.cmake found at: $MLIR_CMAKE_DIR/MLIRConfig.cmake"
else
    print_error "LLVM path is neither a source repository (.git) nor a pre-built installation (lib/cmake/mlir/MLIRConfig.cmake)"
    print_error "Please provide either:"
    print_error "  - Path to LLVM source repository (with .git)"
    print_error "  - Path to pre-built LLVM installation (with lib/cmake/mlir/MLIRConfig.cmake)"
    exit 1
fi

print_success "LLVM project path is valid ($([ "$USE_PREBUILT_LLVM" = true ] && echo "pre-built" || echo "source repository"))"

# Get IMEX commit information
cd "$IMEX_PROJECT_PATH"
IMEX_COMMIT=$(git rev-parse HEAD)
IMEX_BRANCH=$(git rev-parse --abbrev-ref HEAD)
print_info "IMEX branch: $IMEX_BRANCH"
print_info "IMEX commit: $IMEX_COMMIT"

# Update llvm_version.txt
print_section "Updating llvm_version.txt"

LLVM_VERSION_FILE="$IMEX_PROJECT_PATH/build_tools/llvm_version.txt"
OLD_LLVM_SHA=$(cat "$LLVM_VERSION_FILE" | head -n 1)
print_info "Old LLVM SHA in llvm_version.txt: $OLD_LLVM_SHA"

if [ "$USE_PREBUILT_LLVM" = false ]; then
    # Source repository: use git SHA directly
    cd "$LLVM_PROJECT_PATH"
    LLVM_HEAD_SHA=$(git rev-parse HEAD)
    print_info "LLVM source repository SHA: $LLVM_HEAD_SHA"
    cd "$IMEX_PROJECT_PATH"
    echo "$LLVM_HEAD_SHA" > "$LLVM_VERSION_FILE"
    print_success "Updated llvm_version.txt with SHA: $LLVM_HEAD_SHA"
fi

# Configure build (different approach for source repo vs pre-built)
if [ "$USE_PREBUILT_LLVM" = false ]; then
    print_section "Configuring LLVM Build with IMEX as External Project"
else
    print_section "Configuring IMEX Out-of-Tree Build with Pre-built LLVM"
fi

if [ "$USE_PREBUILT_LLVM" = false ]; then
    cd "$LLVM_PROJECT_PATH"
    BUILD_DIR="$LLVM_PROJECT_PATH/build_integration_xegpu"
else
    cd "$IMEX_PROJECT_PATH"
    BUILD_DIR="$IMEX_PROJECT_PATH/build_simt_imex"
fi

if [ -d "$BUILD_DIR" ]; then
    print_warning "Build directory already exists: $BUILD_DIR"
    echo ""
    read -p "Do you want to remove it and create a fresh build? (y/n): " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Removing existing build directory..."
        rm -rf "$BUILD_DIR"
        print_success "Build directory removed"
        mkdir -p "$BUILD_DIR"
    else
        print_info "Keeping existing build directory (will reconfigure)"
    fi
else
    print_info "Build directory does not exist. Creating new build..."
    mkdir -p "$BUILD_DIR"
fi

print_info "Running CMake configuration..."
print_info "Build directory: $BUILD_DIR"

if [ "$USE_PREBUILT_LLVM" = false ]; then
    # Option 1: LLVM External Project Build
    print_info "Build type: IMEX as LLVM External Project"
    print_info "LLVM External Projects: Imex"
    print_info "IMEX Source Directory: $IMEX_PROJECT_PATH"
    print_info "Enabling: MLIR_INCLUDE_INTEGRATION_TESTS, MLIR_ENABLE_LEVELZERO_RUNNER, MLIR_ENABLE_SYCL_RUNNER, IMEX_ENABLE_L0_RUNTIME"
    print_info "Disabling: IMEX_BUILD_VC_CONVERSIONS (obsolete VC backend: ArithToVC, MathToVC, XeGPUToVC, RemoveSingleElemVector, XeGPU layout passes)"

    # Build lit filter pattern for the specific test directories
    if [ -n "$TEST_NAME_FILTER" ]; then
        # User provided a specific test filter
        LIT_FILTER="$TEST_NAME_FILTER"
        print_info "Using custom test filter: $LIT_FILTER"
    else
        # Default: all XeGPU integration test directories
        LIT_FILTER="Integration/Dialect/XeGPU/SG|Integration/Dialect/XeGPU/WG|Integration/Dialect/XeGPU/SIMT|Integration/Dialect/XeVM"
        print_info "Using default test filter for XeGPU integration tests"
    fi

    cmake -S llvm -B "$BUILD_DIR" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_ENABLE_PROJECTS="mlir" \
        -DLLVM_TARGETS_TO_BUILD="X86;SPIRV" \
        -DLLVM_EXTERNAL_PROJECTS="Imex" \
        -DLLVM_EXTERNAL_IMEX_SOURCE_DIR="$IMEX_PROJECT_PATH" \
        -DMLIR_INCLUDE_INTEGRATION_TESTS=ON \
        -DMLIR_ENABLE_LEVELZERO_RUNNER=1 \
        -DMLIR_ENABLE_SYCL_RUNNER=1 \
        -DIMEX_ENABLE_L0_RUNTIME=1 \
        -DIMEX_BUILD_VC_CONVERSIONS=OFF \
        -DLLVM_LIT_ARGS="-v --filter='$LIT_FILTER'"
else
    # Option 3: Out-of-Tree Build with Pre-built LLVM
    print_info "Build type: IMEX Out-of-Tree with Pre-built LLVM"
    print_info "MLIR CMake Directory: $MLIR_CMAKE_DIR"
    print_info "IMEX Source Directory: $IMEX_PROJECT_PATH"
    print_info "Enabling: IMEX_ENABLE_L0_RUNTIME"
    print_info "Disabling: IMEX_BUILD_VC_CONVERSIONS (obsolete VC backend: ArithToVC, MathToVC, XeGPUToVC, RemoveSingleElemVector, XeGPU layout passes)"

    # Build lit filter pattern
    if [ -n "$TEST_NAME_FILTER" ]; then
        # User provided a specific test filter
        LIT_FILTER="$TEST_NAME_FILTER"
        print_info "Using custom test filter: $LIT_FILTER"
    else
        # Default: all XeGPU integration test directories (same as source build)
        LIT_FILTER="Integration/Dialect/XeGPU/SG|Integration/Dialect/XeGPU/WG|Integration/Dialect/XeGPU/SIMT|Integration/Dialect/XeVM"
        print_info "Using default test filter for XeGPU integration tests"
    fi

    cmake -S . -B "$BUILD_DIR" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DMLIR_DIR="$MLIR_CMAKE_DIR" \
        -DIMEX_ENABLE_L0_RUNTIME=1 \
        -DMLIR_ENABLE_LEVELZERO_RUNNER=1 \
        -DMLIR_ENABLE_SYCL_RUNNER=1 \
        -DMLIR_SPIRV_BACKEND_ENABLED=1 \
        -DIMEX_CHECK_LLVM_VERSION=OFF \
        -DIMEX_BUILD_VC_CONVERSIONS=OFF \
        -DLLVM_LIT_ARGS="-v --filter='$LIT_FILTER'"
fi

if [ $? -eq 0 ]; then
    print_success "CMake configuration completed successfully"
else
    print_error "CMake configuration failed"
    exit 1
fi

# Build IMEX
if [ "$USE_PREBUILT_LLVM" = false ]; then
    print_section "Building LLVM/MLIR with IMEX"
else
    print_section "Building IMEX"
fi

print_info "Starting build process (this may take a while)..."
if [ "$USE_PREBUILT_LLVM" = false ]; then
    print_info "Building all targets first to ensure dependencies are ready..."
    ninja -C "$BUILD_DIR"
else
    print_info "Building IMEX with pre-built LLVM..."
    cmake --build "$BUILD_DIR"
fi

if [ $? -eq 0 ]; then
    print_success "Build completed successfully"
else
    print_error "Build failed"
    exit 1
fi

# Run IMEX tests
print_section "Running IMEX XeGPU Integration Tests"

if [ "$USE_PREBUILT_LLVM" = false ]; then
    print_info "Running check-imex target with filtered tests"
    if [ -n "$TEST_NAME_FILTER" ]; then
        print_info "Test filter: $TEST_NAME_FILTER"
    else
        print_info "Test directories:"
        print_info "  - Integration/Dialect/XeGPU/SG"
        print_info "  - Integration/Dialect/XeGPU/WG"
        print_info "  - Integration/Dialect/XeGPU/SIMT"
        print_info "  - Integration/Dialect/XeVM"
    fi
    echo ""

    # Run tests and capture exit code, but don't stop on failure
    set +e
    ninja -C "$BUILD_DIR" check-imex
    IMEX_TEST_EXIT_CODE=$?
    set -e
else
    print_info "Running check-imex target for out-of-tree build"
    if [ -n "$TEST_NAME_FILTER" ]; then
        print_info "Test filter: $TEST_NAME_FILTER"
    fi
    echo ""

    # Run tests and capture exit code, but don't stop on failure
    set +e
    cmake --build "$BUILD_DIR" --target check-imex
    IMEX_TEST_EXIT_CODE=$?
    set -e
fi

# Run upstream MLIR XeGPU integration tests directly (bypassing lit).
# We parse each test's `// RUN:` header, substitute known lit variables,
# and execute the pipeline via the freshly-built tools. Tests whose RUN
# line contains `zebin-chip=cri` are compile-only (mlir-opt with
# binary-format=isa) - the ocloc step is skipped since environments
# outside Intel-internal toolchains typically do not recognise `cri`.
# Tests carrying `// XFAIL:` are skipped rather than reported as failures.
#
# This stage requires an LLVM *source tree* (the tests live under
# mlir/test/Integration/Dialect/XeGPU). It therefore only runs when the caller
# passed an LLVM source repository; a pre-built LLVM installation has no source
# tree and is skipped.
if [ "$RUN_UPSTREAM_TESTS" != true ]; then
    print_info "Skipping upstream MLIR XeGPU integration tests (--no-upstream-tests)"
    UPSTREAM_TEST_EXIT_CODE=0
elif [ "$USE_PREBUILT_LLVM" = true ]; then
    # The upstream tests live in the LLVM source tree (mlir/test/Integration/...).
    # A pre-built build dir has no source tree, so there is nothing to run.
    print_info "Skipping upstream MLIR XeGPU integration tests (pre-built LLVM: no source tree available)"
    UPSTREAM_TEST_EXIT_CODE=0
else
    UPSTREAM_TOOLS_DIR="$BUILD_DIR"
    UPSTREAM_XEGPU_TESTS_DIR="$LLVM_PROJECT_PATH/mlir/test/Integration/Dialect/XeGPU"
fi

if [ "$RUN_UPSTREAM_TESTS" != true ] || [ "$USE_PREBUILT_LLVM" = true ]; then
    : # already handled above; UPSTREAM_TEST_EXIT_CODE set to 0
elif [ -z "$UPSTREAM_XEGPU_TESTS_DIR" ] || [ ! -d "$UPSTREAM_XEGPU_TESTS_DIR" ]; then
    print_warning "Upstream MLIR XeGPU tests directory not found (skipped)"
    UPSTREAM_TEST_EXIT_CODE=0
elif [ ! -x "$UPSTREAM_TOOLS_DIR/bin/mlir-opt" ]; then
    print_warning "mlir-opt not found at $UPSTREAM_TOOLS_DIR/bin/mlir-opt (skipped)"
    UPSTREAM_TEST_EXIT_CODE=0
else
    print_section "Running upstream MLIR XeGPU Integration Tests"
    print_info "Tests dir: $UPSTREAM_XEGPU_TESTS_DIR"
    print_info "Tools dir: $UPSTREAM_TOOLS_DIR"
    print_info "Note: tests with zebin-chip=cri are compile-only (skip mlir-runner)."
    print_info "Note: tests marked '// XFAIL:' are skipped."
    echo ""

    set +e
    UPSTREAM_TOOLS_DIR="$UPSTREAM_TOOLS_DIR" \
    UPSTREAM_XEGPU_TESTS_DIR="$UPSTREAM_XEGPU_TESTS_DIR" \
    python3 - <<'PYEOF'
import os
import re
import subprocess
import sys
from pathlib import Path

RUN_RE = re.compile(r"^\s*//\s*RUN:\s*(.*)$")
XFAIL_RE = re.compile(r"^\s*//\s*XFAIL:\s*(.*)$")


def collect_run_command(text):
    parts = []
    for raw in text.splitlines():
        m = RUN_RE.match(raw)
        if not m:
            if parts:
                break
            continue
        chunk = m.group(1).rstrip()
        if chunk.endswith("\\"):
            parts.append(chunk[:-1].strip())
            continue
        parts.append(chunk)
        break
    return " ".join(parts)


def substitute(cmd, subs):
    for key in sorted(subs, key=len, reverse=True):
        cmd = cmd.replace(f"%{key}", subs[key])
    return cmd


def build_cmd(run_cmd):
    if "zebin-chip=cri" in run_cmd:
        compile_only = run_cmd.split("|", 1)[0].strip()
        if "binary-format" not in compile_only:
            compile_only = compile_only.replace(
                "zebin-chip=cri", "zebin-chip=cri binary-format=isa", 1
            )
        return compile_only, "compile-only (cri)"
    return run_cmd, "full (opt+runner+FileCheck)"


tools = Path(os.environ["UPSTREAM_TOOLS_DIR"]).resolve()
tests_dir = Path(os.environ["UPSTREAM_XEGPU_TESTS_DIR"]).resolve()
bindir = tools / "bin"
libdir = tools / "lib"

for tool in ("mlir-opt", "mlir-runner", "FileCheck"):
    if not (bindir / tool).is_file():
        print(f"error: {bindir / tool} not found", file=sys.stderr)
        sys.exit(2)

subs = {
    "mlir_runner_utils": str(libdir / "libmlir_runner_utils.so"),
    "mlir_c_runner_utils": str(libdir / "libmlir_c_runner_utils.so"),
    "mlir_levelzero_runtime": str(libdir / "libmlir_levelzero_runtime.so"),
    "mlir_async_runtime": str(libdir / "libmlir_async_runtime.so"),
    "mlir_float16_utils": str(libdir / "libmlir_float16_utils.so"),
}

env = os.environ.copy()
env["PATH"] = f"{bindir}{os.pathsep}{env.get('PATH', '')}"

tests = []
for sub in ("LANE", "SG", "WG"):
    d = tests_dir / sub
    if d.is_dir():
        tests.extend(sorted(d.glob("*.mlir")))

if not tests:
    print(f"error: no tests found under {tests_dir}", file=sys.stderr)
    sys.exit(2)

passed, failed, skipped = [], [], []
for t in tests:
    rel = t.relative_to(tests_dir)
    text = t.read_text()

    xfail_line = next(
        (m.group(1).strip() for m in (XFAIL_RE.match(l) for l in text.splitlines()) if m),
        None,
    )
    if xfail_line is not None:
        skipped.append((t, f"XFAIL: {xfail_line}"))
        print(f"SKIP  {rel}  (XFAIL: {xfail_line})")
        continue

    run_cmd = collect_run_command(text)
    if not run_cmd:
        skipped.append((t, "no RUN line"))
        print(f"SKIP  {rel}  (no RUN line)")
        continue

    subs["s"] = str(t)
    cmd, label = build_cmd(substitute(run_cmd, subs))

    print(f"RUN   {rel}  [{label}]")
    proc = subprocess.run(
        cmd, shell=True, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    out = proc.stdout.decode("utf-8", errors="replace")
    if proc.returncode == 0:
        passed.append(t)
        print(f"PASS  {rel}")
    else:
        failed.append((t, proc.returncode, out))
        print(f"FAIL  {rel}  (exit {proc.returncode})")

print()
print("=" * 60)
print(f"Total:   {len(tests)}")
print(f"Passed:  {len(passed)}")
print(f"Failed:  {len(failed)}")
print(f"Skipped: {len(skipped)}")
print("=" * 60)

for t, rc, out in failed:
    rel = t.relative_to(tests_dir)
    print(f"\n--- FAIL: {rel} (exit {rc}) ---")
    print("\n".join(out.splitlines()[-40:]))

sys.exit(1 if failed else 0)
PYEOF
    UPSTREAM_TEST_EXIT_CODE=$?
    set -e
fi

TEST_EXIT_CODE=$(( IMEX_TEST_EXIT_CODE | UPSTREAM_TEST_EXIT_CODE ))

if [ $TEST_EXIT_CODE -eq 0 ]; then
    print_success "All tests passed!"
else
    print_warning "Some tests failed (IMEX exit: $IMEX_TEST_EXIT_CODE, upstream exit: $UPSTREAM_TEST_EXIT_CODE)"
    print_info "Continuing to cleanup section..."
fi

# Final summary
print_section "Summary"
echo -e "${GREEN}LLVM Project:${NC} $LLVM_PROJECT_PATH"
if [ "$USE_PREBUILT_LLVM" = false ]; then
    echo -e "${GREEN}LLVM Type:${NC} Source Repository"
else
    echo -e "${GREEN}LLVM Type:${NC} Pre-built Installation"
    echo -e "${GREEN}MLIR CMake Dir:${NC} $MLIR_CMAKE_DIR"
fi
echo -e "${GREEN}LLVM Version:${NC} $LLVM_HEAD_SHA"
echo -e "${GREEN}IMEX Project:${NC} $IMEX_PROJECT_PATH"
echo -e "${GREEN}IMEX Branch:${NC} $IMEX_BRANCH"
echo -e "${GREEN}IMEX Commit:${NC} $IMEX_COMMIT"
echo -e "${GREEN}Build Directory:${NC} $BUILD_DIR"
if [ "$USE_PREBUILT_LLVM" = false ]; then
    echo -e "${GREEN}Build Type:${NC} IMEX as LLVM External Project"
else
    echo -e "${GREEN}Build Type:${NC} IMEX Out-of-Tree Build"
fi
echo -e "${GREEN}Obsolete VC Backend:${NC} Disabled (IMEX_BUILD_VC_CONVERSIONS=OFF)"
echo -e "${GREEN}Upstream Tests:${NC} $([ "$RUN_UPSTREAM_TESTS" = true ] && echo "Enabled" || echo "Disabled (--no-upstream-tests)")"
if [ -n "$TEST_NAME_FILTER" ]; then
    echo -e "${GREEN}Test Filter:${NC} $TEST_NAME_FILTER"
else
    echo -e "${GREEN}Test Filter:${NC} Default (all XeGPU integration tests)"
fi
echo -e "${GREEN}IMEX Test Exit Code:${NC} $IMEX_TEST_EXIT_CODE"
echo -e "${GREEN}Upstream Test Exit Code:${NC} $UPSTREAM_TEST_EXIT_CODE"
echo -e "${GREEN}Combined Test Exit Code:${NC} $TEST_EXIT_CODE"

print_success "Script completed successfully!"

exit $TEST_EXIT_CODE
