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
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/llvm-project"
    echo "  $0 -t 'load_nd.*f16' /path/to/llvm-project"
    echo "  $0 --test 'transpose' /path/to/llvm-project /path/to/imex"
}

# Parse command-line options
TEST_NAME_FILTER=""
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--test)
            TEST_NAME_FILTER="$2"
            shift 2
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
else
    # Pre-built LLVM: find source repository (typically parent of build directory)
    print_info "Searching for LLVM source repository..."
    LLVM_SOURCE_DIR=""

    # Check parent directory for LLVM source
    PARENT_DIR="$(dirname "$LLVM_PROJECT_PATH")"
    if [ -d "$PARENT_DIR/.git" ] && [ -d "$PARENT_DIR/mlir" ]; then
        LLVM_SOURCE_DIR="$PARENT_DIR"
        print_info "Found LLVM source at: $LLVM_SOURCE_DIR"
    fi

    if [ -z "$LLVM_SOURCE_DIR" ]; then
        print_error "Cannot find LLVM source repository to extract SHA"
        print_error "Pre-built LLVM path: $LLVM_PROJECT_PATH"
        print_error "Tried parent directory: $PARENT_DIR"
        print_error "Please provide path to LLVM source repository instead of build directory"
        exit 1
    fi

    cd "$LLVM_SOURCE_DIR"
    LLVM_HEAD_SHA=$(git rev-parse HEAD)
    print_info "LLVM source repository SHA: $LLVM_HEAD_SHA"
fi

cd "$IMEX_PROJECT_PATH"
echo "$LLVM_HEAD_SHA" > "$LLVM_VERSION_FILE"
print_success "Updated llvm_version.txt with SHA: $LLVM_HEAD_SHA"

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
    BUILD_DIR="$IMEX_PROJECT_PATH/build"
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
    print_info "Disabling: IMEX_BUILD_VC_CONVERSIONS (ArithToVC, MathToVC, XeGPUToVC)"
    print_info "Disabling: IMEX_ENABLE_XEGPU_LAYOUT_PASSES (MaterializeMatrixOp, OptimizeTranspose)"

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
        -DIMEX_ENABLE_XEGPU_LAYOUT_PASSES=OFF \
        -DLLVM_LIT_ARGS="-v --filter='$LIT_FILTER'"
else
    # Option 3: Out-of-Tree Build with Pre-built LLVM
    print_info "Build type: IMEX Out-of-Tree with Pre-built LLVM"
    print_info "MLIR CMake Directory: $MLIR_CMAKE_DIR"
    print_info "IMEX Source Directory: $IMEX_PROJECT_PATH"
    print_info "Enabling: IMEX_ENABLE_L0_RUNTIME"
    print_info "Disabling: IMEX_BUILD_VC_CONVERSIONS (ArithToVC, MathToVC, XeGPUToVC)"
    print_info "Disabling: IMEX_ENABLE_XEGPU_LAYOUT_PASSES (MaterializeMatrixOp, OptimizeTranspose)"

    # Build lit filter pattern
    if [ -n "$TEST_NAME_FILTER" ]; then
        # User provided a specific test filter
        LIT_FILTER="$TEST_NAME_FILTER"
        print_info "Using custom test filter: $LIT_FILTER"
        LLVM_LIT_ARGS_OPTION="-DLLVM_LIT_ARGS=-v --filter='$LIT_FILTER'"
    else
        # No filter for out-of-tree build by default
        LLVM_LIT_ARGS_OPTION=""
    fi

    cmake -S . -B "$BUILD_DIR" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DMLIR_DIR="$MLIR_CMAKE_DIR" \
        -DIMEX_ENABLE_L0_RUNTIME=1 \
        -DIMEX_BUILD_VC_CONVERSIONS=OFF \
        -DIMEX_ENABLE_XEGPU_LAYOUT_PASSES=OFF \
        $LLVM_LIT_ARGS_OPTION
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
    TEST_EXIT_CODE=$?
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
    TEST_EXIT_CODE=$?
    set -e
fi

if [ $TEST_EXIT_CODE -eq 0 ]; then
    print_success "All tests passed!"
else
    print_warning "Some tests failed (exit code: $TEST_EXIT_CODE)"
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
echo -e "${GREEN}VC Conversions:${NC} Disabled (IMEX_BUILD_VC_CONVERSIONS=OFF)"
echo -e "${GREEN}XeGPU Layout Passes:${NC} Disabled (IMEX_ENABLE_XEGPU_LAYOUT_PASSES=OFF)"
if [ -n "$TEST_NAME_FILTER" ]; then
    echo -e "${GREEN}Test Filter:${NC} $TEST_NAME_FILTER"
else
    echo -e "${GREEN}Test Filter:${NC} Default (all XeGPU integration tests)"
fi
echo -e "${GREEN}Test Exit Code:${NC} $TEST_EXIT_CODE"

print_success "Script completed successfully!"

exit $TEST_EXIT_CODE
