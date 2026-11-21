#!/bin/sh
#
# Install a CMake new enough to configure upstream LLVM.
#
# LLVM raised its minimum required CMake version from 3.20 to 3.31 in
# llvm/llvm-project@eb9bb7d826f8 ("[CMake] Bump the minimum required version to
# 3.31"), which is newer than the CMake shipped by the distributions on the CI
# runners. This script downloads an official Kitware binary release and unpacks
# it into the given directory, so that <dir>/bin can be prepended to PATH.
#
# If the CMake already on PATH is new enough, nothing is downloaded.
#
# Usage: scripts/install_cmake.sh [install-dir]   (default: ./cmake)

set -e

CMAKE_VERSION=3.31.10
CMAKE_SHA256=3cb3dd247b6a1de2d0f4b20c6fd4326c9024e894cebc9dc8699758887e566ca7
CMAKE_MINIMUM=3.31.0

DEST=${1:-$PWD/cmake}

version_at_least() {
    # version_at_least <have> <want> -- true when <have> >= <want>
    [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -1)" = "$2" ]
}

if command -v cmake > /dev/null 2>&1; then
    have=$(cmake --version | head -1 | awk '{print $3}')
    if version_at_least "$have" "$CMAKE_MINIMUM"; then
        echo "CMake $have is already installed, skipping download"
        exit 0
    fi
    echo "CMake $have is older than $CMAKE_MINIMUM"
fi

tarball=cmake-$CMAKE_VERSION-linux-x86_64.tar.gz
url=https://github.com/Kitware/CMake/releases/download/v$CMAKE_VERSION/$tarball

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

echo "Downloading $url"
curl -sSLo "$tmpdir/$tarball" "$url"
echo "$CMAKE_SHA256  $tmpdir/$tarball" | sha256sum -c -

mkdir -p "$DEST"
tar -xzf "$tmpdir/$tarball" -C "$DEST" --strip-components=1

"$DEST/bin/cmake" --version
echo "Installed CMake $CMAKE_VERSION into $DEST"
