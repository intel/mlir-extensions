<!--
SPDX-FileCopyrightText: 2022 Intel Corporation

SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# IMEX Integration Example

Example showing how to use IMEX as an external library from another CMake based project.

## Prestep for building

Build and install IMEX

## Building sample-opt
```
cmake -B build -S . -GNinja -DIMEX_DIR=<path to directory with IMEXConfig.cmake> -DMLIR_DIR=<path to directory with MLIRConfig.cmake>
cmake --build build
```
You can find executable sample-opt at build/bin/sample-opt

## License
This code is made available under the Apache License 2.0 with LLVM Exceptions.
See the `LICENSE.txt` file for more details.
