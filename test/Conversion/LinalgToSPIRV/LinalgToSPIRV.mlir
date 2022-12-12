// RUN: imex-opt --split-input-file --convert-linalg-to-spirv %s -verify-diagnostics -o -| FileCheck %s
