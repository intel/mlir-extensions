# ===-- _site_initialize_0.py - For site init -----------------*- Python -*-===#
#
# Copyright 2025 Intel Corporation
# Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===-----------------------------------------------------------------------===#


def context_init_hook(context):
    # this will invoke imex binding registrations
    from imex_mlir.dialects.region import EnvironmentRegionOp  # noqa
