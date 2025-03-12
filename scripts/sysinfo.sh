#!/bin/sh

dpkg -l clang cmake gcc gh \
        python3 \
        opencl-c-headers ocl-icd-opencl-dev intel-level-zero-gpu intel-igc-core intel-opencl-icd || true
