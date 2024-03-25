FROM summerwind/actions-runner:ubuntu-22.04

SHELL ["/bin/bash", "-ex", "-o", "pipefail", "-c"]

USER root

RUN export DEBIAN_FRONTEND=noninteractive \
    && curl -s https://repositories.intel.com/graphics/intel-graphics.key \
        | gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg \
    && echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy max' \
        | tee /etc/apt/sources.list.d/intel-gpu.list \
    && curl https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
        | gpg --dearmor --output /usr/share/keyrings/oneapi-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
        | tee /etc/apt/sources.list.d/oneAPI.list \
    && apt-get update --yes \
    && apt-get install build-essential cmake ninja-build intel-opencl-icd intel-level-zero-gpu clinfo intel-basekit --yes \
    && rm -rf /var/lib/apt/lists/*

USER runner
