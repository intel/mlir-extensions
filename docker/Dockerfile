FROM summerwind/actions-runner:ubuntu-22.04

RUN set -ex \
    && export DEBIAN_FRONTEND=noninteractive \
    && curl -s https://repositories.intel.com/graphics/intel-graphics.key \
        | sudo gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg \
    && echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu jammy max' \
        | sudo tee /etc/apt/sources.list.d/intel-graphics.list \
    && curl https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
        | sudo gpg --dearmor --output /usr/share/keyrings/oneapi-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
        | sudo tee /etc/apt/sources.list.d/oneAPI.list \
    && sudo apt-get update --yes \
    && sudo apt-get install intel-opencl-icd clinfo intel-basekit --yes \
    && sudo rm -rf /var/lib/apt/lists/*
