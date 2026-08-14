FROM ubuntu:noble

ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV="/opt/venv"
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

ARG SYSTEM_DEPS="software-properties-common ca-certificates locales net-tools unzip"
ARG DEVEL_DEPS="pkg-config build-essential cmake gdb git git-lfs curl wget clangd"
ARG PICO_DEPS="gcc-arm-none-eabi binutils-arm-none-eabi libnewlib-arm-none-eabi libstdc++-arm-none-eabi-newlib gdb-multiarch"
ARG CPP_DEPS="libopencv-dev nlohmann-json3-dev libgtest-dev"
ARG PYTHON_DEPS="python3 python3-pip python3-venv"

WORKDIR /tmp

RUN apt-get update && \
    apt-get install -y --no-install-recommends ${SYSTEM_DEPS} && \
    add-apt-repository universe && \
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y --no-install-recommends \
        ${DEVEL_DEPS} \
        ${PICO_DEPS} \
        ${CPP_DEPS} \
        ${PYTHON_DEPS} \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY env/requirements.txt .
RUN python3 -m venv /opt/venv && \
    ln -sf /opt/venv/bin/python3 /opt/venv/bin/python && \
    /opt/venv/bin/pip install --no-cache-dir -U pip wheel setuptools && \
    /opt/venv/bin/pip install --no-cache-dir -Ur requirements.txt