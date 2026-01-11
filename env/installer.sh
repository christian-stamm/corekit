#!/bin/bash
SYS_DEPS="\
    software-properties-common \
    lsb-release \
    pkg-config \
    gnupg \
    git \
    zip \
    wget \
    curl \
"

CPP_DEPS="\
    build-essential \
    cmake \
    clangd \
    ninja-build \
    gcc \
    g++ \
    gdb-multiarch \
    libopencv-dev \
    nlohmann-json3-dev \
    libzstd-dev \
    liblz4-dev \
    libglm-dev \
    libeigen3-dev \
    libfdt-dev \
    libglfw3-dev \
    mesa-utils \
    vulkan-tools \
    libgtest-dev \
"

PYTHON_DEPS="\
    python3 \
    python3-pip \
    python3-venv \
"

sudo apt-get update
sudo apt-get install -y $SYS_DEPS

# Add the Ubuntu Toolchain PPA for newer GCC versions
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test

# Add Kitware repository for latest CMake
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc \
    | gpg --dearmor | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" \
    | sudo tee /etc/apt/sources.list.d/kitware.list > /dev/null

# Add LLVM repository for Clang
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key \
    | gpg --dearmor | sudo tee /usr/share/keyrings/llvm-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/llvm-archive-keyring.gpg] http://apt.llvm.org/$(lsb_release -cs)/ llvm-toolchain-$(lsb_release -cs) main" \
    | sudo tee /etc/apt/sources.list.d/llvm.list > /dev/null

# Install all dependencies
sudo apt-get update
sudo apt-get install -y $CPP_DEPS $PYTHON_DEPS

sudo apt-get full-upgrade -y
sudo apt autoremove -y && sudo apt clean

echo "Installation complete!"