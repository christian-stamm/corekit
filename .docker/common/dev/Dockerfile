FROM ubuntu:noble

ENV DEBIAN_FRONTEND=noninteractive

ARG HOST_ARCHITECTURE="x86_64"
ARG ARM_TOOLCHAIN_VER="14.2.rel1"

ENV VIRTUAL_ENV="/opt/venv"
ARG ENV_ROOT="/opt/pico"

ENV PICO_ARM_PATH="$ENV_ROOT/arm-none-eabi"
ENV PICO_SDK_PATH="$ENV_ROOT/pico-sdk"
ENV PICO_DEBUG_PATH="$ENV_ROOT/pico-debug"
ENV PICO_PTOOL_PATH="$ENV_ROOT/pico-tool"
ENV PICO_PROBE_PATH="$ENV_ROOT/pico-probe"
ENV FREERTOS_KERNEL_PATH="$ENV_ROOT/pico-rtos"
ENV PATH="$PATH:$VIRTUAL_ENV/bin"

# Install dependencies

ARG SYSTEM_DEPS="software-properties-common ca-certificates locales unzip curl wget"
ARG IFACE_DEPS="net-tools usbutils udev"
ARG BUILD_DEPS="build-essential cmake gdb git git-lfs libtool autoconf automake texinfo"
ARG DEVEL_DEPS="curl wget pkg-config libtool autoconf automake texinfo"
ARG PYTHON_DEPS="python3 python3-pip python3-venv"
ARG CPP_DEPS="libgtest-dev"
ARG PICO_DEPS="gcc-arm-none-eabi binutils-arm-none-eabi libnewlib-arm-none-eabi libstdc++-arm-none-eabi-newlib gdb-multiarch"
ARG OPENOCD_DEPS="libtool autoconf automake texinfo libhidapi-hidraw0 libhidapi-dev libusb-1.0-0 libusb-1.0-0-dev"



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
        ${OPENOCD_DEPS} \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY env/requirements.txt .
RUN python3 -m venv /opt/venv && \
    ln -sf /opt/venv/bin/python3 /opt/venv/bin/python && \
    /opt/venv/bin/pip install --no-cache-dir -U pip wheel setuptools && \
    /opt/venv/bin/pip install --no-cache-dir -Ur requirements.txt

# Install Arm GCC
WORKDIR "$PICO_ARM_PATH"
ARG ARM_DEPS="wget tar xz-utils"
ARG ARM_BASE_URL="https://developer.arm.com/-/media/Files/downloads/gnu"
ARG ARM_FILE_NAME="arm-gnu-toolchain-$ARM_TOOLCHAIN_VER-$HOST_ARCHITECTURE-arm-none-eabi.tar.xz"
ARG ARM_GCC_URL="$ARM_BASE_URL/$ARM_TOOLCHAIN_VER/binrel/$ARM_FILE_NAME"
RUN apt update && apt install -y $ARM_DEPS
RUN wget $ARM_GCC_URL && tar xvf $ARM_FILE_NAME -C $PICO_ARM_PATH --strip-components=1
RUN ln -s $PICO_ARM_PATH/bin/arm-none-eabi-gcc /usr/bin/arm-none-eabi-gcc && \
    ln -s $PICO_ARM_PATH/bin/arm-none-eabi-g++ /usr/bin/arm-none-eabi-g++ && \
    ln -s $PICO_ARM_PATH/bin/arm-none-eabi-gdb /usr/bin/arm-none-eabi-gdb && \
    ln -s $PICO_ARM_PATH/bin/arm-none-eabi-size /usr/bin/arm-none-eabi-size && \
    ln -s $PICO_ARM_PATH/bin/arm-none-eabi-objdump /usr/bin/arm-none-eabi-objdump && \
    ln -s $PICO_ARM_PATH/bin/arm-none-eabi-objcopy /usr/bin/arm-none-eabi-objcopy

# Install SDK
ARG SDK_DEPS="git wget curl build-essential gcc g++ make cmake pkg-config"
RUN apt update && apt install -y $SDK_DEPS
RUN git clone https://github.com/raspberrypi/pico-sdk.git $PICO_SDK_PATH --recurse-submodules

# Install OpenOCD
WORKDIR "$PICO_DEBUG_PATH"
ARG OPENOCD_DEPS="libtool libjim-dev autoconf automake texinfo libhidapi-hidraw0 libhidapi-dev libusb-1.0-0 libusb-1.0-0-dev udev"
ARG OPENOCD_PARAMS="--enable-ftdi --enable-sysfsgpio --enable-picoprobe --enable-cmsis-dap"
RUN apt update && apt install -y $OPENOCD_DEPS
RUN git clone https://github.com/raspberrypi/openocd.git $PICO_DEBUG_PATH --recurse-submodules 
RUN ./bootstrap && ./configure $OPENOCD_PARAMS && make -j 8 install && \
    cp "./contrib/60-openocd.rules" "/etc/udev/rules.d/"

# # Install Picotool
# WORKDIR "$PICO_PTOOL_PATH"
# ARG PICOTOOL_DEPS="libusb-1.0-0 libusb-1.0-0-dev udev"
# RUN apt update && apt install -y $PICOTOOL_DEPS
# RUN git clone https://github.com/raspberrypi/picotool.git $PICO_PTOOL_PATH --recurse-submodules
# RUN mkdir -p build && cd build && cmake ../ && make -j 8 install && cd ../ && \
#     cp "./udev/99-picotool.rules" "/etc/udev/rules.d/"

# # Install Picoprobe
# WORKDIR "$PICO_PROBE_PATH"
# RUN git clone https://github.com/raspberrypi/debugprobe.git $PICO_PROBE_PATH --recurse-submodules
# RUN mkdir -p build-rp2040 && cd build-rp2040 && cmake -DDEBUG_ON_PICO=1 -DPICO_BOARD=pico -DPICO_PLATFORM=rp2040 ../ && make -j 8
# RUN mkdir -p build-rp2350 && cd build-rp2350 && cmake -DDEBUG_ON_PICO=1 -DPICO_BOARD=pico2 -DPICO_PLATFORM=rp2350 ../ && make -j 8

# Install FreeRTOS
RUN git clone https://github.com/raspberrypi/FreeRTOS-Kernel.git $FREERTOS_KERNEL_PATH --recurse-submodules
