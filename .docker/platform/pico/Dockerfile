# docker/Dockerfile.toolchain
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ENV GIT_SSL_NO_VERIFY=1

ARG ARM_BUILD_ARCH
ARG ARM_TARGET_ARCH
ARG ARM_TOOLCHAIN_VER

ENV PICO_ENV_ROOT="/opt/pico"
ENV PICO_SDK_PATH="$PICO_ENV_ROOT/pico-sdk"
ENV PICO_OPENOCD_PATH="$PICO_ENV_ROOT/openocd"
ENV PICO_PICOTOOL_PATH="$PICO_ENV_ROOT/picotool"
ENV PICO_DEBUGPROBE_PATH="$PICO_ENV_ROOT/picoprobe"
ENV PICO_TOOLCHAIN_PATH="$PICO_ENV_ROOT/arm-none-eabi"
ENV FREERTOS_KERNEL_PATH="$PICO_ENV_ROOT/freertos"

ENV PATH="${PICO_TOOLCHAIN_PATH}/bin:${PATH}"

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gdb-multiarch \
        libhidapi-dev \
        libhidapi-hidraw0 \
        libjim-dev \
        libusb-1.0-0 \
        libusb-1.0-0-dev \
        tar \
        udev \
        wget \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Arm GCC
WORKDIR "$PICO_TOOLCHAIN_PATH"
ARG ARM_TOOLCHAIN_PREFIX="https://gitlab.arm.com/api/v4/projects/tooling%2Fgnu-toolchains-for-arm/packages/generic/gnu-toolchain"
ARG ARM_TOOLCHAIN_FILE="arm-gnu-toolchain-${ARM_TOOLCHAIN_VER}-${ARM_BUILD_ARCH}-${ARM_TARGET_ARCH}.tar.xz"
ARG ARM_TOOLCHAIN_URL="${ARM_TOOLCHAIN_PREFIX}/${ARM_TOOLCHAIN_VER}/${ARM_TOOLCHAIN_FILE}"
RUN wget --no-check-certificate $ARM_TOOLCHAIN_URL && tar xvf $ARM_TOOLCHAIN_FILE -C $PICO_TOOLCHAIN_PATH --strip-components=1 
# COPY ./platform/pico/$ARM_TOOLCHAIN_FILE .
# RUN tar -xJf $ARM_TOOLCHAIN_FILE && rm $ARM_TOOLCHAIN_FILE
ENV PATH="${PICO_TOOLCHAIN_PATH}/bin:${PATH}"

# Install SDK
WORKDIR "$PICO_SDK_PATH"
RUN git clone --depth=1 --shallow-submodules --recurse-submodules https://github.com/raspberrypi/pico-sdk.git $PICO_SDK_PATH

# Install OpenOCD
WORKDIR "$PICO_OPENOCD_PATH"
ARG OPENOCD_BUILD_PARAMS="--enable-ftdi --enable-sysfsgpio --enable-picoprobe --enable-cmsis-dap"
RUN git clone --depth=1 --shallow-submodules --recurse-submodules https://github.com/raspberrypi/openocd.git $PICO_OPENOCD_PATH
RUN ./bootstrap && ./configure $OPENOCD_BUILD_PARAMS && make install && \
    cp "./contrib/60-openocd.rules" "/etc/udev/rules.d/"

# Install Picotool
WORKDIR "$PICO_PICOTOOL_PATH"
RUN git clone --depth=1 --shallow-submodules --recurse-submodules https://github.com/raspberrypi/picotool.git $PICO_PICOTOOL_PATH
RUN mkdir -p build && cd build && cmake ../ && make install && cd ../ && \
    cp "./udev/60-picotool.rules" "/etc/udev/rules.d/"

# Install Picoprobe
WORKDIR "$PICO_DEBUGPROBE_PATH"
RUN git clone --depth=1 --shallow-submodules --recurse-submodules https://github.com/raspberrypi/debugprobe.git $PICO_DEBUGPROBE_PATH
RUN mkdir -p build-rp2040 && cd build-rp2040 && cmake -DDEBUG_ON_PICO=1 -DPICO_BOARD=pico -DPICO_PLATFORM=rp2040 ../ && make
RUN mkdir -p build-rp2350 && cd build-rp2350 && cmake -DDEBUG_ON_PICO=1 -DPICO_BOARD=pico2 -DPICO_PLATFORM=rp2350 ../ && make

# Install FreeRTOS
WORKDIR "$FREERTOS_KERNEL_PATH"
RUN git clone --depth=1 --shallow-submodules --recurse-submodules https://github.com/raspberrypi/FreeRTOS-Kernel.git $FREERTOS_KERNEL_PATH
