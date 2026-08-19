# docker/base/Dockerfile
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ENV GIT_SSL_NO_VERIFY=1

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8
    
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        autoconf \
        automake \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        gdb \
        gdb-multiarch \
        git \
        git-lfs \
        libgtest-dev \
        libtool \
        locales \
        ninja-build \
        pkg-config \
        python3 \
        python3-pip \
        python3-venv \
        software-properties-common \
        texinfo \
        unzip \
        wget \
        xz-utils \
    && \
    rm -rf /var/lib/apt/lists/*

RUN git lfs install --system