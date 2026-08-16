# docker/base/Dockerfile
FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        wget \
        git \
        git-lfs \
        unzip \
        xz-utils \
        build-essential \
        cmake \
        ninja-build \
        pkg-config \
        python3 \
        python3-pip \
        python3-venv \
        gdb \
    && \
    rm -rf /var/lib/apt/lists/*

RUN git lfs install --system