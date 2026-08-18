# docker/Dockerfile.toolchain
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ENV GIT_SSL_NO_VERIFY=1

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libfastcdr-dev \
        libgtest-dev \
        liblz4-dev \
        libzstd-dev && \
    rm -rf /var/lib/apt/lists/*