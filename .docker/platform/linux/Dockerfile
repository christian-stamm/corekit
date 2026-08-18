# docker/Dockerfile.toolchain
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libfastcdr-dev \
        libgtest-dev \
        liblz4-dev \
        libzstd-dev && \
    rm -rf /var/lib/apt/lists/*