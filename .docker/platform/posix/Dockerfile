# docker/Dockerfile.toolchain
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgtest-dev && \ 
    rm -rf /var/lib/apt/lists/*