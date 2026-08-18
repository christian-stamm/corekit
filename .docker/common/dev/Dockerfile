ARG TOOLCHAIN_IMAGE
FROM ${TOOLCHAIN_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ENV GIT_SSL_NO_VERIFY=1

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends ${SYSTEM_DEPS} && \
    add-apt-repository universe && \
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y --no-install-recommends \
        net-tools \
        usbutils \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY common/dev/requirements.txt .
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"
RUN python3 -m venv ${VIRTUAL_ENV} && \
    ln -sf ${VIRTUAL_ENV}/bin/python3 ${VIRTUAL_ENV}/bin/python && \
    ${VIRTUAL_ENV}/bin/pip install --no-cache-dir -U pip wheel setuptools && \
    ${VIRTUAL_ENV}/bin/pip install --no-cache-dir -Ur requirements.txt
