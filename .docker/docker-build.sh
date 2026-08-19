#!/bin/bash
set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

if [[ -z "$BUILD_PLATFORM" ]]; then
    echo "Error: toolchain must be specified via BUILD_PLATFORM." >&2
    echo "Usage: BUILD_PLATFORM=<pico|linux> $0" >&2
    exit 1
fi

echo "Building corekit-base, corekit-tool-${BUILD_PLATFORM}, and corekit-dev images..."

docker compose -f "${SCRIPT_DIR}/docker-compose.yml" build corekit-base
docker compose -f "${SCRIPT_DIR}/docker-compose.yml" build corekit-tool-${BUILD_PLATFORM}
docker compose -f "${SCRIPT_DIR}/docker-compose.yml" build corekit-dev

echo "Build of corekit-base, corekit-tool-${BUILD_PLATFORM}, and corekit-dev images completed successfully."