cmake_minimum_required(VERSION 3.25)

option(
    COREKIT_BUILD_PLATFORM_LINUX
    "Build Linux Platform"
    ON
)

include("${LINUX_PORTABLE_IMPL_DIR}/bootstrap.cmake")