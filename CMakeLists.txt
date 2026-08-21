cmake_minimum_required(VERSION 3.25)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

set(COREKIT_WORKDIR ${CMAKE_CURRENT_SOURCE_DIR})
set(COREKIT_CMAKE_PATH "${COREKIT_WORKDIR}/cmake")
set(COREKIT_COMMON_PATH "${COREKIT_WORKDIR}/corekit/common")
set(COREKIT_PLATFORM_PATH "${COREKIT_WORKDIR}/corekit/platform/${COREKIT_PAL}")

include("${COREKIT_CMAKE_PATH}/module.cmake")
include("${COREKIT_CMAKE_PATH}/path.cmake")
include("${COREKIT_CMAKE_PATH}/test.cmake")

project(corekit
    VERSION 0.0.1
    DESCRIPTION "COREKIT - A C++ Utility Library"
    LANGUAGES C CXX ASM
)

add_subdirectory(corekit)
add_subdirectory(thirdparty)

corekit_finalize()