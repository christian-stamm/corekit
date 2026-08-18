cmake_minimum_required(VERSION 3.25)

option(
    COREKIT_BUILD_PLATFORM_PICO
    "Build PICO Platform"
    ON
)

set(PICO_CXX_ENABLE_EXCEPTIONS ON)
set(PICO_BOARD_HEADER_DIRS ${CMAKE_SOURCE_DIR}/thirdparty/pico/boards)

include($ENV{PICO_SDK_PATH}/external/pico_sdk_import.cmake)