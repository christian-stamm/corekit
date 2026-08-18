cmake_minimum_required(VERSION 3.25)

option(
    COREKIT_BUILD_BSP_PICO
    "Build PICO Platform"
    ON
)

set(PICO_CXX_ENABLE_EXCEPTIONS ON)
set(PICO_BOARD_HEADER_DIRS ${COREKIT_WORKDIR}/thirdparty/raspberry/pico/boards)

if (NOT EXISTS $ENV{PICO_SDK_PATH})
    message(FATAL_ERROR "No valid PICO_SDK_PATH is set. Scanned path '$ENV{PICO_SDK_PATH}'.")
endif()

if(NOT DEFINED PICO_PLATFORM)
    message(FATAL_ERROR "PICO_PLATFORM is not set. Please set PICO_PLATFORM to <rp2040,rp2350-arm-s,risc-v>.")
endif()

if(NOT DEFINED PICO_BOARD)
    message(FATAL_ERROR "PICO_BOARD is not set. Please set PICO_BOARD to the desired board. <pico,pico_w,...>.")
endif()


set(PICO_SDK_PATH $ENV{PICO_SDK_PATH})
include("${PICO_SDK_PATH}/external/pico_sdk_import.cmake")