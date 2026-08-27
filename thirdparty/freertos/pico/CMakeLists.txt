cmake_minimum_required(VERSION 3.25)

set(PICO_SDK_PATH $ENV{PICO_SDK_PATH})
include(${PICO_SDK_PATH}/external/pico_sdk_import.cmake)

inc_subdirs(${CMAKE_CURRENT_SOURCE_DIR} COREKIT_BSP_KEY)