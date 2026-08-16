cmake_minimum_required(VERSION 3.25)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

set(COREKIT_PLATFORM "PICO" CACHE STRING "Target platform")
set(COREKIT_BUILD_TEST "ON" CACHE STRING "Build Test executables")


if (COREKIT_PLATFORM STREQUAL "PICO")
    set(PICO_BOARD_HEADER_DIRS ${CMAKE_SOURCE_DIR}/targets/pico)

    set(PICO_PLATFORM "rp2350" CACHE STRING "PICO Target platform" FORCE)
    set(PICO_BOARD "bernd" CACHE STRING "PICO Target board")

    set(PICO_CXX_ENABLE_EXCEPTIONS 1)
    set(PICO_SDK_PATH "${CMAKE_CURRENT_LIST_DIR}/lib/pico-sdk")
    include(${PICO_SDK_PATH}/external/pico_sdk_import.cmake)
endif()

if (COREKIT_BUILD_TEST STREQUAL "ON")
    enable_testing()
endif()

if (COREKIT_PLATFORM STREQUAL "POSIX" AND COREKIT_BUILD_TEST STREQUAL "ON")
    set(COREKIT_BUILD_GTEST "ON" CACHE STRING "Build GTest executables")
endif()

if (COREKIT_PLATFORM STREQUAL "PICO" AND COREKIT_BUILD_TEST STREQUAL "ON")
    set(COREKIT_BUILD_PTEST "ON" CACHE STRING "Build Pico Test executables")
endif()


project(corekit 
    VERSION 0.0.1
    LANGUAGES C CXX ASM
    DESCRIPTION "COREKIT - A C++ Utility Library"
)

if(COREKIT_PLATFORM STREQUAL "PICO")
    pico_sdk_init()
endif()


add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/corekit)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty)