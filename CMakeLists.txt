cmake_minimum_required(VERSION 3.25)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

set(COREKIT_BUILD_GTEST "OFF" CACHE STRING "Build GTest executables")
set(COREKIT_PLATFORM "PICO" CACHE STRING "Target platform")

if (COREKIT_PLATFORM STREQUAL "PICO")
    set(PICO_PLATFORM "rp2040" CACHE STRING "PICO Target platform" FORCE)
    set(PICO_BOARD "pico" CACHE STRING "PICO Target board" FORCE)

    set(PICO_CXX_ENABLE_EXCEPTIONS 1)
    set(PICO_SDK_PATH "${CMAKE_CURRENT_LIST_DIR}/lib/pico-sdk")
    include(${PICO_SDK_PATH}/external/pico_sdk_import.cmake)
endif()

project(corekit 
    VERSION 0.0.1
    LANGUAGES C CXX ASM
    DESCRIPTION "COREKIT - A C++ Utility Library"
)

if(COREKIT_PLATFORM STREQUAL "PICO")
    pico_sdk_init()
endif()

enable_testing()
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)