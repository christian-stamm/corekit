cmake_minimum_required(VERSION 3.25)

if(NOT DEFINED COREKIT_PAL)
    message(FATAL_ERROR "COREKIT_PAL is not set. Choose from: stdlib, freertos, baremetal, zephyr, ...")
endif()

if(NOT EXISTS ${COREKIT_PLATFORM_PATH})
    message(FATAL_ERROR "Specified platform '${COREKIT_PAL}' is not supported. Please provide platform implementation at ${COREKIT_PLATFORM_PATH}.")
endif()

message(STATUS "COREKIT_CMAKE_PATH: ${COREKIT_CMAKE_PATH}")
message(STATUS "COREKIT_COMMON_PATH: ${COREKIT_COMMON_PATH}")
message(STATUS "COREKIT_PLATFORM_PATH: ${COREKIT_PLATFORM_PATH}")