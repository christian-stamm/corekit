cmake_minimum_required(VERSION 3.25)

if(NOT DEFINED COREKIT_BUILD_TEST)
    message(WARNING "COREKIT_BUILD_TEST is not set. Disabling tests.")
    option(COREKIT_BUILD_TEST "Build Tests" OFF)
endif()

if (COREKIT_BUILD_TEST)
    enable_testing()
endif()

message(STATUS "COREKIT_BUILD_TEST: ${COREKIT_BUILD_TEST}")
