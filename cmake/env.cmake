cmake_minimum_required(VERSION 3.25)

set(COREKIT_WORKDIR ${CMAKE_CURRENT_SOURCE_DIR})
set(COREKIT_CMAKE_PATH "${COREKIT_WORKDIR}/cmake")
set(COREKIT_RUNTIME_PATH "${COREKIT_WORKDIR}/corekit/runtime/${COREKIT_RUNTIME}")

if(NOT DEFINED COREKIT_RUNTIME)
    message(FATAL_ERROR "COREKIT_RUNTIME is not set. Choose from: stdlib, freertos, baremetal, zephyr, ...")
endif()

if(NOT EXISTS ${COREKIT_RUNTIME_PATH})
    message(FATAL_ERROR "Specified runtime '${COREKIT_RUNTIME}' is not supported. Please provide runtime implementation at ${COREKIT_RUNTIME_PATH}.")
endif()

include(${COREKIT_CMAKE_PATH}/helper.cmake)
include(${COREKIT_CMAKE_PATH}/module.cmake)