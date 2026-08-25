cmake_minimum_required(VERSION 3.25)

set(COREKIT_WORKDIR ${CMAKE_CURRENT_SOURCE_DIR})
set(COREKIT_CMAKE_PATH "${COREKIT_WORKDIR}/cmake")

include(${COREKIT_CMAKE_PATH}/helper.cmake)
include(${COREKIT_CMAKE_PATH}/module.cmake)