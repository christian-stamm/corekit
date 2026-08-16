cmake_minimum_required(VERSION 3.25)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

set(COREKIT_WORKDIR ${CMAKE_CURRENT_SOURCE_DIR})
set(COREKIT_CMAKE_OS_FILE "${COREKIT_WORKDIR}/cmake/os/${COREKIT_OS_NAME}.cmake")
set(COREKIT_CMAKE_BSP_PATH "${COREKIT_WORKDIR}/cmake/bsp/${COREKIT_BSP_PATH}")

if(NOT DEFINED COREKIT_OS_NAME)
    message(FATAL_ERROR "COREKIT_OS_NAME is not set")
endif()

if(EXISTS ${COREKIT_CMAKE_OS_FILE})
    include(${COREKIT_CMAKE_OS_FILE})
else()
    message(FATAL_ERROR "Unsupported COREKIT_BASE_OS: ${COREKIT_BASE_OS}. File not found: ${COREKIT_CMAKE_OS_FILE}")
endif()
    
if(NOT EXISTS ${COREKIT_CMAKE_BSP_PATH})
    message(FATAL_ERROR "Unsupported COREKIT_CMAKE_BSP_PATH: ${COREKIT_BSP_PATH}.")
endif()

if(NOT DEFINED COREKIT_CTEST_ENABLED)
    message(WARNING "COREKIT_CTEST_ENABLED is not set. Disabling tests.")
    option(COREKIT_CTEST_ENABLED "Build Tests" OFF)
endif()

if (COREKIT_CTEST_ENABLED)
    enable_testing()
endif()

message(STATUS "COREKIT_BASE_OS: ${COREKIT_BASE_OS}")
message(STATUS "COREKIT_CTEST_ENABLED: ${COREKIT_CTEST_ENABLED}")
message(STATUS "COREKIT_CMAKE_OS_FILE: ${COREKIT_CMAKE_OS_FILE}")
message(STATUS "COREKIT_CMAKE_BSP_PATH: ${COREKIT_CMAKE_BSP_PATH}")

if(COMMAND corekit_bootstrap)
    corekit_bootstrap("${COREKIT_CMAKE_BSP_PATH}/bootstrap.cmake")
endif()

project(corekit
    VERSION 0.0.1
    DESCRIPTION "COREKIT - A C++ Utility Library"
    LANGUAGES C CXX ASM
)

if(COMMAND corekit_configure)
    corekit_configure("${COREKIT_CMAKE_BSP_PATH}/configure.cmake")
endif()

add_subdirectory(corekit)
add_subdirectory(thirdparty)