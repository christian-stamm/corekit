cmake_minimum_required(VERSION 3.25)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/core.cmake)

if(EXISTS "${COREKIT_BOOTSTRAP_FILE}")
    include("${COREKIT_BOOTSTRAP_FILE}")
endif()

project(corekit
    VERSION 0.0.1
    DESCRIPTION "COREKIT - A C++ Utility Library"
    LANGUAGES C CXX ASM
)

if(EXISTS ${COREKIT_CONFIGURE_FILE})
    include(${COREKIT_CONFIGURE_FILE})
endif()

add_subdirectory(corekit)
add_subdirectory(thirdparty)