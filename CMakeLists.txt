cmake_minimum_required(VERSION 3.25)

set(COREKIT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

include("${COREKIT_ROOT}/cmake/helper.cmake")
include("${COREKIT_ROOT}/cmake/module.cmake")
include("${COREKIT_ROOT}/cmake/pal/${COREKIT_PAL_FILE}")
include("${COREKIT_ROOT}/cmake/bsp/${COREKIT_BSP_FILE}")

if(COMMAND corekit_bsp_bootstrap)
    corekit_bsp_bootstrap()
endif()

project(corekit
    VERSION 0.0.1
    DESCRIPTION "COREKIT - A C++ Utility Library"
    LANGUAGES C CXX ASM
)

if(COMMAND corekit_bsp_launch)
    corekit_bsp_launch()
endif()

add_subdirectory(thirdparty)
add_subdirectory(corekit)

corekit_build_modules()