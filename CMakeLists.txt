cmake_minimum_required(VERSION 3.25)
set(COREKIT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

if(EXISTS ${COREKIT_PAL_FILE})
    include(${COREKIT_PAL_FILE})
endif()

if(EXISTS ${COREKIT_BSP_FILE})
    include(${COREKIT_BSP_FILE})
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/helper.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/module.cmake")

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

enable_testing()
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/helper.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/module.cmake")

add_subdirectory(corekit)
add_subdirectory(demos)
add_subdirectory(thirdparty)

corekit_build_modules()