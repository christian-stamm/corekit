cmake_minimum_required(VERSION 3.25)

enable_testing()
add_compile_options(-Wno-psabi)

set(COREKIT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

if(EXISTS ${COREKIT_PAL_FILE})
    include(${COREKIT_PAL_FILE})
endif()

if(EXISTS ${COREKIT_BSP_FILE})
    include(${COREKIT_BSP_FILE})
endif()

include("${COREKIT_ROOT}/cmake/helper.cmake")
include("${COREKIT_ROOT}/cmake/module.cmake")

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

add_subdirectory(corekit)
add_subdirectory(thirdparty)

corekit_build_modules()