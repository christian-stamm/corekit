cmake_minimum_required(VERSION 3.25)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/helper.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/module.cmake")

add_subdirectory(corekit)
add_subdirectory(demos)
add_subdirectory(thirdparty)

if(COMMAND corekit_bsp_bootstrap)
    corekit_bsp_bootstrap()
endif()

project(corekit
    VERSION 0.0.1
    DESCRIPTION "COREKIT - A C++ Utility Library"
    LANGUAGES C CXX ASM
)

enable_testing()

if(COMMAND corekit_bsp_configure)
    corekit_bsp_configure()
endif()

corekit_build_modules()