cmake_minimum_required(VERSION 3.25)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/env.cmake")

add_subdirectory(corekit)
add_subdirectory(thirdparty)

if(COMMAND corekit_runtime_bootstrap)
    corekit_runtime_bootstrap()
endif()

project(corekit
    VERSION 0.0.1
    DESCRIPTION "COREKIT - A C++ Utility Library"
    LANGUAGES C CXX ASM
)

if(COMMAND corekit_runtime_configure)
    corekit_runtime_configure()
endif()

enable_testing()
corekit_finalize()