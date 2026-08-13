cmake_minimum_required(VERSION 3.25)

project(corekit 
    VERSION 0.0.1
    LANGUAGES CXX
    DESCRIPTION "COREKIT - A C++ Utility Library"
)

enable_testing()
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

set(COREKIT_PLATFORM "POSIX" CACHE STRING "Target platform")

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)