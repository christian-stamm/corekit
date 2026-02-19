cmake_minimum_required(VERSION 3.19)

project(corekit 
    VERSION 0.0.1
    LANGUAGES CXX CUDA
    DESCRIPTION "COREKIT - A C++ Utility Library"
)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

set(COREKIT_PLATFORM "UNIX" CACHE STRING "Target platform (UNIX or PICO)")
set(BUILD_COREKIT_CORE TRUE)
set(BUILD_COREKIT_CUDA TRUE)
set(BUILD_COREKIT_OPENGL TRUE)

########################################################################

if(DEFINED RESSOURCE_DIR)
add_compile_definitions(RESSOURCE_DIR="${RESSOURCE_DIR}")
message(STATUS "Defined RESSOURCE_DIR: ${RESSOURCE_DIR}")
endif()

########################################################################

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)