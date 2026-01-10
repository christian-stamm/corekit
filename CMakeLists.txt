cmake_minimum_required(VERSION 3.19)
project(corekit 
    VERSION 0.0.1
    LANGUAGES CXX
    DESCRIPTION "COREKIT - A C++ Utility Library"
)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(PIOASM_VERSION_STRING "0.1.0")

add_definitions(-DPLATFORM_UNIX) # Define PLATFORM_UNIX / PLATFORM_RTOS

if(DEFINED ${RESSOURCE_DIR})
    message(STATUS "Using cmake defined RESSOURCE_DIR: ${RESSOURCE_DIR}")
elseif(DEFINED $ENV{RESSOURCE_DIR})
    set(RESSOURCE_DIR $ENV{RESSOURCE_DIR})
    message(STATUS "Using environment defined RESSOURCE_DIR: $ENV{RESSOURCE_DIR}")
else()
    set(RESSOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/res")
    message(STATUS "Using default corekit defined RESSOURCE_DIR: ${RESSOURCE_DIR}")
endif()

add_definitions(-DRESSOURCE_DIR="${RESSOURCE_DIR}") # Define RESSOURCE_DIR for ressource path

include(FindPkgConfig)
find_package(glm REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

include(lib/external/fastcdr.cmake)
include(lib/external/threadpool.cmake)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib/glad)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib/mcap)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib/pioasm)

add_library(${PROJECT_NAME} SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/src/device/device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/device/serial.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/logging/logger.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/logging/stream.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/logging/watch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/opengl/context.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/opengl/program.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/opengl/shader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/opengl/texture.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/opengl/window.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/structs/buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core.cpp
)

target_compile_definitions(${PROJECT_NAME} PUBLIC 
    RESSOURCE_DIR=\"${RESSOURCE_DIR}\"
)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${GLM_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PUBLIC
    glad
    mcap
    fastcdr
    Eigen3::Eigen
    dp::thread-pool
    ${OpenCV_LIBS}
    ${GLM_LIBRARIES}

)

