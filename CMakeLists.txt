cmake_minimum_required(VERSION 3.19)
project(corekit 
    VERSION 0.0.1
    LANGUAGES CXX
    DESCRIPTION "COREKIT - A C++ Utility Library"
)

# Permit dependencies that still ship configs with pre-3.5 cmake_minimum_required
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(PIOASM_VERSION_STRING "0.1.0")
set(PLATFORM_UNIX TRUE)
set(PLATFORM_PICO FALSE)
set(BUILD_TESTS TRUE)

########################################################################

if(RESSOURCE_DIR)
    message(STATUS "Using cmake defined RESSOURCE_DIR: ${RESSOURCE_DIR}")
elseif(DEFINED $ENV{RESSOURCE_DIR})
    set(RESSOURCE_DIR $ENV{RESSOURCE_DIR})
    message(STATUS "Using environment defined RESSOURCE_DIR: $ENV{RESSOURCE_DIR}")
else()
    set(RESSOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/res")
    message(STATUS "Using default corekit defined RESSOURCE_DIR: ${RESSOURCE_DIR}")
endif()

add_definitions(-DRESSOURCE_DIR="${RESSOURCE_DIR}") # Define RESSOURCE_DIR for ressource path

########################################################################

include(FindPkgConfig)
find_package(glm REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

include(lib/external/fastcdr.cmake)
include(lib/external/threadpool.cmake)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib/glad)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib/mcap)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib/pioasm)

########################################################################

add_library(${PROJECT_NAME} SHARED)

target_compile_definitions(${PROJECT_NAME} PUBLIC 
    RESSOURCE_DIR=\"${RESSOURCE_DIR}\"
)

if(PLATFORM_UNIX)
    message(STATUS "Building for Platform UNIX")
    target_compile_definitions(${PROJECT_NAME} PUBLIC PLATFORM_UNIX)
elseif(PLATFORM_PICO)
    message(STATUS "Building for Platform PICO")
    target_compile_definitions(${PROJECT_NAME} PUBLIC PLATFORM_PICO)
endif()

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${GLM_INCLUDE_DIRS}
)

target_sources(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/device/driver/serial.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/device/device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/render/opengl/program.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/render/opengl/shader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/render/opengl/texture.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/render/opengl/window.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/system/concurrency/mutex.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/system/concurrency/smphr.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/system/concurrency/thread.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/system/diagnostics/assert.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/system/diagnostics/logger.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/system/diagnostics/stream.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/system/diagnostics/watch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/system/system.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/utils/memory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/utils/math.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/utils/file.cpp
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


########################################################################

if(BUILD_TESTS)
    include(lib/external/gtest.cmake)

    enable_testing()

    add_executable(${PROJECT_NAME}_check 
        ${CMAKE_CURRENT_SOURCE_DIR}/test/core.cpp
    )

    target_include_directories(${PROJECT_NAME}_check PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/test
    )

    target_link_libraries(${PROJECT_NAME}_check PRIVATE
        ${PROJECT_NAME}
        gtest
        gtest_main
    )

    include(GoogleTest)
    gtest_discover_tests(${PROJECT_NAME}_check DISCOVERY_MODE PRE_TEST)

endif()