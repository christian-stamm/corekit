cmake_minimum_required(VERSION 3.25)

add_library(corekit-core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/basedevice.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/serialdevice.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/watch.cpp
)

target_include_directories(corekit-core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(corekit-core PUBLIC
    corekit::platform
)

add_library(corekit::core ALIAS corekit-core)