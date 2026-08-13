cmake_minimum_required(VERSION 3.25)

add_library(corekit-devices-common
    ${CMAKE_CURRENT_SOURCE_DIR}/src/basedevice.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/serialdevice.cpp
)

target_include_directories(corekit-devices-common PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(corekit-devices-common PUBLIC 
    corekit::system
    corekit::monitor
)

add_library(corekit::devices::common ALIAS corekit-devices-common)

add_library(corekit-devices-common-test
    ${CMAKE_CURRENT_SOURCE_DIR}/test/device.cpp
)

target_link_libraries(corekit-devices-common-test PUBLIC 
    corekit::devices::common
    gtest_main
)