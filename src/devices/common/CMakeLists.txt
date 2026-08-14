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

if(COREKIT_BUILD_GTEST STREQUAL "ON")

    add_executable(corekit-devices-common-test
        ${CMAKE_CURRENT_SOURCE_DIR}/test/basedevice.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test/serialdevice.cpp
    )

    target_link_libraries(corekit-devices-common-test PUBLIC 
        corekit::devices
        gtest_main
    )

    include(GoogleTest)
    gtest_discover_tests(corekit-devices-common-test)

endif()