cmake_minimum_required(VERSION 3.25)

add_library(corekit-peripherals-common
    ${CMAKE_CURRENT_SOURCE_DIR}/src/basedevice.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/serialdevice.cpp
)

target_include_directories(corekit-peripherals-common PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(corekit-peripherals-common PUBLIC 
    corekit::concurrency
    corekit::logging
    corekit::system
)

add_library(corekit::peripherals::common ALIAS corekit-peripherals-common)

if(COREKIT_BUILD_TEST AND COREKIT_BUILD_PLATFORM_LINUX)

    find_package(GTest REQUIRED)

    add_executable(corekit-peripherals-common-test
        ${CMAKE_CURRENT_SOURCE_DIR}/test/basedevice.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test/serialdevice.cpp
    )

    target_link_libraries(corekit-peripherals-common-test PUBLIC 
        corekit::peripherals::common
        GTest::gtest_main
    )

    include(GoogleTest)
    gtest_discover_tests(corekit-peripherals-common-test)

endif()