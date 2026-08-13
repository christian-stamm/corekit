cmake_minimum_required(VERSION 3.25)

if(COREKIT_PLATFORM STREQUAL "POSIX")

    add_library(corekit-devices-posix
        ${CMAKE_CURRENT_SOURCE_DIR}/src/console.cpp
    )

    target_include_directories(corekit-devices-posix
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
            $<INSTALL_INTERFACE:include>
    )

    target_link_libraries(corekit-devices-posix
        PUBLIC
            corekit::system
            corekit::devices::common
    )

    add_library(corekit::devices ALIAS corekit-devices-posix)

    add_library(corekit-devices-posix-test
        ${CMAKE_CURRENT_SOURCE_DIR}/test/console.cpp
    )

    target_link_libraries(corekit-devices-posix-test
        PUBLIC
            corekit::devices
            gtest_main
    )

endif()