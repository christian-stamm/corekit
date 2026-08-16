cmake_minimum_required(VERSION 3.25)

if(COREKIT_BUILD_PLATFORM_LINUX)

    add_library(corekit-peripherals-platform-linux
        ${CMAKE_CURRENT_SOURCE_DIR}/src/console.cpp
    )

    target_include_directories(corekit-peripherals-platform-linux
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
            $<INSTALL_INTERFACE:include>
    )

    target_link_libraries(corekit-peripherals-platform-linux
        PUBLIC
            corekit::system
            corekit::peripherals::common
    )

    add_library(corekit::peripherals ALIAS corekit-peripherals-platform-linux)

    if(COREKIT_BUILD_TEST AND COREKIT_BUILD_PLATFORM_LINUX)

        find_package(GTest REQUIRED)

        add_executable(corekit-peripherals-platform-linux-test
            ${CMAKE_CURRENT_SOURCE_DIR}/test/console.cpp
        )

        target_link_libraries(corekit-peripherals-platform-linux-test
            PUBLIC
                corekit::peripherals
                GTest::gtest_main
        )

        include(GoogleTest)
        gtest_discover_tests(corekit-peripherals-platform-linux-test)
        
    endif()

endif()