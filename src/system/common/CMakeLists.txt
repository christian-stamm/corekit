cmake_minimum_required(VERSION 3.25)

if(COREKIT_PLATFORM STREQUAL "POSIX")

    add_library(corekit-system-common INTERFACE)

    target_include_directories(corekit-system-common
        INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
            $<INSTALL_INTERFACE:include>
    )

    add_library(corekit::system::common ALIAS corekit-system-common)

    add_executable(corekit-system-core-test
        ${CMAKE_CURRENT_SOURCE_DIR}/src/concepts.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test/atomic.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test/mutex.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test/semphr.cpp
    )

    target_link_libraries(corekit-system-core-test PUBLIC 
        corekit::system
        gtest_main
    )

    include(GoogleTest)
    gtest_discover_tests(corekit-system-core-test)

endif()