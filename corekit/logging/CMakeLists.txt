cmake_minimum_required(VERSION 3.25)

add_library(corekit-logging
    ${CMAKE_CURRENT_SOURCE_DIR}/src/watch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/logger.cpp
)

target_include_directories(corekit-logging
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(corekit-logging
    PUBLIC
        corekit::concurrency
        corekit::peripherals
)

add_library(corekit::logging ALIAS corekit-logging)

find_package(GTest REQUIRED)

if(COREKIT_BUILD_TEST AND GTest_FOUND)

    include(GoogleTest)

    add_executable(corekit-logging-test
        ${CMAKE_CURRENT_SOURCE_DIR}/test/watch.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test/logger.cpp
    )

    target_link_libraries(corekit-logging-test
        PUBLIC
            corekit::logging
            GTest::gtest_main
    )

    gtest_discover_tests(corekit-logging-test)

endif()