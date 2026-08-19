cmake_minimum_required(VERSION 3.25)

add_library(corekit-concurrency-api INTERFACE)

target_include_directories(corekit-concurrency-api
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
        $<INSTALL_INTERFACE:include>
)

add_library(corekit::concurrency::api ALIAS corekit-concurrency-api)

add_library(corekit-concurrency 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/check.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/task.cpp
)

target_link_libraries(corekit-concurrency
    PUBLIC
        corekit::concurrency::api
        corekit::concurrency::pal
)

add_library(corekit::concurrency ALIAS corekit-concurrency)

find_package(GTest QUIET)

if(COREKIT_BUILD_TEST AND GTest_FOUND)

    include(GoogleTest)

    add_executable(corekit-concurrency-test
        ${CMAKE_CURRENT_SOURCE_DIR}/test/atomic.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test/mutex.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test/smphr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test/rtoshook.cpp
    )

    target_link_libraries(corekit-concurrency-test PUBLIC 
        corekit::concurrency
        GTest::gtest_main
    )

    gtest_discover_tests(corekit-concurrency-test)

endif()