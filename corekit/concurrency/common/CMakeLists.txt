cmake_minimum_required(VERSION 3.25)

add_library(corekit-concurrency-common INTERFACE)

target_include_directories(corekit-concurrency-common
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
        $<INSTALL_INTERFACE:include>
)

add_library(corekit::concurrency::common ALIAS corekit-concurrency-common)

if(COREKIT_BUILD_TEST AND COREKIT_BUILD_PLATFORM_LINUX)

    find_package(GTest REQUIRED)
    
    add_executable(corekit-concurrency-test
        ${CMAKE_CURRENT_SOURCE_DIR}/src/concepts.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test/atomic.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test/mutex.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test/semphr.cpp
    )

    target_link_libraries(corekit-concurrency-test PUBLIC 
        corekit::concurrency
        GTest::gtest_main
    )

    include(GoogleTest)
    gtest_discover_tests(corekit-concurrency-test)

endif()