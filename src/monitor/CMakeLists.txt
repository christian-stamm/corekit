cmake_minimum_required(VERSION 3.25)

add_library(corekit-monitor
    ${CMAKE_CURRENT_SOURCE_DIR}/src/watch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/logger.cpp
)

target_include_directories(corekit-monitor
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(corekit-monitor
    PUBLIC
        corekit::system
        corekit::devices
)

add_library(corekit::monitor ALIAS corekit-monitor)

add_executable(corekit-monitor-test
    ${CMAKE_CURRENT_SOURCE_DIR}/test/watch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test/logger.cpp
)

target_link_libraries(corekit-monitor-test
    PUBLIC
        corekit::monitor
        gtest_main
)