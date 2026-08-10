cmake_minimum_required(VERSION 3.25)

add_executable(corekit-test-bench
    ${CMAKE_CURRENT_SOURCE_DIR}/src/queue.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/task.cpp
)

target_link_libraries(corekit-test-bench
    PRIVATE
        corekit::core
        gtest_main
)