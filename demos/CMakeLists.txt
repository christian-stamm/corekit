cmake_minimum_required(VERSION 3.25)

project(demos
    VERSION 0.0.1
    DESCRIPTION "COREKIT - DEMOS"
    LANGUAGES C CXX ASM
)

add_executable(demo-hello hello.cpp)

target_link_libraries(demo-hello PRIVATE corekit::uartdevice corekit::logger corekit::executor pico_stdlib)