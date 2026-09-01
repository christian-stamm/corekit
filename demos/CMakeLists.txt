cmake_minimum_required(VERSION 3.25)

project(demos
    VERSION 0.0.1
    DESCRIPTION "COREKIT - DEMOS"
    LANGUAGES C CXX ASM
)

add_executable(demo-hello hello.cpp)

target_link_libraries(demo-hello PUBLIC pio_examples pico_stdlib corekit::executor corekit::logger corekit::uartdevice corekit::piodevice corekit::dmadevice)

# target_link_libraries(demo-hello PUBLIC corekit::executor corekit::logger)