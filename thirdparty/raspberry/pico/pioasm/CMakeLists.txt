cmake_minimum_required(VERSION 3.25)
include(${CMAKE_CURRENT_SOURCE_DIR}/helper.cmake)

pioasm_generate_headers(
    LIB_NAME PIO_EXAMPLES
    IMPORT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/programs
    EXPORT_DIR ${CMAKE_BINARY_DIR}/generated
)

add_library(pio_examples INTERFACE)
target_link_libraries(pio_examples INTERFACE PIO_EXAMPLES)
