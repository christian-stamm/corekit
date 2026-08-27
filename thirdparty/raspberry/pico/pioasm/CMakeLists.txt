cmake_minimum_required(VERSION 3.25)
include(${CMAKE_CURRENT_SOURCE_DIR}/helper.cmake)


pioasm_generate_headers(
    TARGET PIO_EXAMPLES
    PROGRAMS
        ${CMAKE_CURRENT_SOURCE_DIR}/programs/shuffler.pio
        ${CMAKE_CURRENT_SOURCE_DIR}/programs/shifter.pio
        ${CMAKE_CURRENT_SOURCE_DIR}/programs/pulser.pio
    FLAGS
        

)