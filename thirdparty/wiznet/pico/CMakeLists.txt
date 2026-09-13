cmake_minimum_required(VERSION 3.25)

if(NOT BUILD_RP2350_MODULES)
    return()
endif()

set(PORT_DIR ${CMAKE_CURRENT_LIST_DIR})

pioasm_generate_headers(
        LIB_NAME    wiznet-qspi-pio
        IMPORT_DIR  ${PORT_DIR}
        EXPORT_DIR  ${PORT_DIR}
)

add_library(wiznet-inet STATIC
        ${PORT_DIR}/src/wizchip_spi.c
        ${PORT_DIR}/src/wizchip_qspi_pio.c
)

target_compile_definitions(wiznet-inet PUBLIC
        DEVICE_BOARD_NAME=W6300_EVB_PICO2
        _WIZCHIP_QSPI_MODE_=QSPI_QUAD_MODE
)

target_include_directories(wiznet-inet PUBLIC
        ${PORT_DIR}/inc
        ${PORT_DIR}/pioasm/
)

target_link_libraries(wiznet-inet PUBLIC
        ETHERNET_FILES
        wiznet-qspi-pio
        pico_stdlib
        hardware_pio
        hardware_spi
        hardware_dma
        hardware_clocks
)
